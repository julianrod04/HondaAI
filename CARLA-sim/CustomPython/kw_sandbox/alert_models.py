"""alert_models.py

Outputs a control/presentation vector for the graphical alert system shown
to the participant hero-car driver. The vector encodes:

  Shared parameters (always present):
    location  – 0: control-panel,  1: windshield projection
    color     – 0: standard RGB,   1: colorblind-friendly palette
    vibration – 0: no haptic,      1: steering-wheel vibration
    lag       – seconds of lookahead (0 = current timestep, up to MAX_LAG)

  GUI-type selection (one of 3):
    0  arrow   – directional arrow to AV's future position (lag seconds ahead)
    1  route   – full AV trajectory rendered as a path overlay
    2  sound   – audio "left"/"right" cue based on lateral offset at lag seconds

  GUI-specific parameters (3 floats in [0,1], meaning depends on gui_type):
    arrow   [scale, opacity, vibration_dist]
    route   [line_width, opacity, vibration_dist]
    sound   [lateral_threshold, cooldown, volume]

Two models are provided:

  GaussianAlertModel
    Reward-weighted kernel density approach (no neural network). Maintains a
    replay buffer of (state, alert_vector, score) triples. New alerts are
    sampled from a Gaussian mixture where each component is centred on a past
    alert vector and weighted by state-kernel similarity × score value.
    Analogous to reward-weighted regression / kernel herding.

  MoEAlertModel
    A two-level mixture-of-experts MLP.
      Layer 1 – GatingNetwork   → Categorical(3)     which GUI type to show
      Layer 2 – ExpertNetwork × 3, one per GUI type → gui_params distribution
      Shared  – SharedHead      → location / color / vibration / lag
    Trained with a PPO-style clipped surrogate loss using driver-deviation
    scores as the advantage signal. Old policy log-probs are stored at sample
    time so that the importance-sampling ratio r = π_new / π_old can be
    computed without needing ground-truth labels.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Categorical, Normal


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

NUM_GUI_TYPES = 3
GUI_PARAM_DIM = 3    # number of continuous params per GUI type
SHARED_PARAM_DIM = 4 # location, color, vibration, lag
ALERT_DIM = SHARED_PARAM_DIM + 1 + GUI_PARAM_DIM  # 8 total raw floats
MAX_LAG = 2.0        # maximum lookahead in seconds

# Default state: AI car (dx, dy, heading, speed, steer, throttle) = 6
#                hero car (speed, heading, steer) = 3
#                relative (distance, bearing, rel_speed) = 3
#                route (angle_to_wp, progress) = 2  → total 14
DEFAULT_STATE_DIM = 14

# Three alert types rendered by carla_alert_output.py:
#   0  arrow  – directional arrow to AV's future position (lag seconds ahead)
#               uses: location, color, vibration, lag
#               params: [scale, opacity, vibration_dist]
#   1  route  – full AV trajectory rendered as a path overlay
#               uses: location, color, vibration  (no lag — shows whole route)
#               params: [line_width, opacity, vibration_dist]
#   2  sound  – audio "left"/"right" cue based on lateral offset at lag seconds
#               uses: lag  (no location, no color — audio only)
#               params: [lateral_threshold, cooldown, volume]
GUI_TYPES: Dict[int, str] = {
    0: "arrow",
    1: "route",
    2: "sound",
}

# Number of independent distribution components in the joint log-prob.
# Dividing the total log-prob by this constant keeps PPO ratios
# exp(new/K - old/K) in a stable range even when individual components drift.
_LOG_PROB_COMPONENTS = 1 + GUI_PARAM_DIM + (SHARED_PARAM_DIM - 1) + 1  # gui + params + discretes + lag

GUI_PARAM_NAMES: Dict[int, List[str]] = {
    0: ["scale", "opacity", "vibration_dist"],
    1: ["line_width", "opacity", "vibration_dist"],
    2: ["lateral_threshold", "cooldown", "volume"],
}


# ─────────────────────────────────────────────────────────────────────────────
# AlertVector — the output type both models return
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AlertVector:
    """Decoded, human-readable alert output vector.

    Construct via ``AlertVector.from_raw(raw_vec)`` or directly.
    ``to_raw()`` returns the 8-float numpy array for model I/O.
    """

    location: int           # 0 = control-panel, 1 = windshield
    color: int              # 0 = standard RGB,  1 = colorblind-friendly
    vibration: int          # 0 = no haptic,     1 = steering-wheel vibration
    lag: float              # seconds ahead in [0, MAX_LAG]
    gui_type: int           # integer in [0, NUM_GUI_TYPES)
    gui_params: np.ndarray  # 3 floats in [0, 1]

    # raw vector layout:
    # [location, color, vibration, lag, gui_type, p0, p1, p2]

    @property
    def gui_name(self) -> str:
        return GUI_TYPES[self.gui_type]

    @property
    def param_names(self) -> List[str]:
        return GUI_PARAM_NAMES[self.gui_type]

    def to_raw(self) -> np.ndarray:
        """Return 8-float numpy array (location, color, vib, lag, gui, p0-p2)."""
        return np.array(
            [
                float(self.location),
                float(self.color),
                float(self.vibration),
                float(self.lag),
                float(self.gui_type),
                *self.gui_params.tolist(),
            ],
            dtype=np.float32,
        )

    @staticmethod
    def from_raw(v: np.ndarray) -> "AlertVector":
        """Decode an 8-float raw vector (thresholds discrete values)."""
        v = np.asarray(v, dtype=np.float32)
        return AlertVector(
            location=int(round(float(np.clip(v[0], 0, 1)))),
            color=int(round(float(np.clip(v[1], 0, 1)))),
            vibration=int(round(float(np.clip(v[2], 0, 1)))),
            lag=float(np.clip(v[3], 0.0, MAX_LAG)),
            gui_type=int(round(float(np.clip(v[4], 0, NUM_GUI_TYPES - 1)))),
            gui_params=np.clip(v[5:8], 0.0, 1.0),
        )

    def __repr__(self) -> str:  # pragma: no cover
        pnames = self.param_names
        pvals = ", ".join(f"{n}={v:.3f}" for n, v in zip(pnames, self.gui_params))
        return (
            f"AlertVector("
            f"location={'windshield' if self.location else 'panel'}, "
            f"color={'colorblind' if self.color else 'RGB'}, "
            f"vibration={bool(self.vibration)}, "
            f"lag={self.lag:.2f}s, "
            f"gui={self.gui_name}, "
            f"params=[{pvals}])"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Scoring utility  (episode-level — all inputs are observable after the episode)
# ─────────────────────────────────────────────────────────────────────────────

def compute_episode_score(
    route_completion: float,
    n_collisions: int,
    n_lane_violations: int,
    mean_dist_to_ai: float,
    time_ratio: float = 1.0,
    w_completion: float = 0.40,
    w_collision: float = 0.25,
    w_lane: float = 0.15,
    w_distance: float = 0.15,
    w_time: float = 0.05,
    max_collisions: int = 3,
    max_lane_violations: int = 10,
    max_distance: float = 50.0,
    max_time_ratio: float = 3.0,
) -> float:
    """Return a score in [0, 1] based on end-of-episode observables only.

    All inputs are quantities that can only be fully known once a driving
    episode has ended — no instantaneous signals (speed, steering) are used.

    Args:
        route_completion    : fraction of the AI route the hero completed [0, 1]
        n_collisions        : total collision events during the episode
        n_lane_violations   : total lane-boundary crossings during the episode
        mean_dist_to_ai     : mean spatial distance from the AI car over the
                              episode (metres) — measures how closely the hero
                              tracked the autonomous vehicle's path
        time_ratio          : episode_duration / optimal_duration; 1.0 = matched
                              the AI's pace exactly, >1 = slower
        w_*                 : per-component loss weights (must sum to 1)
        max_*               : normalisation denominators for each error term

    Returns:
        float in [0, 1]  — higher is better
    """
    completion_score = float(np.clip(route_completion, 0.0, 1.0))

    collision_err = min(n_collisions, max_collisions) / max(max_collisions, 1)
    lane_err = min(n_lane_violations, max_lane_violations) / max(max_lane_violations, 1)
    dist_err = float(np.clip(mean_dist_to_ai, 0.0, max_distance)) / max(max_distance, 1e-6)
    time_err = float(np.clip(abs(time_ratio - 1.0), 0.0, max_time_ratio - 1.0)) / max(
        max_time_ratio - 1.0, 1e-6
    )

    penalty = (
        w_collision * collision_err
        + w_lane * lane_err
        + w_distance * dist_err
        + w_time * time_err
    )
    return float(np.clip(w_completion * completion_score + (1.0 - w_completion) * (1.0 - penalty), 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Model 1: GaussianAlertModel (reward-weighted KDE, no neural network)
# ─────────────────────────────────────────────────────────────────────────────

class GaussianAlertModel:
    """Reward-weighted kernel density estimation alert sampler.

    How it works
    ────────────
    The buffer stores tuples (state, alert_raw_vector, score).  When asked
    to sample for a new state s*:

      1. Compute state-kernel weights:
             k_i = exp(‖s* − s_i‖² / (−2 σ_bw²))
      2. Compute score weights:
             r_i = exp(score_i / temperature)
      3. Combine:  w_i ∝ k_i · r_i   (normalised)
      4. Draw index i ~ Categorical(w), then sample:
             a ~ N(alert_i, σ_sample² · I)
      5. Clip and decode the sample into an AlertVector.

    On cold start (buffer < min_samples) the model returns a uniform random
    sample until enough experience is accumulated.
    """

    def __init__(
        self,
        state_dim: int = DEFAULT_STATE_DIM,
        bandwidth: Optional[float] = None,
        temperature: float = 0.5,
        sample_std: float = 0.15,
        max_buffer: int = 2000,
        min_samples: int = 5,
        exploration_rate: float = 0.10,
        seed: Optional[int] = None,
    ) -> None:
        self.state_dim = state_dim
        # None → compute adaptively at sample time via median pairwise distance.
        # A fixed float overrides this (useful when state scale is known).
        self.bandwidth = bandwidth
        self.temperature = temperature
        self.exploration_rate = exploration_rate
        self.sample_std = sample_std
        self.max_buffer = max_buffer
        self.min_samples = min_samples

        self._rng = np.random.default_rng(seed)

        # Each entry: {"state": ndarray, "alert": ndarray (8,), "score": float}
        self._buffer: deque = deque(maxlen=max_buffer)

        # Cache adaptive bandwidth — recomputed every 50 updates, not every call.
        self._bw_cache: float = 1.0
        self._bw_last_n: int = 0

        # Pending episode: set by begin_episode, consumed by end_episode
        self._pending: Optional[Tuple[np.ndarray, AlertVector]] = None

        # Running stats for logging
        self._n_updates = 0
        self._n_samples = 0
        self._score_history: deque = deque(maxlen=200)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def begin_episode(self, state: np.ndarray) -> AlertVector:
        """Sample an alert at the start of an episode and cache it.

        Call ``end_episode(score)`` once the episode has finished to
        commit the (state, alert, score) triple to the buffer.

        Args:
            state: driving-state feature vector at episode start

        Returns:
            The sampled AlertVector to display during the episode.
        """
        alert = self.sample(state)
        self._pending = (np.asarray(state, dtype=np.float32), alert)
        return alert

    def end_episode(self, score: float) -> None:
        """Commit the pending episode result using the end-of-episode score.

        Args:
            score: value from ``compute_episode_score()`` in [0, 1]
        """
        if self._pending is None:
            raise RuntimeError("end_episode() called without a matching begin_episode()")
        state, alert = self._pending
        self._pending = None
        self.update(state, alert, score)

    def sample(self, state: np.ndarray) -> AlertVector:
        """Sample an AlertVector given the current driving state."""
        self._n_samples += 1
        state = np.asarray(state, dtype=np.float32)

        if len(self._buffer) < self.min_samples:
            return self._random_sample()

        # Epsilon-greedy: escape mode collapse by occasionally sampling randomly.
        if self._rng.random() < self.exploration_rate:
            return self._random_sample()

        buf_states = np.stack([e["state"] for e in self._buffer])  # (N, D)
        buf_alerts = np.stack([e["alert"] for e in self._buffer])  # (N, 8)
        buf_scores = np.array([e["score"] for e in self._buffer])  # (N,)

        # Adaptive bandwidth: shrinks as the buffer fills and states concentrate,
        # ensuring the kernel only pulls from truly nearby states.
        bw = self._compute_bandwidth(buf_states)

        # State-kernel weights
        diff = buf_states - state[np.newaxis, :]
        sq_dist = np.sum(diff ** 2, axis=1)
        kernel_w = np.exp(-sq_dist / (2.0 * bw ** 2 + 1e-8))

        # LOCAL score normalisation: subtract the kernel-weighted local mean
        # rather than the global max. Without this, high-reward examples from
        # one state region (e.g., highlight in the "close" region) flood the
        # weights for all other regions because their scores appear better than
        # the global mean everywhere.
        local_mean = float(np.sum(kernel_w * buf_scores) / (kernel_w.sum() + 1e-8))
        local_shifted = buf_scores - local_mean
        score_w = np.exp(local_shifted / max(self.temperature, 1e-6))

        # Combined and normalised
        combined = kernel_w * score_w
        total = combined.sum()
        if total < 1e-12:
            return self._random_sample()
        probs = combined / total

        # Draw component
        idx = int(self._rng.choice(len(self._buffer), p=probs))
        center = buf_alerts[idx]

        # Sample from Gaussian around that component
        noise = self._rng.standard_normal(ALERT_DIM).astype(np.float32)
        raw = center + noise * self.sample_std

        return AlertVector.from_raw(raw)

    def _compute_bandwidth(self, buf_states: np.ndarray) -> float:
        """Adaptive bandwidth via the median pairwise distance (subsampled).

        Uses the standard Silverman-style heuristic:
            bw = median(pairwise_distances) / sqrt(2 * log(N))

        Result is cached and recomputed only every 50 buffer updates so the
        O(M²·D) cost doesn't dominate at high episode counts.
        A fixed ``self.bandwidth`` overrides this entirely.
        """
        if self.bandwidth is not None:
            return self.bandwidth

        N = len(buf_states)
        if (N - self._bw_last_n) < 50:
            return self._bw_cache

        max_sub = min(N, 80)
        idx = self._rng.choice(N, size=max_sub, replace=False)
        sub = buf_states[idx]

        diff = sub[:, np.newaxis, :] - sub[np.newaxis, :, :]  # (M, M, D)
        sq = np.sum(diff ** 2, axis=-1)                        # (M, M)
        upper_tri = sq[np.triu_indices(max_sub, k=1)]
        if len(upper_tri) == 0:
            return 1.0

        median_d = float(np.sqrt(np.median(upper_tri)))
        bw = median_d / math.sqrt(2.0 * math.log(max(N, 2)))
        self._bw_cache = max(bw, 0.05)
        self._bw_last_n = N
        return self._bw_cache

    def update(self, state: np.ndarray, alert: AlertVector, score: float) -> None:
        """Add an observed (state, alert, score) triple to the buffer."""
        self._buffer.append({
            "state": np.asarray(state, dtype=np.float32),
            "alert": alert.to_raw(),
            "score": float(score),
        })
        self._n_updates += 1
        self._score_history.append(score)

    def get_stats(self) -> Dict:
        """Return a dict of diagnostic statistics."""
        scores = list(self._score_history)
        return {
            "model": "GaussianAlertModel",
            "buffer_size": len(self._buffer),
            "n_updates": self._n_updates,
            "n_samples": self._n_samples,
            "mean_score_recent": float(np.mean(scores)) if scores else 0.0,
            "max_score_recent": float(np.max(scores)) if scores else 0.0,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _random_sample(self) -> AlertVector:
        """Uniform cold-start sample over the full output space."""
        raw = self._rng.random(ALERT_DIM).astype(np.float32)
        raw[3] *= MAX_LAG
        # Use integers so every gui_type is equally likely (0–4 each at 20%)
        raw[4] = float(self._rng.integers(0, NUM_GUI_TYPES))
        return AlertVector.from_raw(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Model 2 internals: neural-network sub-modules
# ─────────────────────────────────────────────────────────────────────────────

def _mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 2) -> nn.Sequential:
    """Helper to build a small MLP with ReLU activations."""
    modules: List[nn.Module] = []
    prev = in_dim
    for _ in range(layers):
        modules += [nn.Linear(prev, hidden), nn.ReLU()]
        prev = hidden
    modules.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*modules)


class _GatingNetwork(nn.Module):
    """Maps state → logits over NUM_GUI_TYPES choices."""

    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = _mlp(state_dim, hidden_dim, NUM_GUI_TYPES, layers=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, 3) raw logits


class _ExpertNetwork(nn.Module):
    """One expert per GUI type.  Outputs mean and log-std for GUI params."""

    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.trunk = _mlp(state_dim, hidden_dim, hidden_dim, layers=2)
        self.mean_head = nn.Linear(hidden_dim, GUI_PARAM_DIM)
        self.log_std = nn.Parameter(torch.full((GUI_PARAM_DIM,), -2.5))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        mean = torch.sigmoid(self.mean_head(h))        # (B, 3) in (0,1)
        std = torch.exp(self.log_std.clamp(-5.0, -1.0)) # (3,) std in ~[0.007, 0.37]
        return mean, std.expand(x.size(0), -1)


class _SharedHead(nn.Module):
    """Outputs location, color, vibration (Bernoulli logits) and lag (Normal)."""

    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.trunk = _mlp(state_dim, hidden_dim, hidden_dim, layers=2)
        self.discrete_head = nn.Linear(hidden_dim, 3)   # logits for loc/col/vib
        self.lag_mean_head = nn.Linear(hidden_dim, 1)
        self.lag_log_std = nn.Parameter(torch.tensor(-3.0))  # std ≈ 0.10 × MAX_LAG

        # Near-zero init keeps Bernoulli logits close to 0 (p ≈ 0.5) at the
        # start so gradients are maximally large (sigmoid slope = 0.25 at 0).
        nn.init.normal_(self.discrete_head.weight, std=0.01)
        nn.init.zeros_(self.discrete_head.bias)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        disc_logits = self.discrete_head(h)             # (B, 3)
        lag_mean = torch.sigmoid(self.lag_mean_head(h)) * MAX_LAG  # (B, 1)
        lag_std = torch.exp(self.lag_log_std.clamp(-4.0, 0.0)) * MAX_LAG
        return disc_logits, lag_mean, lag_std


# ─────────────────────────────────────────────────────────────────────────────
# Model 2: MoEAlertModel (MLP mixture-of-experts + PPO training)
# ─────────────────────────────────────────────────────────────────────────────

class MoEAlertModel(nn.Module):
    """Mixture-of-experts alert model trained with a PPO-style clipped loss.

    Architecture
    ────────────
    • GatingNetwork   : state → Categorical(3)  which GUI type to use
    • ExpertNetwork×3 : state → Normal(μ, σ)     GUI-specific params
    • SharedHead      : state → Bernoulli logits + Normal(μ, σ) for lag

    Training (PPO-style)
    ────────────────────
    Because no ground-truth labels exist, the model uses policy-gradient
    learning with driver-deviation scores as the reward signal.

    At each call to ``sample()``, the log-probability of the chosen alert is
    stored alongside the alert and state.  When enough experiences accumulate,
    ``train_step()`` runs a PPO update:

        advantage  =  (score − mean_score) / std_score
        ratio      =  exp(log_π_new − log_π_old)
        L_policy   = −mean( min(ratio·A,  clip(ratio, 1−ε, 1+ε)·A) )
        L_entropy  = −entropy_coeff · mean(entropy)
        loss       =  L_policy + L_entropy

    The ratio r = π_new/π_old is exactly the "change in parameters relative to
    the change in loss" signal: it measures how much the policy has shifted
    since the experience was collected and prevents destabilising updates.
    """

    def __init__(
        self,
        state_dim: int = DEFAULT_STATE_DIM,
        hidden_dim: int = 128,
        expert_hidden_dim: int = 64,
        lr: float = 5e-4,
        clip_eps: float = 0.15,
        entropy_coeff: float = 0.01,
        ppo_epochs: int = 12,
        batch_size: int = 16,
        buffer_size: int = 48,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        if seed is not None:
            torch.manual_seed(seed)

        self.state_dim = state_dim
        self.clip_eps = clip_eps
        self.entropy_coeff = entropy_coeff
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        # Networks
        self.gating = _GatingNetwork(state_dim, hidden_dim)
        self.experts = nn.ModuleList(
            [_ExpertNetwork(state_dim, expert_hidden_dim) for _ in range(NUM_GUI_TYPES)]
        )
        self.shared_head = _SharedHead(state_dim, hidden_dim)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Experience buffer for PPO
        # Each entry: {"state", "gui_type", "gui_params", "discretes", "lag", "old_log_prob", "score"}
        self._buffer: List[Dict] = []

        # Pending episode: set by begin_episode, consumed by end_episode
        self._pending: Optional[Tuple[np.ndarray, AlertVector, float]] = None

        # EMA baseline for stable advantage estimation.
        # Using batch mean/std alone is fragile when reward variance is small
        # (rewards in ~[0.4, 1.0] give std < 0.05, causing exploding advantages).
        # The EMA accumulates signal across many updates instead.
        self._reward_baseline: float = 0.5
        self._baseline_alpha: float = 0.05  # slow EMA — keeps historical context

        # Logging
        self._n_updates = 0
        self._n_samples = 0
        self._loss_history: deque = deque(maxlen=200)
        self._score_history: deque = deque(maxlen=200)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def begin_episode(self, state: np.ndarray) -> AlertVector:
        """Sample an alert at the start of an episode and cache it.

        Call ``end_episode(score)`` once the episode has finished to
        commit the experience to the PPO buffer.

        Args:
            state: driving-state feature vector at episode start

        Returns:
            The sampled AlertVector to display during the episode.
        """
        alert, log_prob = self.sample(state)
        self._pending = (np.asarray(state, dtype=np.float32), alert, log_prob)
        return alert

    def end_episode(self, score: float) -> Optional[float]:
        """Commit the pending episode result using the end-of-episode score.

        Triggers a PPO update when the internal buffer is full.

        Args:
            score: value from ``compute_episode_score()`` in [0, 1]

        Returns:
            Training loss if a PPO update was triggered, else None.
        """
        if self._pending is None:
            raise RuntimeError("end_episode() called without a matching begin_episode()")
        state, alert, log_prob = self._pending
        self._pending = None
        return self.store_experience(state, alert, score, log_prob)

    def sample(self, state: np.ndarray) -> Tuple[AlertVector, float]:
        """Sample an alert for the given state.

        Returns:
            (AlertVector, log_prob): log_prob must be passed to
            ``store_experience()`` for the PPO update.
        """
        self._n_samples += 1
        state_t = torch.FloatTensor(state).unsqueeze(0)  # (1, D)

        with torch.no_grad():
            # --- GUI type from gating ---
            gating_logits = self.gating(state_t)
            gating_dist = Categorical(logits=gating_logits)
            gui_type = gating_dist.sample()               # (1,)
            lp_gui = gating_dist.log_prob(gui_type)       # (1,)

            # --- GUI params from selected expert ---
            g = gui_type.item()
            mean, std = self.experts[g](state_t)
            expert_dist = Normal(mean, std)
            gui_params_t = expert_dist.sample()
            gui_params_t = gui_params_t.clamp(0.0, 1.0)
            lp_params = expert_dist.log_prob(gui_params_t).sum(dim=-1)  # (1,)

            # --- Shared params ---
            disc_logits, lag_mean, lag_std = self.shared_head(state_t)
            disc_dist = Bernoulli(logits=disc_logits)
            discretes = disc_dist.sample()                               # (1,3)
            lag_dist = Normal(lag_mean.squeeze(-1), lag_std)
            lag_t = lag_dist.sample().clamp(0.0, MAX_LAG)               # (1,)

            lp_disc = disc_dist.log_prob(discretes).sum(dim=-1)         # (1,)
            lp_lag = lag_dist.log_prob(lag_t)                           # (1,)

            total_log_prob = (lp_gui + lp_params + lp_disc + lp_lag).item() / _LOG_PROB_COMPONENTS

        alert = AlertVector(
            location=int(discretes[0, 0].round().item()),
            color=int(discretes[0, 1].round().item()),
            vibration=int(discretes[0, 2].round().item()),
            lag=float(lag_t.item()),
            gui_type=g,
            gui_params=gui_params_t.squeeze(0).numpy(),
        )
        return alert, total_log_prob

    def store_experience(
        self,
        state: np.ndarray,
        alert: AlertVector,
        score: float,
        log_prob: float,
    ) -> Optional[float]:
        """Store one transition.  Triggers a PPO update when buffer is full.

        Returns:
            Training loss if an update occurred, else None.
        """
        self._buffer.append({
            "state": np.asarray(state, dtype=np.float32),
            "gui_type": alert.gui_type,
            "gui_params": alert.gui_params.astype(np.float32),
            "discretes": np.array(
                [alert.location, alert.color, alert.vibration], dtype=np.float32
            ),
            "lag": np.float32(alert.lag),
            "old_log_prob": np.float32(log_prob),
            "score": np.float32(score),
        })
        self._score_history.append(score)

        if len(self._buffer) >= self.buffer_size:
            loss = self._ppo_update()
            self._buffer.clear()
            return loss
        return None

    def get_stats(self) -> Dict:
        scores = list(self._score_history)
        losses = list(self._loss_history)
        return {
            "model": "MoEAlertModel",
            "buffer_fill": len(self._buffer),
            "n_updates": self._n_updates,
            "n_samples": self._n_samples,
            "mean_score_recent": float(np.mean(scores)) if scores else 0.0,
            "max_score_recent": float(np.max(scores)) if scores else 0.0,
            "mean_loss_recent": float(np.mean(losses)) if losses else 0.0,
        }

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def _ppo_update(self) -> float:
        """Run ppo_epochs of minibatch gradient updates over the current buffer."""
        buf = self._buffer

        states = torch.FloatTensor(np.stack([e["state"] for e in buf]))
        gui_types = torch.LongTensor([e["gui_type"] for e in buf])
        gui_params = torch.FloatTensor(np.stack([e["gui_params"] for e in buf]))
        discretes = torch.FloatTensor(np.stack([e["discretes"] for e in buf]))
        lags = torch.FloatTensor(np.array([e["lag"] for e in buf]))
        old_log_probs = torch.FloatTensor(np.array([e["old_log_prob"] for e in buf]))
        scores = torch.FloatTensor(np.array([e["score"] for e in buf]))

        # Advantage via EMA baseline — avoids exploding gradients when the
        # batch reward std is tiny (common early in training or with narrow
        # reward ranges). Normalize by mean absolute deviation for stable scale.
        batch_mean = scores.mean().item()
        self._reward_baseline = (
            (1.0 - self._baseline_alpha) * self._reward_baseline
            + self._baseline_alpha * batch_mean
        )
        adv = scores - self._reward_baseline
        # std-based normalisation with a hard floor of 0.1 prevents explosion
        # when all rewards in the batch are nearly identical (tiny std).
        # Clamp to [-5, 5] as a final safety net.
        adv = adv / (adv.std() + 0.1)
        adv = adv.clamp(-5.0, 5.0)

        total_loss = 0.0
        n = len(buf)

        for _ in range(self.ppo_epochs):
            # Shuffle minibatches
            idx = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                mb = idx[start : start + self.batch_size]

                new_lp = self._compute_log_probs(
                    states[mb],
                    gui_types[mb],
                    gui_params[mb],
                    discretes[mb],
                    lags[mb],
                )
                ratio = torch.exp(new_lp - old_log_probs[mb])
                a = adv[mb]

                # Clipped surrogate objective (PPO-clip)
                surr1 = ratio * a
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * a
                policy_loss = -torch.min(surr1, surr2).mean()

                # Entropy bonus to maintain exploration
                entropy = self._compute_entropy(states[mb])

                # ---- AWR auxiliary loss ------------------------------------------------
                # PPO alone is too weak to fine-tune continuous params (gui_params, lag)
                # because gui_type dominates 50% of the reward.  Once the gating network
                # picks the right GUI type, the policy gradient for expert means is tiny.
                #
                # Fix: Advantage-Weighted Regression on the continuous outputs.
                # For each sample, compute softmax(score * T) weights so the highest-
                # scoring transitions pull the expert mean and lag head toward their
                # observed values.  This is direct supervised learning on good samples
                # without needing explicit labels — the oracle is implicit in the scores.
                awr_w = torch.softmax(scores[mb] * 5.0, dim=0)  # (mb,) sums to 1

                awr_loss = torch.tensor(0.0)
                for g in range(NUM_GUI_TYPES):
                    gmask = gui_types[mb] == g
                    if gmask.any():
                        mean_g, _ = self.experts[g](states[mb][gmask])
                        mse_g = (mean_g - gui_params[mb][gmask]).pow(2).sum(dim=-1)
                        awr_loss = awr_loss + (awr_w[gmask] * mse_g).sum()

                # Lag regression
                _, lag_mean_new, _ = self.shared_head(states[mb])
                lag_norm = lags[mb] / MAX_LAG          # normalise to [0,1]
                lag_pred = lag_mean_new.squeeze(-1) / MAX_LAG
                lag_mse = (lag_pred - lag_norm).pow(2)
                awr_loss = awr_loss + (awr_w * lag_mse).sum()
                # -----------------------------------------------------------------------

                loss = policy_loss - self.entropy_coeff * entropy + 0.5 * awr_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
                self.optimizer.step()

                total_loss += loss.item()

        self._n_updates += 1
        avg_loss = total_loss / max(self.ppo_epochs * math.ceil(n / self.batch_size), 1)
        self._loss_history.append(avg_loss)
        return avg_loss

    # ------------------------------------------------------------------
    # Log-probability and entropy helpers
    # ------------------------------------------------------------------

    def _compute_log_probs(
        self,
        states: torch.Tensor,
        gui_types: torch.Tensor,
        gui_params: torch.Tensor,
        discretes: torch.Tensor,
        lags: torch.Tensor,
    ) -> torch.Tensor:
        """Compute joint log P(alert | state) for a batch."""
        # Gating
        gating_logits = self.gating(states)
        gating_dist = Categorical(logits=gating_logits)
        lp_gui = gating_dist.log_prob(gui_types)               # (B,)

        # Expert for each GUI type — loop over types and mask
        lp_params = torch.zeros(states.size(0), device=states.device)
        for g in range(NUM_GUI_TYPES):
            mask = gui_types == g
            if mask.any():
                mean, std = self.experts[g](states[mask])
                dist = Normal(mean, std)
                lp_params[mask] = dist.log_prob(gui_params[mask]).sum(dim=-1)

        # Shared head
        disc_logits, lag_mean, lag_std = self.shared_head(states)
        disc_dist = Bernoulli(logits=disc_logits)
        lp_disc = disc_dist.log_prob(discretes).sum(dim=-1)    # (B,)

        lag_dist = Normal(lag_mean.squeeze(-1), lag_std)
        lp_lag = lag_dist.log_prob(lags)                        # (B,)

        return (lp_gui + lp_params + lp_disc + lp_lag) / _LOG_PROB_COMPONENTS

    def _compute_entropy(self, states: torch.Tensor) -> torch.Tensor:
        """Mean entropy of the full joint policy for a batch of states."""
        gating_logits = self.gating(states)
        gating_dist = Categorical(logits=gating_logits)
        h_gui = gating_dist.entropy().mean()

        # Weight each expert's entropy by the gating probability so we only
        # encourage exploration in experts that are actually being used.
        gate_probs = torch.softmax(gating_logits, dim=-1)  # (B, NUM_GUI_TYPES)
        h_experts = torch.tensor(0.0)
        for g, expert in enumerate(self.experts):
            _, std = expert(states)
            expert_dist = Normal(torch.zeros_like(std), std)
            h = expert_dist.entropy().sum(dim=-1)           # (B,)
            h_experts = h_experts + (gate_probs[:, g] * h).mean()

        disc_logits, _, lag_std = self.shared_head(states)
        disc_dist = Bernoulli(logits=disc_logits)
        h_disc = disc_dist.entropy().sum(dim=-1).mean()

        lag_dist = Normal(torch.zeros(states.size(0)), lag_std)
        h_lag = lag_dist.entropy().mean()

        return h_gui + h_experts + h_disc + h_lag


# ─────────────────────────────────────────────────────────────────────────────
# Factory helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_gaussian_model(
    state_dim: int = DEFAULT_STATE_DIM,
    bandwidth: Optional[float] = None,
    temperature: float = 0.5,
    sample_std: float = 0.15,
    exploration_rate: float = 0.10,
    seed: Optional[int] = None,
) -> GaussianAlertModel:
    """Convenience constructor for GaussianAlertModel."""
    return GaussianAlertModel(
        state_dim=state_dim,
        bandwidth=bandwidth,
        temperature=temperature,
        sample_std=sample_std,
        exploration_rate=exploration_rate,
        seed=seed,
    )


def make_moe_model(
    state_dim: int = DEFAULT_STATE_DIM,
    hidden_dim: int = 128,
    lr: float = 3e-4,
    clip_eps: float = 0.2,
    buffer_size: int = 48,
    seed: Optional[int] = None,
) -> MoEAlertModel:
    """Convenience constructor for MoEAlertModel."""
    return MoEAlertModel(
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        lr=lr,
        clip_eps=clip_eps,
        buffer_size=buffer_size,
        seed=seed,
    )
