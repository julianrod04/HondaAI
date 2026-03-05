"""test_alert_models.py

Standalone test for alert_models.py — no CARLA simulation required.

Simulates many episodes of a participant driving alongside an autonomous
vehicle. Each episode:
  1. A driver-profile state is generated (stable features that characterise
     WHO is driving, not what is happening right now).
  2. Each model proposes an alert configuration.
  3. A fake reward is computed based on how close the proposed alert is
     to the oracle's best configuration for that driver type.
  4. The model is updated with the reward.

Driver-profile state (4 features + 4-element identity vector = 8 total, all in [0, 1]):
  [0] mean_route_completion  — fraction of AI route the hero completes
  [1] mean_dist_to_ai_norm   — normalised mean following distance (0=close)
  [2] experience_level       — driving expertise proxy (0=novice, 1=expert)
  [3] reaction_speed         — estimated reaction latency (0=slow, 1=fast)
  [4-7] identity vector      — 4-element driver-type encoding (types 0-3: one-hot; type 4: all-zeros)

Three archetypal driver profiles cycle across episodes.  The profile is
completely static — no noise — reflecting that the system has converged on
a stable driver model and is now optimising alert parameters for that driver.

Usage:
    python test_alert_models.py
    python test_alert_models.py --episodes 20000
"""

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Allow import from sibling directory when run directly
sys.path.insert(0, str(Path(__file__).parent))

from alert_models import (
    GUI_TYPES,
    MAX_LAG,
    NUM_GUI_TYPES,
    AlertVector,
    make_gaussian_model,
    make_moe_model,
)

# ─────────────────────────────────────────────────────────────────────────────
# Module-level constants
# ─────────────────────────────────────────────────────────────────────────────

PLOT_WINDOW = 200  # rolling window size for plots and print_summary

_DRIVER_FEATURE_DIM = 4   # [completion, dist_to_ai, experience_level, reaction_speed]
_NUM_DRIVER_TYPES = 5     # novice / cautious / intermediate / confident / expert
_DRIVER_ID_DIM = 4        # identity vector length (4-element encoding for 5 types)
# Encoding: types 0-3 → standard one-hot; type 4 → all-zeros (unique pattern)
# Total state passed to models = features + identity vector
DRIVER_STATE_DIM = _DRIVER_FEATURE_DIM + _DRIVER_ID_DIM  # 8

#   Feature layout: [completion, dist_to_ai, experience_level, reaction_speed]
DRIVER_PROFILES = [
    np.array([0.35, 0.80, 0.05, 0.20], dtype=np.float32),  # novice
    np.array([0.55, 0.70, 0.25, 0.30], dtype=np.float32),  # cautious
    np.array([0.70, 0.45, 0.50, 0.55], dtype=np.float32),  # intermediate
    np.array([0.82, 0.30, 0.70, 0.70], dtype=np.float32),  # confident
    np.array([0.93, 0.10, 0.95, 0.90], dtype=np.float32),  # expert
]
DRIVER_NAMES = {0: "Novice", 1: "Cautious", 2: "Intermediate", 3: "Confident", 4: "Expert"}


# ─────────────────────────────────────────────────────────────────────────────
# Oracle: defines which alert configuration is "best" per driver type
# ─────────────────────────────────────────────────────────────────────────────
#
# Alerts adapt to WHO is driving (driver skill / profile), not to every
# momentary change in the driving scene.  The discriminant is:
#
#   expertise = route_completion (state[0]) − dist_to_ai (state[1])
#
# Three alert types (NUM_GUI_TYPES = 3):
#   0  arrow  – directional arrow to AV future position (uses lag, color, vibration, location)
#               params: [scale, opacity, vibration_dist]
#   1  route  – full AV trajectory overlay             (uses color, vibration, location; no lag)
#               params: [line_width, opacity, vibration_dist]
#   2  sound  – audio "left"/"right" cue               (uses lag; no color, no location)
#               params: [lateral_threshold, cooldown, volume]
#
# Driver types → alert oracle (expertise = completion - dist_to_ai):
# ─────────────────────────────────────────────────────────────────────────────────────────
#   Novice       (exp < -0.2)       : windshield arrow + colorblind + vibration, lag=0.3s
#   Cautious     (-0.2 ≤ exp < 0.1) : windshield arrow, no vibration,            lag=0.5s
#   Intermediate (0.1 ≤ exp < 0.4)  : panel route, no vibration,                 lag=0.6s
#   Confident    (0.4 ≤ exp < 0.7)  : panel route, less prominent,               lag=0.8s
#   Expert       (exp ≥ 0.7)        : sound only, high lag,                      lag=1.2s
#
# Expertise scores per profile:
#   Novice -0.45 | Cautious -0.15 | Intermediate +0.25 | Confident +0.52 | Expert +0.83
#
# Raw vector layout: [location, color, vibration, lag, gui_type, p0, p1, p2]

def _oracle_alert(state: np.ndarray) -> np.ndarray:
    """Return the ideal raw alert vector for a given driver profile."""
    expertise = float(state[0]) - float(state[1])
    if expertise < -0.2:    # novice — windshield arrow, colorblind palette, haptic
        #                      loc=1  col=1  vib=1  lag=0.3  gui=0(arrow)  scale=0.9  opacity=0.9  vib_dist=0.2
        return np.array([1.0, 1.0, 1.0, 0.3, 0.0, 0.9, 0.9, 0.2], dtype=np.float32)
    elif expertise < 0.1:   # cautious — windshield arrow, no haptic
        #                      loc=1  col=0  vib=0  lag=0.5  gui=0(arrow)  scale=0.6  opacity=0.8  vib_dist=0.5
        return np.array([1.0, 0.0, 0.0, 0.5, 0.0, 0.6, 0.8, 0.5], dtype=np.float32)
    elif expertise < 0.4:   # intermediate — panel route
        #                      loc=0  col=0  vib=0  lag=0.6  gui=1(route)  lw=0.5  opacity=0.8  vib_dist=0.6
        return np.array([0.0, 0.0, 0.0, 0.6, 1.0, 0.5, 0.8, 0.6], dtype=np.float32)
    elif expertise < 0.7:   # confident — panel route, less prominent
        #                      loc=0  col=0  vib=0  lag=0.8  gui=1(route)  lw=0.3  opacity=0.5  vib_dist=0.8
        return np.array([0.0, 0.0, 0.0, 0.8, 1.0, 0.3, 0.5, 0.8], dtype=np.float32)
    else:                   # expert — sound only
        #                      loc=0  col=0  vib=0  lag=1.2  gui=2(sound)  lat_thresh=0.7  cooldown=0.5  vol=0.7
        return np.array([0.0, 0.0, 0.0, 1.2, 2.0, 0.7, 0.5, 0.7], dtype=np.float32)


def compute_fake_episode_reward(
    alert: AlertVector, state: np.ndarray, rng: np.random.Generator
) -> float:
    """Return an end-of-episode score in [0, 1] measuring how close the alert
    is to the oracle configuration for this driver type.

    Direct exponential of weighted-L1 distance from the oracle.
    - Range: [~0.02, 1.0]
    - gui_type carries 50% of the weight
    - Small Gaussian noise (std=0.02) simulates participant variability
    """
    oracle = _oracle_alert(state)
    sampled = alert.to_raw()

    norm_s = sampled.copy(); norm_s[3] /= MAX_LAG; norm_s[4] /= (NUM_GUI_TYPES - 1)
    norm_o = oracle.copy();  norm_o[3] /= MAX_LAG; norm_o[4] /= (NUM_GUI_TYPES - 1)

    weights = np.array([0.10, 0.04, 0.08, 0.10, 0.50, 0.08, 0.06, 0.04], dtype=np.float32)
    mismatch = float(np.sum(weights * np.abs(norm_s - norm_o)))
    reward = float(np.exp(-4.0 * mismatch))
    return float(np.clip(reward + rng.normal(0, 0.02), 0.0, 1.0))


def generate_state(episode: int = 0) -> np.ndarray:
    """Return a static driver-profile state with an explicit one-hot driver-type label.

    Layout: [8 behavioural features | 3-dim one-hot driver type]
    The one-hot gives the model a direct, unambiguous identity signal instead
    of forcing it to infer driver type from raw feature values alone.

    Episodes cycle through 3 archetypes (novice → intermediate → expert).
    Profiles are static — no noise — for clean convergence testing.
    """
    pid = episode % _NUM_DRIVER_TYPES
    # 4-element identity: types 0-3 → standard one-hot; type 4 → all-zeros
    identity = np.zeros(_DRIVER_ID_DIM, dtype=np.float32)
    if pid < _DRIVER_ID_DIM:
        identity[pid] = 1.0
    return np.concatenate([DRIVER_PROFILES[pid], identity])


# ─────────────────────────────────────────────────────────────────────────────
# Experiment runners — return trained model alongside metrics
# ─────────────────────────────────────────────────────────────────────────────

def run_gaussian(n_episodes: int, seed: int = 42) -> Dict:
    """Run GaussianAlertModel for n_episodes and return metrics + trained model."""
    rng = np.random.default_rng(seed)
    model = make_gaussian_model(state_dim=DRIVER_STATE_DIM, seed=seed)

    episode_rewards: List[float] = []
    episode_gui_types: List[int] = []

    for ep in range(n_episodes):
        state = generate_state(ep)
        alert = model.begin_episode(state)
        reward = compute_fake_episode_reward(alert, state, rng)
        model.end_episode(reward)
        episode_rewards.append(reward)
        episode_gui_types.append(alert.gui_type)

    return {
        "rewards": episode_rewards,
        "gui_types": episode_gui_types,
        "stats": model.get_stats(),
        "model": model,
    }


def run_moe(
    n_episodes: int,
    buffer_size: int = 48,
    hidden_dim: int = 128,
    lr: float = 5e-4,
    seed: int = 42,
) -> Dict:
    """Run MoEAlertModel for n_episodes and return metrics + trained model."""
    rng = np.random.default_rng(seed)
    model = make_moe_model(
        state_dim=DRIVER_STATE_DIM,
        hidden_dim=hidden_dim,
        lr=lr,
        buffer_size=buffer_size,
        seed=seed,
    )

    episode_rewards: List[float] = []
    episode_gui_types: List[int] = []
    update_losses: List[Tuple[int, float]] = []

    for ep in range(n_episodes):
        state = generate_state(ep)
        alert = model.begin_episode(state)
        reward = compute_fake_episode_reward(alert, state, rng)
        loss = model.end_episode(reward)
        if loss is not None:
            update_losses.append((ep, loss))
        episode_rewards.append(reward)
        episode_gui_types.append(alert.gui_type)

    return {
        "rewards": episode_rewards,
        "gui_types": episode_gui_types,
        "update_losses": update_losses,
        "stats": model.get_stats(),
        "model": model,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reporting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rolling_mean(values: List[float], window: int = PLOT_WINDOW) -> np.ndarray:
    """O(n) rolling mean via cumsum."""
    arr = np.array(values, dtype=np.float64)
    cs = np.concatenate([[0.0], np.cumsum(arr)])
    # Full window: (cs[i+w] - cs[i]) / w; warm-up: cs[i+1] / (i+1)
    full = (cs[window:] - cs[:-window]) / window
    warm = cs[1:window] / np.arange(1, window)
    return np.concatenate([warm, full])


def _rolling_freq_by_type(
    gui_types: List[int], window: int = PLOT_WINDOW
) -> Dict[int, np.ndarray]:
    """O(n * NUM_GUI_TYPES) rolling selection frequency via binary mask + cumsum."""
    arr = np.array(gui_types, dtype=np.int32)
    return {
        g: _rolling_mean((arr == g).astype(float).tolist(), window)
        for g in range(NUM_GUI_TYPES)
    }


def _gui_distribution(gui_types: List[int], last_n: int = PLOT_WINDOW) -> Dict[str, float]:
    recent = gui_types[-last_n:]
    counts = Counter(recent)
    total = len(recent)
    return {GUI_TYPES[g]: round(counts.get(g, 0) / total, 3) for g in range(NUM_GUI_TYPES)}


def print_summary(name: str, result: Dict, n_episodes: int, window: int = PLOT_WINDOW) -> None:
    rewards = result["rewards"]
    gui_types = result["gui_types"]

    early_mean = float(np.mean(rewards[:window]))
    late_mean = float(np.mean(rewards[-window:]))
    best = float(np.max(rewards))

    print(f"\n{'─'*55}")
    print(f"  {name}")
    print(f"{'─'*55}")
    print(f"  Episodes          : {n_episodes}")
    print(f"  Mean reward (first {window}): {early_mean:.4f}")
    print(f"  Mean reward (last  {window}): {late_mean:.4f}")
    print(f"  Improvement        : {late_mean - early_mean:+.4f}")
    print(f"  Best reward seen   : {best:.4f}")
    print(f"  GUI distribution (last {window} eps):")
    for gui_name, freq in _gui_distribution(gui_types, last_n=window).items():
        bar = "█" * int(freq * 20)
        print(f"    {gui_name:<12}: {freq:.3f}  {bar}")
    print(f"  Model stats:")
    for k, v in result["stats"].items():
        if k != "model":
            print(f"    {k:<25}: {v}")

    if "update_losses" in result and result["update_losses"]:
        losses = [l for _, l in result["update_losses"]]
        print(f"  PPO updates        : {len(losses)}")
        print(f"  Loss first update  : {losses[0]:.4f}")
        print(f"  Loss last update   : {losses[-1]:.4f}")


def print_oracle_summary() -> None:
    """Print the oracle's ideal alert for each driver archetype."""
    print("\n" + "=" * 55)
    print("  ORACLE CONFIGURATION (per driver type)")
    print("=" * 55)
    for pid, name in DRIVER_NAMES.items():
        state = DRIVER_PROFILES[pid]
        expertise = float(state[0]) - float(state[1])
        av = AlertVector.from_raw(_oracle_alert(state))
        print(f"\n  {name}  (expertise score = {expertise:+.2f})")
        print(f"    {av}")


def _plot_freq_panel(ax, eps, freq: Dict[int, np.ndarray], title: str) -> None:
    """Plot rolling GUI-type selection frequency lines onto ax."""
    type_colors = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#D55E00"]
    for g in range(NUM_GUI_TYPES):
        ax.plot(eps, freq[g], label=GUI_TYPES[g], color=type_colors[g], linewidth=1.4)
    ax.axhline(1 / NUM_GUI_TYPES, color="grey", linestyle=":", linewidth=0.8,
               label="uniform (0.20)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Selection frequency")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)


def maybe_plot(gaussian_result: Dict, moe_result: Dict, window: int = PLOT_WINDOW) -> None:
    """Convergence plot: reward + per-GUI-type rolling selection frequency."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[plot] matplotlib not installed — skipping plot.")
        return

    g_rolling = _rolling_mean(gaussian_result["rewards"], window)
    m_rolling = _rolling_mean(moe_result["rewards"], window)
    n = len(g_rolling)
    eps = np.arange(n)

    g_freq = _rolling_freq_by_type(gaussian_result["gui_types"], window)
    m_freq = _rolling_freq_by_type(moe_result["gui_types"], window)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Alert Model Convergence (rolling window = {window} episodes)", fontsize=13)

    ax = axes[0]
    ax.plot(eps, g_rolling, label="Gaussian KDE", color="steelblue", linewidth=1.5)
    ax.plot(eps, m_rolling, label="MoE (PPO+AWR)", color="darkorange", linewidth=1.5)
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8, label="Oracle max")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rolling mean reward")
    ax.set_title("Reward over episodes")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    _plot_freq_panel(axes[1], eps, g_freq, "Gaussian KDE — GUI type frequency")
    _plot_freq_panel(axes[2], eps, m_freq, "MoE (PPO+AWR) — GUI type frequency")

    plt.tight_layout()
    save_path = Path(__file__).parent / "test_alert_output.png"
    fig.savefig(save_path, dpi=150)
    print(f"\n\nFigure saved to: {save_path}\n\n")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(n_episodes: int = 20_000) -> None:
    print("=" * 55)
    print("  alert_models.py — standalone convergence test")
    print("=" * 55)
    print_oracle_summary()

    print(f"\nRunning {n_episodes} episodes for each model …")
    np.random.seed(0)

    g_result = run_gaussian(n_episodes, seed=42)
    m_result = run_moe(n_episodes, buffer_size=48, seed=42)

    print_summary("GaussianAlertModel", g_result, n_episodes)
    print_summary("MoEAlertModel (PPO+AWR)", m_result, n_episodes)

    print("\n" + "=" * 55)
    print("  FINAL SAMPLE — what each model proposes after training")
    print("=" * 55)

    g_model = g_result["model"]
    m_model = m_result["model"]

    for pid, name in DRIVER_NAMES.items():
        identity = np.zeros(_DRIVER_ID_DIM, dtype=np.float32)
        if pid < _DRIVER_ID_DIM:
            identity[pid] = 1.0
        state = np.concatenate([DRIVER_PROFILES[pid], identity])
        oracle = AlertVector.from_raw(_oracle_alert(state))
        g_alert = g_model.sample(state)
        m_alert, _ = m_model.sample(state)
        print(f"\n  {name}")
        print(f"    Oracle  : {oracle}")
        print(f"    Gaussian: {g_alert}")
        print(f"    MoE     : {m_alert}")

    maybe_plot(g_result, m_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test alert_models without CARLA")
    parser.add_argument("--episodes", type=int, default=20_000, help="Number of episodes")
    args = parser.parse_args()
    main(n_episodes=args.episodes)
