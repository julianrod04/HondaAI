#!/usr/bin/env python3
"""alert_pipeline.py — Adaptive alert pipeline (Windows, CARLA UE4 display)

HOW TO RUN
==========

Prerequisites
-------------
All dependencies are already in sb3/requirements.txt.  No extra installs needed.

  pip install -r ../../sb3/requirements.txt   # if not already done

Step 1 — Start the CARLA server on Windows
------------------------------------------
  Open a terminal and run:

      CarlaUE4.exe -world-port=2000

  The UE4 window is what the participant sees.  Keep it visible.
  The pipeline drives the world in synchronous mode; do not click inside the
  UE4 window or it may steal focus.

Step 2 — Run the pipeline
-------------------------
  Open a second terminal (PowerShell or cmd) in kw_sandbox/:

      python alert_pipeline.py ^
          --model ..\\..\\..\run\\pdmorl_Train_Session_2-16-2026_bestCombined.zip ^
          --scenario intersection ^
          --calibration-duration 120 ^
          --max-iterations 20 ^
          --port 2000

  (Linux/macOS) replace ^ with \\

Step 3 — Quick smoke test (no CARLA, no model needed)
------------------------------------------------------
  python alert_pipeline.py --test

  Runs unit tests for state building, colour helpers, and data classes.
  Prints PASS / FAIL for each and exits.  Useful to verify the script
  imports correctly before starting CARLA.

CONTROLS
--------
  Logitech G920 (or any HID wheel/pedal):
      Axis 0  steering    (-1 full left, +1 full right)
      Axis 1  brake       (-1 rest, +1 fully pressed)
      Axis 3  accelerator (-1 rest, +1 fully pressed)
      Button 5 handbrake

  Keyboard fallback (W/A/S/D):
      W = throttle, S = brake, A = steer left, D = steer right
      SPACE = handbrake, R = toggle reverse

  The pipeline opens a small input window (200×60px, top-left corner).
  Focus it if keyboard input stops responding — it captures all key events.
  The participant's main view is the CARLA UE4 window.

ARCHITECTURE: CARLA AS DATA BUS
--------------------------------
  CARLA provides:
    • Vehicle state each tick  (get_vehicle_measurements, get_transform, etc.)
    • World actor list          (for closest_car / HumanStyleRegressor)
    • Debug drawing API         (world.debug.draw_arrow / draw_line / draw_string)
      → This is the ONLY rendering path.  No OpenCV, no extra windows.

  Python handles:
    • Alert model (MoEAlertModel) — sampled once per episode, fixed type
    • Style regression (HumanStyleRegressor) — grades human each tick
    • Training loop — PPO/AWR updates after each human run
    • Logging to CSV, model checkpointing

PIPELINE PHASES
---------------
  0  Calibration  – human drives freely; 4-D style profile extracted.
                    Result saved to save_dir/style_profile.npy and reused.

  1  AV run       – headless PDMORL-TD3 episode using the human's style as
                    preference weights.  UE4 rendering disabled for speed.
                    AVTrajectory (world-space positions) recorded.

  2  Human run    – human drives same scenario with alerts:
                    • Alert MODEL sampled ONCE at episode start.
                    • gui_type, location, color, vibration, lag, gui_params
                      are all FIXED for the entire episode.
                    • Each tick: live AV position queried from AVTrajectory
                      → arrow direction / colour gradient / sound trigger update.
                    • Alerts drawn via world.debug API into the UE4 window.

  3  Training     – experience tuples fed to MoEAlertModel.store_experience();
                    PPO+AWR updates fire automatically when buffer fills.

  → Repeat from Phase 1 until training loss converges or max_iterations reached.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

# ── PATH SETUP ────────────────────────────────────────────────────────────────
# Directory layout:
#   HondaAI/
#     run/                    ← _RUN_ROOT  (PDMORL model .zip files)
#     sb3/                    ← _SB3_ROOT  (CarlaEnv, PDMORL_TD3, …)
#     CARLA-sim/
#       CustomPython/
#         kw_sandbox/         ← _THIS_DIR  (this file)

_THIS_DIR = Path(__file__).resolve().parent            # …/kw_sandbox
_HONDA_AI = _THIS_DIR.parent.parent.parent             # …/HondaAI
_SB3_ROOT = _HONDA_AI / "sb3"
_RUN_ROOT = _HONDA_AI / "run"

for _p in [str(_SB3_ROOT), str(_THIS_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

print(f"[init] sb3 root : {_SB3_ROOT}")
print(f"[init] run root : {_RUN_ROOT}  ({len(list(_RUN_ROOT.glob('*.zip')))} .zip files found)"
      if _RUN_ROOT.exists() else f"[init] run root : {_RUN_ROOT}  (NOT FOUND)")

# ── TYPE-CHECK-ONLY IMPORTS (no runtime cost, silences Pylance warnings) ──────
if TYPE_CHECKING:
    import carla as _carla_t  # noqa: F401

# ── CARLA ─────────────────────────────────────────────────────────────────────
try:
    import carla  # type: ignore[import]
    print("[init] CARLA Python API loaded.")
except ImportError:
    carla = None  # type: ignore[assignment]
    print("[init] WARNING: CARLA not found on sys.path.")

# ── PYGAME (input only — no participant-facing window) ────────────────────────
try:
    import pygame  # type: ignore[import]
    print("[init] pygame loaded.")
except ImportError:
    pygame = None  # type: ignore[assignment]
    print("[init] CRITICAL: pygame not found. Install sb3/requirements.txt.")

# ── PROJECT IMPORTS ───────────────────────────────────────────────────────────
from steering_control import get_wheel_control, get_keyboard_control, ffb_init, ffb_shutdown  # noqa: E402
from alert_models import AlertVector, MoEAlertModel, DEFAULT_STATE_DIM, MAX_LAG  # noqa: E402
from carla_alert_output import AVTrajectory, _play_direction  # noqa: E402
from human_style_regression import HumanStyleRegressor  # noqa: E402

try:
    from carla_gym_env import CarlaEnv          # type: ignore[import]
    from carla_env_utils import CarlaEnvUtils   # type: ignore[import]
    from config import Hyperparameters          # type: ignore[import]
    from pdmorl_train import PDMORL_TD3         # type: ignore[import]
    _SB3_OK = True
    print("[init] sb3 modules loaded (CarlaEnv, PDMORL_TD3, CarlaEnvUtils).")
except ImportError as _e:
    CarlaEnv = CarlaEnvUtils = Hyperparameters = PDMORL_TD3 = None  # type: ignore[assignment]
    _SB3_OK = False
    print(f"[init] WARNING: sb3 unavailable ({_e}). AV phase will fail.")

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
STYLE_LABELS = ["speed", "efficiency", "aggressiveness", "comfort"]
FPS          = 20
_DT          = 1.0 / FPS
_DRAW_LT     = _DT * 1.5          # debug draw lifetime — just over one tick
_MAX_DIST_M  = 80.0               # distance at which colour gradient saturates
_STORE_EVERY = 30                  # store one training sample every N ticks
_CONV_THRESHOLD = 0.02            # loss plateau threshold for convergence
_CONV_WINDOW    = 5               # iterations to average for convergence check

def find_model(explicit_path: str = "") -> str:
    """Resolve the PDMORL model path.

    Priority:
      1. explicit_path if provided on the command line
      2. A file named *bestCombined*.zip in _RUN_ROOT
      3. The most recently modified *.zip in _RUN_ROOT
      4. Raises RuntimeError with a helpful message

    Prints the resolved path so the user can verify the right model was picked.
    """
    if explicit_path:
        p = Path(explicit_path)
        if not p.exists():
            raise FileNotFoundError(f"Model not found: {p.resolve()}")
        print(f"[model] Using explicit path: {p.resolve()}")
        return str(p.resolve())

    if not _RUN_ROOT.exists():
        raise RuntimeError(
            f"run/ folder not found at {_RUN_ROOT}.\n"
            f"Pass --model explicitly, e.g.:\n"
            f"  python alert_pipeline.py --model /path/to/model.zip"
        )

    zips = sorted(_RUN_ROOT.glob("*.zip"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not zips:
        raise RuntimeError(f"No .zip files found in {_RUN_ROOT}. Pass --model explicitly.")

    # Prefer *bestCombined* if it exists
    best = next((z for z in zips if "bestCombined" in z.name), zips[0])
    print(f"[model] Auto-selected model: {best}")
    if best != zips[0]:
        print(f"[model] (Most recent is {zips[0].name} — override with --model if needed)")
    return str(best)


_SCENARIO_MAPS: Dict[str, str] = {
    "intersection": "Town01",
    "traffic_high": "Town02",
    "traffic_low":  "Town02",
    "tunnel":       "Town03",
    "roundabout":   "Town03",
    "highway":      "Town04",
    "crossing":     "Town04",
}


# ── DATA CLASSES ──────────────────────────────────────────────────────────────

@dataclass
class AVStepData:
    sim_time: float
    x: float;  y: float;  z: float
    yaw: float
    speed: float       # m/s from measurements[0]
    steer: float
    throttle: float
    wp_index: int
    total_waypoints: int

    def progress(self) -> float:
        return self.wp_index / max(self.total_waypoints, 1)


@dataclass
class HumanStepData:
    sim_time:     float
    x:            float
    y:            float
    speed:        float
    steer:        float
    throttle:     float
    style_scores: List[float]   # [speed, eff, aggr, comfort]  all in [0,1]
    alert_state:  List[float]   # 14-D state used to sample the episode alert
    alert_raw:    List[float]   # 8-D raw alert vector (FIXED for episode)
    alert_score:  float         # per-tick alignment score


# ── INPUT WINDOW + CONTROL READER ─────────────────────────────────────────────

class ControlReader:
    """Thin wrapper around steering_control.py for wheel + keyboard input.

    Opens a small 200×60 pygame window for keyboard focus.
    The participant looks at the CARLA UE4 window; this window just captures
    key/button events.  It is placed at the top-left corner.

    Wheel (G920):  steer=Axis0, brake=Axis1, throttle=Axis3, handbrake=Button5
    Keyboard:      W/A/S/D, SPACE=handbrake, R=toggle reverse, Q=quit
    """

    def __init__(self) -> None:
        if pygame is None:
            raise RuntimeError("pygame is required for input.")

        pygame.init()
        pygame.joystick.init()
        pygame.display.set_caption("Alert Pipeline — Input")

        # Small, unobtrusive window in the top-left corner
        import os
        os.environ.setdefault("SDL_VIDEO_WINDOW_POS", "0,0")
        self._screen = pygame.display.set_mode((200, 60), pygame.NOFRAME)
        self._screen.fill((20, 20, 20))
        font = pygame.font.SysFont("consolas", 13)
        self._screen.blit(
            font.render("Alert Pipeline input window", True, (180, 180, 180)),
            (4, 22),
        )
        pygame.display.flip()

        self._wheel: Optional[pygame.joystick.JoystickType] = None
        if pygame.joystick.get_count() > 0:
            self._wheel = pygame.joystick.Joystick(0)
            self._wheel.init()
            print(f"[input] Steering wheel detected: {self._wheel.get_name()}")
            print("[input]   Axis 0 = steering | Axis 1 = brake | Axis 3 = throttle")
            ffb_init()
        else:
            print("[input] No wheel detected — keyboard fallback (W/A/S/D).")

        self._control     = carla.VehicleControl()
        self._reverse     = False
        self.quit_request = False

    @property
    def reverse(self) -> bool:
        return self._reverse

    def read(self) -> carla.VehicleControl:
        """Process events and return the current VehicleControl."""
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self.quit_request = True
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_r:
                    self._reverse = not self._reverse
                    print(f"[input] Reverse: {self._reverse}")
                if ev.key == pygame.K_q:
                    self.quit_request = True
            # G920 wheel: button 8 (left paddle shifter) toggles reverse
            if ev.type == pygame.JOYBUTTONDOWN and ev.button == 8:
                self._reverse = not self._reverse
                print(f"[input] Reverse (wheel button 8): {self._reverse}")

        if self._wheel is not None:
            get_wheel_control(self._wheel, self._control, self._reverse)
        else:
            keys = pygame.key.get_pressed()
            get_keyboard_control(keys, self._control, self._reverse)

        return self._control

    def close(self) -> None:
        ffb_shutdown()
        pygame.quit()


# ── CARLA HELPERS ─────────────────────────────────────────────────────────────

def _connect(host: str, port: int, render: bool = True) -> Tuple[object, object]:
    print(f"[carla] Connecting to {host}:{port} …")
    client = carla.Client(host, port)
    client.set_timeout(30.0)
    world  = client.get_world()
    s = world.get_settings()
    s.synchronous_mode    = True
    s.fixed_delta_seconds = _DT
    s.no_rendering_mode   = not render
    world.apply_settings(s)
    print(f"[carla] Connected. Map: {world.get_map().name.split('/')[-1]}  "
          f"render={'ON' if render else 'OFF'}")
    return client, world


def _load_map(client: object, world: object, name: str) -> object:
    current = world.get_map().name.split("/")[-1]  # type: ignore[union-attr]
    if current != name:
        print(f"[carla] Loading map {name} (was {current}) …")
        world = client.load_world(name)  # type: ignore[union-attr]
        s = world.get_settings()  # type: ignore[union-attr]
        s.synchronous_mode    = True
        s.fixed_delta_seconds = _DT
        s.no_rendering_mode   = False
        world.apply_settings(s)  # type: ignore[union-attr]
        print(f"[carla] Map {name} loaded.")
    return world


def _disconnect(world: object) -> None:
    s = world.get_settings()  # type: ignore[union-attr]
    s.synchronous_mode = False
    world.apply_settings(s)  # type: ignore[union-attr]
    print("[carla] Disconnected (synchronous mode off).")


def _spectator_follow(world: object, vehicle: object) -> None:
    tf  = vehicle.get_transform()  # type: ignore[union-attr]
    yaw = math.radians(tf.rotation.yaw)

    # Compensate for 1-tick rendering lag: predict where the car will be next tick
    # so the camera stays locked to the seat rather than drifting back at speed.
    vel = vehicle.get_velocity()  # type: ignore[union-attr]
    pred_x = tf.location.x + vel.x * _DT
    pred_y = tf.location.y + vel.y * _DT
    pred_z = tf.location.z + vel.z * _DT

    # Driver's seat: 0.3 m forward + 0.35 m to the RIGHT in world space
    # (vehicle local +Y is right; in world: right = (sin yaw, -cos yaw)).
    # Previous code used (-sin, +cos) which landed on the passenger side — flip here.
    fwd_offset   = 0.3
    right_offset = 0.35  # positive = driver's right = physically left of car centre (LHD)
    world.get_spectator().set_transform(carla.Transform(  # type: ignore[union-attr]
        carla.Location(
            x=pred_x + fwd_offset * math.cos(yaw) + right_offset * math.sin(yaw),
            y=pred_y + fwd_offset * math.sin(yaw) - right_offset * math.cos(yaw),
            z=pred_z + 1.3,
        ),
        carla.Rotation(pitch=0.0, yaw=tf.rotation.yaw, roll=0.0),
    ))


def _spawn_hero(world: object, spawn_index: int = 8) -> object:
    bp  = world.get_blueprint_library().find("vehicle.lincoln.mkz_2020")  # type: ignore[union-attr]
    bp.set_attribute("role_name", "hero")
    pts = world.get_map().get_spawn_points()  # type: ignore[union-attr]
    # Try the preferred index first, then walk through all points until one is free.
    indices = [spawn_index % len(pts)] + [i for i in range(len(pts)) if i != spawn_index % len(pts)]
    for idx in indices:
        v = world.try_spawn_actor(bp, pts[idx])  # type: ignore[union-attr]
        if v is not None:
            print(f"[spawn] Hero spawned at spawn point {idx}.")
            return v
    raise RuntimeError("Could not spawn hero at any spawn point — all occupied.")


# ── DEBUG ALERT DRAWING (CARLA UE4 WINDOW) ───────────────────────────────────

def _carla_rgb(rgb: Tuple[int, int, int]) -> object:
    return carla.Color(r=rgb[0], g=rgb[1], b=rgb[2])


def _dist_color(dist_m: float, colorblind: bool = False) -> Tuple[int, int, int]:
    t = float(np.clip(dist_m / _MAX_DIST_M, 0.0, 1.0))
    if colorblind:
        return (int(t * 230), int(114 - t * 114), int(178 - t * 178))
    return (int(t * 220), int((1 - t) * 200), 0)


# Module-level sound cooldown (reset at episode start)
_last_sound_t: float = -999.0


def draw_alert(
    world:    object,
    alert:    AlertVector,
    hero:     object,
    traj:     AVTrajectory,
    sim_time: float,
) -> None:
    """Draw one frame of alerts into the UE4 world via CARLA debug API.

    Parameters are fixed from the episode-start sample.
    Only the AV's live world position (from AVTrajectory) changes.
    """
    if alert.gui_type == 0:
        _draw_arrow(world, alert, hero, traj, sim_time)
    elif alert.gui_type == 1:
        _draw_route(world, alert, hero, traj)
    elif alert.gui_type == 2:
        _tick_sound(alert, hero, traj, sim_time)


def _draw_arrow(world, alert, hero, traj, sim_time):
    # Look up where the AV will be lag seconds from now — NOT its current position.
    look_ahead_t = sim_time + alert.lag
    target = traj.get_position_at(look_ahead_t)
    if target is None:
        return

    scale    = float(alert.gui_params[0])
    vib_dist = float(alert.gui_params[2]) * _MAX_DIST_M

    hloc = hero.get_location()
    htf  = hero.get_transform()
    yaw  = math.radians(htf.rotation.yaw)

    # Distance to AV for color
    dist = math.sqrt((target[0] - hloc.x)**2 + (target[1] - hloc.y)**2)
    color = _dist_color(dist, bool(alert.color))

    # Windshield anchor: 1.3 m ahead of driver, at eye height
    ws_x      = hloc.x + 1.3 * math.cos(yaw)
    ws_y      = hloc.y + 1.3 * math.sin(yaw)
    ws_z      = hloc.z + 1.25   # windshield centre height

    # ── Clock-hand direction ──────────────────────────────────────────────────
    # Project the hero→AV horizontal vector onto the driver's frame:
    #   forward component → vertical on windshield (up = AV ahead, down = AV behind)
    #   lateral component → horizontal on windshield (right = AV to the right)
    fwd_x  =  math.cos(yaw)   # vehicle forward in world X
    fwd_y  =  math.sin(yaw)   # vehicle forward in world Y
    right_x =  math.sin(yaw)  # vehicle right in world X
    right_y = -math.cos(yaw)  # vehicle right in world Y

    dx = target[0] - hloc.x
    dy = target[1] - hloc.y

    forward_comp = dx * fwd_x   + dy * fwd_y    # + = AV is ahead  → arrow up
    lateral_comp = dx * right_x + dy * right_y  # + = AV is right  → arrow right

    mag = math.sqrt(forward_comp ** 2 + lateral_comp ** 2) + 1e-6
    fwd_n = forward_comp / mag
    lat_n = lateral_comp / mag

    arm = 0.12 + scale * 0.04   # clock-hand length in metres (small)

    tip_x = ws_x + lat_n * right_x * arm   # left/right offset on windshield
    tip_y = ws_y + lat_n * right_y * arm
    tip_z = ws_z + fwd_n * arm              # up if AV ahead, down if AV behind

    # Thin, non-glowing, very transparent — no glow
    world.debug.draw_arrow(
        begin=carla.Location(x=ws_x, y=ws_y, z=ws_z),
        end=carla.Location(  x=tip_x, y=tip_y, z=tip_z),
        thickness=0.010,
        arrow_size=0.04,
        color=carla.Color(r=color[0], g=color[1], b=color[2], a=12),
        life_time=_DRAW_LT,
    )
    if bool(alert.vibration) and dist > vib_dist:
        world.debug.draw_string(
            location=carla.Location(x=hloc.x, y=hloc.y, z=hloc.z + 3.0),
            text="! STEER !",
            color=_carla_rgb((255, 80, 0)),
            life_time=_DRAW_LT,
        )


def _draw_route(world, alert, hero, traj):
    positions = traj.all_positions()
    if len(positions) < 2:
        return

    line_w   = max(0.09, float(alert.gui_params[0]) * 0.2)  # min 3× original 0.03
    opacity  = float(alert.gui_params[1])
    vib_dist = float(alert.gui_params[2]) * _MAX_DIST_M

    hloc    = hero.get_location()
    nearest = min(math.sqrt((p[0]-hloc.x)**2 + (p[1]-hloc.y)**2) for p in positions)
    color   = _dist_color(nearest, bool(alert.color))

    if bool(alert.vibration) and nearest > vib_dist:
        world.debug.draw_string(
            location=carla.Location(x=hloc.x, y=hloc.y, z=hloc.z + 3.0),
            text="! OFF ROUTE !",
            color=_carla_rgb((255, 80, 0)),
            life_time=_DRAW_LT,
        )

    # Draw every Nth segment to keep debug budget low
    step = max(1, len(positions) // 200)
    for i in range(0, len(positions) - step, step):
        a, b = positions[i], positions[i + step]
        world.debug.draw_line(
            begin=carla.Location(x=a[0], y=a[1], z=a[2] + 0.3),
            end=carla.Location(  x=b[0], y=b[1], z=b[2] + 0.3),
            thickness=line_w * opacity,
            color=_carla_rgb(color),
            life_time=_DRAW_LT,
        )


def _tick_sound(alert, hero, traj, sim_time):
    global _last_sound_t
    lat_thresh = float(alert.gui_params[0]) * 10.0    # [0,1] → [0,10] m
    cooldown   = float(alert.gui_params[1]) * 150.0  # [0,1] → [0,150] s (15× reduction)
    volume     = float(alert.gui_params[2])

    if sim_time - _last_sound_t < cooldown:
        return
    offset = traj.lateral_offset_at(sim_time + alert.lag, hero)
    if offset is None or abs(offset) < lat_thresh:
        return

    direction = "right" if offset > 0 else "left"
    _play_direction(direction, volume)
    _last_sound_t = sim_time
    print(f"[sound] Played '{direction}' (lateral offset={offset:.2f}m)")


# ── STATE BUILDER ─────────────────────────────────────────────────────────────

def _build_state(
    hero:          object,
    hero_meas:     np.ndarray,
    av_step:       AVStepData,
    route_prog:    float,
    next_wp_angle: float,
) -> np.ndarray:
    """Build the 14-D alert model input state."""
    tf   = hero.get_transform()
    hloc = tf.location
    hyaw = math.radians(tf.rotation.yaw)

    dx   = av_step.x - hloc.x
    dy   = av_step.y - hloc.y
    dist = math.sqrt(dx*dx + dy*dy)

    return np.array([
        float(np.clip(dx / 100.0,                  -1.0,  1.0)),
        float(np.clip(dy / 100.0,                  -1.0,  1.0)),
        float(math.sin(math.radians(av_step.yaw))),
        float(np.clip(av_step.speed / 30.0,         0.0,  1.0)),
        float(av_step.steer),
        float(av_step.throttle),
        float(np.clip(float(hero_meas[0]) / 30.0,   0.0,  1.0)),
        float(math.sin(hyaw)),
        float(hero_meas[9]),
        float(np.clip(dist / 100.0,                 0.0,  1.0)),
        float((math.atan2(dy, dx) - hyaw) / math.pi),
        float(np.clip((av_step.speed - float(hero_meas[0])) / 30.0, -1.0, 1.0)),
        float(next_wp_angle / math.pi),
        float(np.clip(route_prog, 0.0, 1.0)),
    ], dtype=np.float32)


def _deviation_score(human: np.ndarray, target: np.ndarray) -> float:
    return max(0.0, 1.0 - float(np.mean(np.abs(human - target))))


def _nearest_av(steps: List[AVStepData], t: float) -> AVStepData:
    if not steps:
        return AVStepData(0, 0, 0, 0, 0, 0, 0, 0, 0, 1)
    lo, hi = 0, len(steps) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if steps[mid].sim_time <= t:
            lo = mid
        else:
            hi = mid
    return steps[lo]


# ── PHASE 0: CALIBRATION ──────────────────────────────────────────────────────

def run_calibration(
    client:       object,
    world:        object,
    duration_sec: int,
) -> np.ndarray:
    """Human drives freely. CARLA UE4 window is the only display.
    Returns 4-D normalised style profile [speed, eff, aggr, comfort].
    """
    print(f"\n{'='*60}")
    print(f"PHASE 0  CALIBRATION  ({duration_sec}s free drive)")
    print("Watch the CARLA UE4 window. Drive freely to calibrate your style.")
    print(f"{'='*60}")

    hero      = _spawn_hero(world)
    regressor = HumanStyleRegressor()
    regressor.attach_collision_sensor(world, hero)
    print("[calibration] Collision sensor attached.")

    ctrl_reader = ControlReader()
    clock   = pygame.time.Clock()
    tick = 0
    start_t = time.time()

    try:
        while True:
            elapsed   = time.time() - start_t
            remaining = int(duration_sec - elapsed)
            if remaining <= 0 or ctrl_reader.quit_request:
                print("[calibration] Duration reached, ending calibration.")
                break

            ctrl = ctrl_reader.read()
            hero.apply_control(ctrl)
            world.tick()
            clock.tick(FPS)   # cap to FPS so sim time ≈ real time

            profile = regressor.tick(hero, world, _DT)
            _spectator_follow(world, hero)

            # Draw HUD on the windshield (same plane as the arrow)
            loc = hero.get_location()
            htf = hero.get_transform()
            _yaw = math.radians(htf.rotation.yaw)
            ws_x = loc.x + 1.3 * math.cos(_yaw)
            ws_y = loc.y + 1.3 * math.sin(_yaw)
            rev_tag = "  [REVERSE]" if ctrl_reader.reverse else ""
            world.debug.draw_string(
                location=carla.Location(x=ws_x, y=ws_y, z=loc.z + 1.45),
                text=(f"CAL {remaining}s{rev_tag} | "
                      + "  ".join(f"{l[0]}:{v:.2f}"
                                  for l, v in zip(STYLE_LABELS, profile))),
                color=carla.Color(r=220, g=220, b=220),
                life_time=_DRAW_LT,
            )

            tick += 1
            if tick % (FPS * 5) == 0:
                print(f"  [cal] {elapsed:.0f}s/{duration_sec}s  "
                      + "  ".join(f"{l}={v:.3f}" for l, v in zip(STYLE_LABELS, profile)))

    finally:
        ctrl_reader.close()
        regressor.destroy_collision_sensor()
        hero.destroy()
        print("[calibration] Hero destroyed.")

    profile = regressor.get_style_profile()
    print(f"\n[calibration] DONE → "
          + "  ".join(f"{l}={v:.3f}" for l, v in zip(STYLE_LABELS, profile)))
    return profile


# ── PHASE 1: AV HEADLESS RUN ──────────────────────────────────────────────────

def run_av_episode(
    model_path:    str,
    style_profile: np.ndarray,
    scenario:      str,
    port:          int = 2000,
) -> Tuple[AVTrajectory, List[AVStepData]]:
    """Headless AV episode. CarlaEnv manages its own CARLA connection.
    UE4 rendering is disabled during this phase for speed.
    Returns (trajectory, per-step data).
    """
    if not _SB3_OK:
        raise RuntimeError("sb3 modules not available — AV phase cannot run.")

    print(f"\n{'='*60}")
    print(f"PHASE 1  AV HEADLESS RUN  scenario={scenario}")
    print(f"         pref weights = {np.round(style_profile, 3).tolist()}")
    print("         UE4 window will go dark — this is normal.")
    print(f"{'='*60}")

    config                       = Hyperparameters()
    config.scenario              = scenario
    config.next_map              = _SCENARIO_MAPS.get(scenario, "Town01")
    config.evaluate              = True
    config.client_port           = port
    config.SPECATE               = False   # disables UE4 rendering (no_rendering_mode=True)
    config.waypoint_timeout_ticks = 10_000  # ~500 s at 20 FPS — lets AV finish full route
    config.NUM_TRAFFIC_VEHICLES   = 0       # no NPC traffic during AV phase

    _av_client = carla.Client("localhost", port)
    _av_client.set_timeout(30.0)
    env = CarlaEnv(_av_client, config)

    # Inject the calibrated style as fixed preference weights.
    # get_pref_weights_step() is called inside get_observation() every tick.
    _w = style_profile.astype(np.float32).copy()
    env.pref_weights_round    = _w
    env.get_pref_weights      = lambda: None      # no-op; preserves _w set above
    env.get_pref_weights_step = lambda: _w.copy()
    print(f"[av_run] Preference weights injected: {_w.tolist()}")

    model = PDMORL_TD3.load(model_path)
    print(f"[av_run] Model loaded: {model_path}")

    traj:  AVTrajectory     = AVTrajectory()
    steps: List[AVStepData] = []
    sim_t  = 0.0
    done   = False
    tick   = 0

    obs, _ = env.reset()
    total  = max(len(env.route), 1)
    print(f"[av_run] Environment reset. Route: {total} waypoints.")

    try:
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = env.step(action)
            sim_t += _DT

            v    = env.vehicle
            tf   = v.get_transform()
            ctrl = v.get_control()
            meas = CarlaEnvUtils.get_vehicle_measurements(v)

            traj.add(sim_t, v)
            steps.append(AVStepData(
                sim_time=sim_t,
                x=tf.location.x, y=tf.location.y, z=tf.location.z,
                yaw=tf.rotation.yaw,
                speed=float(meas[0]),
                steer=float(ctrl.steer),
                throttle=float(ctrl.throttle),
                wp_index=int(info.get("current_wp_index", 0))
                         if isinstance(info, dict) else 0,
                total_waypoints=total,
            ))
            tick += 1
            if tick % (FPS * 10) == 0:
                print(f"  [av_run] t={sim_t:.1f}s  wp={steps[-1].wp_index}/{total}"
                      f"  speed={steps[-1].speed:.1f}m/s")
    finally:
        env.close()
        print("[av_run] CarlaEnv closed.")

    prog = steps[-1].progress() * 100 if steps else 0.0
    print(f"\n[av_run] DONE → {len(steps)} steps, route completion {prog:.1f}%")
    return traj, steps


# ── PHASE 2: HUMAN RUN WITH ALERTS ────────────────────────────────────────────

def run_human_episode(
    client:       object,
    world:        object,
    traj:         AVTrajectory,
    av_steps:     List[AVStepData],
    av_style:     np.ndarray,
    alert_model:  MoEAlertModel,
    max_duration: float = 300.0,
    force_arrow:  bool  = False,
) -> List[HumanStepData]:
    """Human drives with alert overlays drawn into the UE4 window.

    Alert model is sampled ONCE at episode start.
    GUI type and all parameters are fixed for the entire episode.
    Only positional rendering updates (arrow direction, colour, sound trigger).
    """
    print(f"\n{'='*60}")
    print(f"PHASE 2  HUMAN RUN WITH ALERTS  (max {int(max_duration)}s)")
    print("Watch the CARLA UE4 window. Follow the route shown by the alert.")
    print(f"{'='*60}")

    # Destroy any NPC vehicles left over from the AV phase before spawning hero.
    for actor in world.get_actors().filter("vehicle.*"):  # type: ignore[union-attr]
        actor.destroy()
    print("[human_run] Cleared leftover traffic actors.")

    hero      = _spawn_hero(world)
    regressor = HumanStyleRegressor()
    regressor.attach_collision_sensor(world, hero)
    print("[human_run] Hero spawned and collision sensor attached.")

    # ── Sample alert ONCE ─────────────────────────────────────────────────
    # Use the calibrated style profile as the initial state.  Since style is
    # constant for the episode, the model sees a stable distribution.
    first_av  = av_steps[0] if av_steps else AVStepData(0,0,0,0,0,0,0,0,0,1)
    init_meas = CarlaEnvUtils.get_vehicle_measurements(hero)
    init_state = _build_state(hero, init_meas, first_av, 0.0, 0.0)
    alert, log_prob = alert_model.sample(init_state)
    if force_arrow:
        alert.gui_type   = 0
        alert.gui_params = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        # Guarantee a visible lag so the arrow shows a future position, not the
        # AV's current position.  Clamp to MAX_LAG in case the model went high.
        alert.lag = float(np.clip(max(alert.lag, 1.0), 0.0, MAX_LAG))
        print(f"[human_run] force_arrow=True — gui_type=arrow, lag={alert.lag:.2f}s")
    alert_raw = alert.to_raw()

    print(f"\n[human_run] ── Episode alert (FIXED for this run) ──────────────")
    print(f"            GUI type  : {alert.gui_name}  (index {alert.gui_type})")
    print(f"            Location  : {'windshield' if alert.location else 'panel'}")
    print(f"            Color     : {'colorblind' if alert.color else 'standard RGB'}")
    print(f"            Vibration : {'enabled' if alert.vibration else 'disabled'}")
    print(f"            Lag       : {alert.lag:.2f}s")
    for n, v in zip(alert.param_names, alert.gui_params):
        print(f"            {n:<18}: {v:.3f}")
    print(f"[human_run] ────────────────────────────────────────────────────\n")
    # ─────────────────────────────────────────────────────────────────────

    global _last_sound_t
    _last_sound_t = -999.0   # reset sound cooldown for new episode

    av_end_t    = av_steps[-1].sim_time if av_steps else max_duration
    ctrl_reader = ControlReader()
    clock       = pygame.time.Clock()
    step_log: List[HumanStepData] = []
    sim_t = 0.0
    tick  = 0

    try:
        while True:
            if sim_t >= max(max_duration, av_end_t):
                print(f"[human_run] Episode duration reached ({sim_t:.1f}s).")
                break
            if ctrl_reader.quit_request:
                print("[human_run] User ended episode early (Q pressed).")
                break

            ctrl = ctrl_reader.read()
            hero.apply_control(ctrl)
            world.tick()
            clock.tick(FPS)   # cap to FPS so sim time ≈ real time
            sim_t += _DT
            tick  += 1

            meas        = CarlaEnvUtils.get_vehicle_measurements(hero)
            human_style = regressor.tick(hero, world, _DT)
            av_step     = _nearest_av(av_steps, sim_t)
            progress    = float(np.clip(sim_t / max(av_end_t, 1e-9), 0.0, 1.0))

            next_pos = traj.get_position_at(sim_t + 1.0)
            if next_pos is not None:
                hloc   = hero.get_location()
                hyaw   = math.radians(hero.get_transform().rotation.yaw)
                wp_ang = (math.atan2(next_pos[1] - hloc.y,
                                     next_pos[0] - hloc.x) - hyaw)
            else:
                wp_ang = 0.0

            score = _deviation_score(human_style, av_style)

            # ── CARLA debug rendering ─────────────────────────────────────
            draw_alert(world, alert, hero, traj, sim_t)
            _spectator_follow(world, hero)

            loc = hero.get_location()
            world.debug.draw_string(
                location=carla.Location(x=loc.x, y=loc.y, z=loc.z + 5.0),
                text=(f"t={sim_t:.0f}s  align={score:.2f}  "
                      + "  ".join(f"{l[0]}:{v:.2f}"
                                  for l, v in zip(STYLE_LABELS, human_style))),
                color=carla.Color(r=200, g=230, b=200),
                life_time=_DRAW_LT,
            )

            # ── Store training experience every _STORE_EVERY ticks ────────
            if tick % _STORE_EVERY == 0:
                cur_state = _build_state(hero, meas, av_step, progress, wp_ang)
                alert_model.store_experience(cur_state, alert, score, log_prob)

            if tick % (FPS * 10) == 0:
                print(f"  [human_run] t={sim_t:.1f}s  align={score:.3f}  "
                      + "  ".join(f"{l}={v:.3f}"
                                  for l, v in zip(STYLE_LABELS, human_style)))

            step_log.append(HumanStepData(
                sim_time=sim_t,
                x=loc.x, y=loc.y,
                speed=float(meas[0]),
                steer=float(ctrl.steer),
                throttle=float(ctrl.throttle),
                style_scores=human_style.tolist(),
                alert_state=init_state.tolist(),   # fixed for episode
                alert_raw=alert_raw.tolist(),       # fixed for episode
                alert_score=score,
            ))

    finally:
        ctrl_reader.close()
        regressor.destroy_collision_sensor()
        hero.destroy()
        print("[human_run] Hero destroyed.")

    mean_score = np.mean([s.alert_score for s in step_log]) if step_log else 0.0
    print(f"\n[human_run] DONE → {len(step_log)} steps, "
          f"mean alignment score={mean_score:.3f}")
    return step_log


# ── PHASE 3: TRAINING FLUSH ───────────────────────────────────────────────────

def flush_training(alert_model: MoEAlertModel, step_log: List[HumanStepData]) -> float:
    """Resubmit all episode steps so the model can drain its buffer fully."""
    import torch  # deferred — avoids slow import at the top

    print(f"\n{'='*60}")
    print(f"PHASE 3  TRAINING FLUSH  ({len(step_log)} steps)")

    losses: List[float] = []
    for step in step_log:
        state = np.array(step.alert_state, dtype=np.float32)
        av    = AlertVector.from_raw(np.array(step.alert_raw, dtype=np.float32))

        st = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            gl  = alert_model.gating(st)
            lp_g = torch.distributions.Categorical(logits=gl).log_prob(
                torch.tensor([av.gui_type]))
            mu, sd = alert_model.experts[av.gui_type](st)
            lp_p = torch.distributions.Normal(mu, sd).log_prob(
                torch.FloatTensor(av.gui_params).unsqueeze(0)).sum(-1)
            lp   = (lp_g + lp_p).item() / 8.0

        loss = alert_model.store_experience(state, av, step.alert_score, lp)
        if loss is not None:
            losses.append(loss)

    mean_loss = float(np.mean(losses)) if losses else float("nan")
    print(f"[training] Updates: {len(losses)}  Mean loss: {mean_loss:.4f}")
    return mean_loss


# ── PIPELINE ORCHESTRATOR ─────────────────────────────────────────────────────

class AlertPipeline:

    def __init__(
        self,
        model_path:           str,
        scenario:             str  = "intersection",
        calibration_duration: int  = 20,
        max_iterations:       int  = 20,
        host:                 str  = "localhost",
        port:                 int  = 2000,
        save_dir:             str  = "./pipeline_runs",
    ) -> None:
        self.model_path           = model_path
        self.scenario             = scenario
        self.calibration_duration = calibration_duration
        self.max_iterations       = max_iterations
        self.host                 = host
        self.port                 = port
        self.save_dir             = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.alert_model    = MoEAlertModel(state_dim=DEFAULT_STATE_DIM)
        self.style_profile: Optional[np.ndarray] = None
        self._loss_history: deque = deque(maxlen=_CONV_WINDOW)
        print(f"[pipeline] Ready. Saving to: {self.save_dir.resolve()}")

    # ── Persistence ──────────────────────────────────────────────────────

    def _save(self, iteration: int) -> None:
        import torch
        p = self.save_dir / f"alert_model_iter{iteration:03d}.pt"
        torch.save(self.alert_model.state_dict(), p)
        np.save(self.save_dir / "style_profile.npy", self.style_profile)
        print(f"[pipeline] Saved model to {p}")

    def _load_style(self) -> bool:
        p = self.save_dir / "style_profile.npy"
        if p.exists():
            self.style_profile = np.load(str(p))
            print(f"[pipeline] Loaded cached style: {self.style_profile.tolist()}")
            return True
        return False

    def _save_csv(self, step_log: List[HumanStepData], iteration: int) -> None:
        if not step_log:
            return
        p = self.save_dir / f"human_run_iter{iteration:03d}.csv"
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=asdict(step_log[0]).keys())
            w.writeheader()
            for s in step_log:
                w.writerow(asdict(s))
        print(f"[pipeline] CSV saved to {p}")

    def _converged(self) -> bool:
        if len(self._loss_history) < _CONV_WINDOW:
            return False
        r = list(self._loss_history)
        return (max(r) - min(r)) < _CONV_THRESHOLD

    # ── Run ──────────────────────────────────────────────────────────────

    def run(self) -> None:
        target_map = _SCENARIO_MAPS.get(self.scenario, "Town01")

        # Phase 0 — calibration (always run, never skipped)
        print(f"\n[pipeline] Starting calibration (map={target_map}) …")
        client, world = _connect(self.host, self.port, render=True)
        world = _load_map(client, world, target_map)
        try:
            self.style_profile = run_calibration(
                client, world, self.calibration_duration)
        finally:
            _disconnect(world)
        print(f"[pipeline] Style profile: {self.style_profile.tolist()}")

        # Iteration loop
        for it in range(self.max_iterations):
            print(f"\n{'#'*60}")
            print(f"# ITERATION {it+1} / {self.max_iterations}")
            print(f"{'#'*60}")

            # Phase 1 — headless AV run (CarlaEnv handles its own connection)
            traj, av_steps = run_av_episode(
                self.model_path, self.style_profile,
                self.scenario, self.port,
            )

            # Phase 2 — human run (direct connection, rendering ON)
            print(f"\n[pipeline] Re-enabling UE4 rendering for human run …")
            client, world = _connect(self.host, self.port, render=True)
            world = _load_map(client, world, target_map)
            try:
                step_log = run_human_episode(
                    client, world, traj, av_steps,
                    self.style_profile, self.alert_model,
                    force_arrow=(it == 0),
                )
            finally:
                _disconnect(world)

            # Phase 3 — train
            loss = flush_training(self.alert_model, step_log)
            self._loss_history.append(loss)
            self._save_csv(step_log, it)
            self._save(it)

            print(f"\n[pipeline] Iteration {it+1} done.  Loss={loss:.4f}  "
                  f"History={[f'{v:.4f}' for v in self._loss_history]}")

            if self._converged():
                rng = max(self._loss_history) - min(self._loss_history)
                print(f"\n[pipeline] CONVERGED (loss range={rng:.4f} < {_CONV_THRESHOLD})"
                      f" after {it+1} iterations.")
                break

        print("\n[pipeline] All iterations complete.")


# ── SMOKE TEST (--test flag, no CARLA required) ───────────────────────────────

def _run_smoke_test() -> None:
    """Basic unit tests runnable without CARLA or a model.  python alert_pipeline.py --test"""
    print("\n" + "="*60)
    print("SMOKE TEST")
    print("="*60)
    failures = 0

    def _check(name: str, cond: bool) -> None:
        nonlocal failures
        status = "PASS" if cond else "FAIL"
        print(f"  [{status}] {name}")
        if not cond:
            failures += 1

    # AVStepData
    s = AVStepData(1.0, 10.0, 5.0, 0.0, 0.0, 5.5, 0.1, 0.5, 50, 200)
    _check("AVStepData.progress()", abs(s.progress() - 0.25) < 1e-6)

    # deviation_score
    a = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    b = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    _check("deviation_score: identical → 1.0", abs(_deviation_score(a, b) - 1.0) < 1e-6)
    c = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    d = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    _check("deviation_score: opposite → 0.0", abs(_deviation_score(c, d)) < 1e-6)

    # _dist_color
    rgb_near = _dist_color(0.0,  False)
    rgb_far  = _dist_color(80.0, False)
    _check("dist_color: near = mostly green", rgb_near[1] > rgb_near[0])
    _check("dist_color: far  = mostly red",  rgb_far[0]  > rgb_far[1])
    cb_near  = _dist_color(0.0, True)
    _check("dist_color: colorblind near = mostly blue", cb_near[2] > cb_near[0])

    # AVTrajectory
    from carla_alert_output import AVTrajectory as AT, _TrajectoryPoint
    traj = AT()
    for i in range(100):
        traj._points.append(_TrajectoryPoint(t=float(i)*0.05, x=float(i), y=0.0, z=0.0, yaw=0.0))
    pos = traj.get_position_at(2.5)
    _check("AVTrajectory interpolation", pos is not None and abs(pos[0] - 50.0) < 0.5)

    # HumanStepData round-trip
    hs = HumanStepData(1.0, 0.0, 0.0, 5.0, 0.1, 0.3,
                       [0.5]*4, [0.0]*14, [0.0]*8, 0.8)
    d  = asdict(hs)
    _check("HumanStepData dict round-trip", len(d) == 10)

    # _nearest_av binary search
    steps = [AVStepData(float(i)*0.05, 0,0,0,0,0,0,0,0,1) for i in range(200)]
    found = _nearest_av(steps, 5.0)
    _check("_nearest_av binary search", abs(found.sim_time - 5.0) < 0.06)

    print(f"\n{'='*60}")
    if failures == 0:
        print("All tests PASSED.")
    else:
        print(f"{failures} test(s) FAILED.")
    print("="*60)


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Adaptive alert pipeline — see module docstring for full usage.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", default="",
                   help="Path to PDMORL_TD3 .zip model file. "
                        "If omitted, auto-selects *bestCombined*.zip from "
                        f"{_RUN_ROOT}")
    p.add_argument("--scenario", default="intersection",
                   choices=list(_SCENARIO_MAPS.keys()))
    p.add_argument("--calibration-duration", type=int, default=20,
                   help="Calibration free-drive duration in seconds")
    p.add_argument("--max-iterations", type=int, default=20)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--save-dir", default="./pipeline_runs")
    p.add_argument("--test", action="store_true",
                   help="Run smoke tests (no CARLA or model needed) then exit")
    args = p.parse_args()

    if args.test:
        _run_smoke_test()
        return

    model_path = find_model(args.model)   # resolves or raises with a clear message

    AlertPipeline(
        model_path=model_path,
        scenario=args.scenario,
        calibration_duration=args.calibration_duration,
        max_iterations=args.max_iterations,
        host=args.host,
        port=args.port,
        save_dir=args.save_dir,
    ).run()


if __name__ == "__main__":
    main()
