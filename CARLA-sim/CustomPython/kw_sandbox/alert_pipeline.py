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
import copy
import csv
import math
import os
import random
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, field, replace as dc_replace
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


# ── TYPE-CHECK-ONLY IMPORTS (no runtime cost, silences Pylance warnings) ──────
if TYPE_CHECKING:
    import carla as _carla_t  # noqa: F401

# ── CARLA ─────────────────────────────────────────────────────────────────────
try:
    import carla  # type: ignore[import]
except ImportError:
    carla = None  # type: ignore[assignment]

# ── PYGAME (input only — no participant-facing window) ────────────────────────
try:
    import pygame  # type: ignore[import]
except ImportError:
    pygame = None  # type: ignore[assignment]

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
except ImportError as _e:
    CarlaEnv = CarlaEnvUtils = Hyperparameters = PDMORL_TD3 = None  # type: ignore[assignment]
    _SB3_OK = False

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
STYLE_LABELS = ["speed", "efficiency", "aggressiveness", "comfort"]
FPS          = 20
_DT          = 1.0 / FPS
_DRAW_LT     = _DT * 1.5          # debug draw lifetime — just over one tick
_MAX_DIST_M  = 80.0               # distance used for vibration threshold scaling
_COLOR_DIST_M = 30.0              # distance at which colour gradient fully saturates red
_CAR_LENGTH_M        = 4.5        # approximate vehicle length (m)
_CAR_WIDTH_M         = 2.0        # approximate vehicle width (m)
_HALF_LANE_WIDTH     = 1.75       # m from lane centre to dashed line — min clearance from obstacle
_WP_CLEAR_RADIUS     = _CAR_LENGTH_M * 2          # 9.0 m — hard zone: full peak offset applied here
_WP_TRANSITION_MULT  = 3.5        # transition extends to this multiple of _WP_CLEAR_RADIUS (31.5 m)
_DETOUR_PEAK_M       = 1.0        # peak lateral detour offset in metres (scene-by-scene tuning)
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


# ── MINI SCENARIO DEFINITIONS ─────────────────────────────────────────────────

@dataclass
class MiniScenario:
    """A short obstacle scenario layered on top of an existing CarlaEnv scenario.

    Obstacles are placed relative to a fixed map spawn point (spawn_index) using
    forward/right offsets, so they always land on the same road segment regardless
    of physics drift.

    Attributes:
        name          : human-readable label shown in pipeline output
        base_scenario : CarlaEnv scenario name used for the AV route
        route_length  : max_waypoints cap — keeps the run to 1-3 blocks
        spawn_index   : index into world.get_map().get_spawn_points() used as the
                        reference origin for obstacle placement AND the human hero
                        start position.  Must be on (or very near) the AV's route
                        so both drivers encounter the obstacles.
        obstacles     : list of (forward_m, right_m, blueprint_id) tuples for
                        STATIC parked vehicles (physics disabled, never move).
                        forward_m    — metres ahead along the spawn-point heading
                        right_m      — metres right of that heading (negative = left)
                        blueprint_id — CARLA vehicle blueprint string
        npc_autopilot : list of (forward_m, right_m, blueprint_id) tuples for
                        slow-moving NPC vehicles spawned with autopilot enabled.
                        Place them 40–80 m ahead so the AV/human must overtake or
                        follow.  Keep this list short (0–2 vehicles) so the AV can
                        handle them without collision.
        npc_crash     : list of (forward_m, right_m, blueprint_id, steer) tuples
                        for vehicles that immediately veer off-road and crash.
                        forward_m — spawn distance ahead (20–40 m recommended)
                        right_m   — lateral offset at spawn (0.0 = lane centre)
                        steer     — steering applied instantly: +1.0 = hard right
                                    (off right kerb), -1.0 = hard left (hits barrier
                                    or opposing-lane wall).  Use 0.7–1.0 magnitude.
                        Physics is enabled; a throttle=0.7 control is applied once
                        and persists, so the vehicle drives itself off the road and
                        creates a wreck the driver must navigate around.
    """
    name:          str
    base_scenario: str
    route_length:  int
    spawn_index:   int                                    = 8
    obstacles:     List[Tuple[float, float, str]]         = field(default_factory=list)
    npc_autopilot: List[Tuple[float, float, str]]         = field(default_factory=list)
    npc_crash:     List[Tuple[float, float, str, float]]  = field(default_factory=list)
    detour_peak_m: Optional[float]                        = None  # overrides _DETOUR_PEAK_M when set
    route_type:    Optional[str]                          = None  # "straight_left" / "line_then_left"
    route_params:  Optional[dict]                         = None  # extra params for route_type


# ── Original 3 scenarios (Town01, spawn 8) ────────────────────────────────────
# Obstacles are parked cars placed ~1.5 m to the side of the lane centre so
# the AV can pass without collision while the human driver must steer around
# them.  Adjust forward_m / right_m if the obstacles land in the wrong spot.
#
# ── 10 additional scenarios (Town01–Town04, various spawn points) ─────────────
# Spawn indices are chosen to land the hero on a straight, unobstructed road
# segment that also sits on the AV's CarlaEnv route for the matching
# base_scenario.  If an index is blocked at runtime _spawn_hero() falls back
# to the nearest free point automatically.
MINI_SCENARIOS: List[MiniScenario] = [
    # ── original ──────────────────────────────────────────────────────────────
    MiniScenario(
        name          = "parked_right",
        base_scenario = "intersection",
        route_length  = 160,
        spawn_index   = 8,
        obstacles     = [
            (20.0,  2.5, "vehicle.tesla.model3"),   # parked car 1 m further right
        ],
        detour_peak_m = -0.5,  # 0.5 m to the right
    ),
    MiniScenario(
        name          = "parked_left",
        base_scenario = "intersection",
        route_length  = 160,
        spawn_index   = 8,
        obstacles     = [
            (20.0, -1.5, "vehicle.tesla.model3"),   # parked car on left side of lane
        ],
    ),
    MiniScenario(
        name          = "double_park",
        base_scenario = "intersection",
        route_length  = 200,
        spawn_index   = 8,
        obstacles     = [
            (20.0,  1.5, "vehicle.tesla.model3"),   # right side 20 m ahead
            (45.0, -1.5, "vehicle.audi.a2"),         # left side  45 m ahead
        ],
    ),

    # ── new: Town01 variants (CarlaEnv spawnpoint=8 for "intersection") ──────────
    MiniScenario(
        name          = "town01_far_parked",
        base_scenario = "intersection",
        route_length  = 200,
        spawn_index   = 8,           # matches CarlaEnv spawnpoint for "intersection"
        obstacles     = [
            (78.0, -30.5, "vehicle.nissan.micra", -90),  # around corner, rotated 90° user-right
        ],
        detour_peak_m = 0.0,   # no route shift — obstacle is far off road
    ),
    MiniScenario(
        name          = "town01_chicane",
        base_scenario = "intersection",
        route_length  = 240,
        spawn_index   = 8,
        obstacles     = [
            (36.0, -1.5, "vehicle.mini.cooperst"),    # left   36 m
            (54.0,  1.5, "vehicle.citroen.c3"),       # right  54 m
            (0, 0, "vehicle.nissan.micra", -90, 138.0, 1.5),  # absolute world coords, detour enabled
        ],
    ),
    MiniScenario(
        name          = "town01_tight_gap",
        base_scenario = "intersection",
        route_length  = 180,
        spawn_index   = 8,
        obstacles     = [],
        route_type    = "line_then_left",
        route_params  = {"target_y": 55.5, "turn_x": 100.0, "after_turn": "straight"},
    ),
    MiniScenario(
        name          = "town01_triple_stagger",
        base_scenario = "intersection",
        route_length  = 260,
        spawn_index   = 8,
        obstacles     = [],
        route_type    = "line_then_left",
        route_params  = {"target_y": 55.5, "turn_x": 100.0},
    ),

    # ── new: Town02 variants (CarlaEnv spawnpoint=57 for "traffic_low/high") ────
    MiniScenario(
        name          = "town02_parked_right",
        base_scenario = "traffic_low",
        route_length  = 160,
        spawn_index   = 57,          # matches CarlaEnv spawnpoint for "traffic_low"
        obstacles     = [
            (22.0,  1.5, "vehicle.citroen.c3"),       # single parked car, right
        ],
    ),
    MiniScenario(
        name          = "town02_double_stagger",
        base_scenario = "traffic_low",
        route_length  = 200,
        spawn_index   = 57,
        obstacles     = [
            (20.0,  1.5, "vehicle.audi.a2"),
            (42.0, -1.5, "vehicle.tesla.model3"),
            (0, 0, "vehicle.nissan.micra", 0, 174.4, 237.1, False, 0.2),  # absolute coords, z=0.2
            (0, 0, "vehicle.tesla.model3", 0, 181.0, 308.0, False, 0.4),  # absolute coords, z=0.4
        ],
    ),
    MiniScenario(
        name          = "town02_slow_npc",
        base_scenario = "traffic_high",
        route_length  = 200,
        spawn_index   = 57,          # matches CarlaEnv spawnpoint for "traffic_high"
        obstacles     = [
            (22.0,  1.5, "vehicle.nissan.micra"),     # parked car forces lane change
            (0, 0, "vehicle.tesla.model3",  0, 144.7, 236.6, False, 0.3),  # absolute coords
            (0, 0, "vehicle.audi.a2",       0,  92.0, 306.0, False, 0.3),  # absolute coords
        ],
        npc_autopilot = [
            (55.0,  0.0, "vehicle.mini.cooperst"),    # slow-moving vehicle ahead on autopilot
        ],
    ),


    # ── new: Town04 variants ───────────────────────────────────────────────────
    MiniScenario(
        name          = "town04_crossing_block",
        base_scenario = "crossing",
        route_length  = 160,
        spawn_index   = 166,         # matches CarlaEnv spawnpoint for "crossing"
        obstacles     = [
            (25.0,  1.5, "vehicle.nissan.micra",  0, None, None, False),  # spawn only, no detour
            (50.0, -1.5, "vehicle.audi.a2",       0, None, None, False),  # spawn only, no detour
        ],
        npc_autopilot = [
            (70.0,  0.0, "vehicle.mini.cooperst"),    # slow autopilot vehicle to overtake
        ],
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # 30 ADDITIONAL SCENARIOS
    # Crash-NPC scenarios (15): a vehicle spawns ahead, immediately veers hard,
    # and hits a wall/kerb — driver must navigate the wreck.
    # Regular scenarios (15): varied static + slow-NPC configurations.
    #
    # spawn_index values match CarlaEnv's hardcoded spawnpoint per scenario:
    #   intersection  → 8    traffic_low/high → 57
    #   tunnel        → 78   roundabout       → 0
    #   highway       → 9    crossing         → 166
    # ══════════════════════════════════════════════════════════════════════════

    # ── crash-NPC 1–4 : Town01 / intersection ────────────────────────────────
    MiniScenario(
        name          = "town01_crash_veer_right",
        base_scenario = "intersection",
        route_length  = 200,
        spawn_index   = 8,
        npc_crash     = [
            (48.0,  0.0, "vehicle.tesla.model3",   0.9),   # spawns in lane, veers right off road
        ],
    ),
    MiniScenario(
        name          = "town01_crash_veer_left",
        base_scenario = "intersection",
        route_length  = 200,
        spawn_index   = 8,
        npc_crash     = [
            (28.0,  0.0, "vehicle.audi.a2",        -0.9),  # spawns in lane, veers left into barrier
        ],
        detour_peak_m = -0.5,
    ),
    MiniScenario(
        name          = "town01_crash_with_parked",
        base_scenario = "intersection",
        route_length  = 220,
        spawn_index   = 8,
        obstacles     = [
            (55.0,  1.5, "vehicle.nissan.micra"),           # parked car further ahead
        ],
        npc_crash     = [
            (28.0,  6.0, "vehicle.mini.cooperst",  0.85),  # crash blocks near lane; parked blocks far
        ],
    ),
    MiniScenario(
        name          = "town01_crash_two_veers",
        base_scenario = "intersection",
        route_length  = 260,
        spawn_index   = 8,
        npc_crash     = [
            (35.0,  0.0, "vehicle.citroen.c3",     0.9),   # first crash, veer right
            (60.0,  0.0, "vehicle.volkswagen.t2",  -0.9),  # second crash, veer left
        ],
    ),

    # ── crash-NPC 5–7 : Town02 / traffic_low & traffic_high ──────────────────
    MiniScenario(
        name          = "town02_crash_veer_right",
        base_scenario = "traffic_low",
        route_length  = 180,
        spawn_index   = 57,
        npc_crash     = [
            (33.0,  0.0, "vehicle.audi.a2",        0.9),
        ],
    ),
    MiniScenario(
        name          = "town02_crash_veer_left",
        base_scenario = "traffic_low",
        route_length  = 180,
        spawn_index   = 57,
        npc_crash     = [
            (0, 0, "vehicle.nissan.micra",  -0.9, 144.7, 236.6),
        ],
    ),
    MiniScenario(
        name          = "town02_crash_plus_slow_npc",
        base_scenario = "traffic_high",
        route_length  = 220,
        spawn_index   = 57,
        npc_autopilot = [
            (60.0,  0.0, "vehicle.mini.cooperst"),          # slow vehicle further ahead
        ],
        npc_crash     = [
            (28.0,  0.0, "vehicle.citroen.c3",     0.85),  # crash blocks near lane
        ],
    ),


    # ── crash-NPC 11 : Town04 / crossing ─────────────────────────────────────
    MiniScenario(
        name          = "town04_crossing_crash",
        base_scenario = "crossing",
        route_length  = 160,
        spawn_index   = 166,
        npc_crash     = [
            (28.0,  0.0, "vehicle.nissan.micra",   0.9),
        ],
    ),

    # ── crash-NPC 14–15 : multi-car pile-ups ─────────────────────────────────
    MiniScenario(
        name          = "town01_crash_pileup",
        base_scenario = "intersection",
        route_length  = 260,
        spawn_index   = 8,
        npc_crash     = [
            (22.0,  0.0, "vehicle.citroen.c3",    0.85),   # first car clips right kerb
            (35.0,  0.0, "vehicle.audi.a2",       -0.85),  # second car spins left
            (48.0,  0.0, "vehicle.nissan.micra",   0.9),   # third car goes wide right
        ],
    ),

    # ── regular 1–4 : Town01 / intersection ──────────────────────────────────
    MiniScenario(
        name          = "town01_four_obstacles",
        base_scenario = "intersection",
        route_length  = 260,
        spawn_index   = 8,
        obstacles     = [
            (20.0,  1.5, "vehicle.audi.a2"),
            (20.0, -1.5, "vehicle.nissan.micra"),
            (50.0,  1.5, "vehicle.citroen.c3"),
            (50.0, -1.5, "vehicle.mini.cooperst"),
        ],
    ),
    MiniScenario(
        name          = "town01_slow_traffic",
        base_scenario = "intersection",
        route_length  = 240,
        spawn_index   = 8,
        npc_autopilot = [
            (35.0,  0.0, "vehicle.nissan.micra"),
            (55.0,  0.0, "vehicle.citroen.c3"),
        ],
    ),
    MiniScenario(
        name          = "town01_parked_plus_slow",
        base_scenario = "intersection",
        route_length  = 240,
        spawn_index   = 8,
        obstacles     = [
            (22.0,  1.5, "vehicle.audi.a2"),
            (45.0, -1.5, "vehicle.tesla.model3"),
        ],
        npc_autopilot = [
            (65.0,  0.0, "vehicle.nissan.micra"),
        ],
    ),
    MiniScenario(
        name          = "town01_close_pair",
        base_scenario = "intersection",
        route_length  = 180,
        spawn_index   = 8,
        obstacles     = [
            (22.0,  1.4, "vehicle.mini.cooperst"),
            (28.0, -1.4, "vehicle.citroen.c3"),    # close stagger — narrow corridor
        ],
    ),

    # ── regular 5–8 : Town02 / traffic_low & traffic_high ────────────────────
    MiniScenario(
        name          = "town02_three_parked",
        base_scenario = "traffic_low",
        route_length  = 240,
        spawn_index   = 57,
        obstacles     = [
            (20.0,  1.5, "vehicle.audi.a2"),
            (40.0, -1.5, "vehicle.volkswagen.t2"),
            (60.0,  1.5, "vehicle.nissan.micra"),
        ],
    ),
    MiniScenario(
        name          = "town02_slow_convoy",
        base_scenario = "traffic_low",
        route_length  = 200,
        spawn_index   = 57,
        npc_autopilot = [
            (40.0,  0.0, "vehicle.mini.cooperst"),
            (55.0,  0.0, "vehicle.citroen.c3"),
        ],
    ),
    MiniScenario(
        name          = "town02_four_parked",
        base_scenario = "traffic_high",
        route_length  = 260,
        spawn_index   = 57,
        obstacles     = [
            (18.0,  1.5, "vehicle.audi.a2"),
            (35.0, -1.5, "vehicle.tesla.model3"),
            (52.0,  1.5, "vehicle.nissan.micra"),
            (69.0, -1.5, "vehicle.mini.cooperst"),
        ],
    ),
    MiniScenario(
        name          = "town02_parked_plus_slow",
        base_scenario = "traffic_high",
        route_length  = 220,
        spawn_index   = 57,
        obstacles     = [
            (22.0,  1.5, "vehicle.citroen.c3"),
        ],
        npc_autopilot = [
            (50.0,  0.0, "vehicle.audi.a2"),
            (65.0,  0.0, "vehicle.volkswagen.t2"),
        ],
    ),


    # ── regular 12–13 : Town04 / crossing ────────────────────────────────────
    MiniScenario(
        name          = "town04_crossing_triple",
        base_scenario = "crossing",
        route_length  = 160,
        spawn_index   = 166,
        obstacles     = [
            (20.0,  1.5, "vehicle.nissan.micra"),
            (40.0, -1.5, "vehicle.audi.a2"),
            (60.0,  1.5, "vehicle.volkswagen.t2"),
        ],
    ),
    MiniScenario(
        name          = "town04_crossing_parked_plus_slow",
        base_scenario = "crossing",
        route_length  = 160,
        spawn_index   = 166,
        obstacles     = [
            (22.0,  1.5, "vehicle.citroen.c3"),
            (45.0, -1.5, "vehicle.tesla.model3"),
        ],
        npc_autopilot = [
            (65.0,  0.0, "vehicle.mini.cooperst"),
        ],
    ),
]

# Snapshot of every scenario before NPC removal — preserved for reference /
# future restore.  OLD_SCENARIOS[i] mirrors MINI_SCENARIOS[i] exactly.
OLD_SCENARIOS: List[MiniScenario] = copy.deepcopy(MINI_SCENARIOS)

# Strip all actors and route offsets from every active scenario.
MINI_SCENARIOS = [
    dc_replace(ms, obstacles=[], npc_autopilot=[], npc_crash=[], detour_peak_m=0)
    for ms in MINI_SCENARIOS
]


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
            pass  # ffb_init() disabled — SDK init overrides G HUB spring settings
        else:
            pass

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
                if ev.key == pygame.K_q:
                    self.quit_request = True
            # G920 wheel: button 8 (left paddle shifter) toggles reverse
            if ev.type == pygame.JOYBUTTONDOWN and ev.button == 8:
                self._reverse = not self._reverse

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
    client = carla.Client(host, port)
    client.set_timeout(30.0)
    world  = client.get_world()
    s = world.get_settings()
    s.synchronous_mode    = True
    s.fixed_delta_seconds = _DT
    s.no_rendering_mode   = not render
    world.apply_settings(s)
    return client, world


def _load_map(client: object, world: object, name: str) -> object:
    current = world.get_map().name.split("/")[-1]  # type: ignore[union-attr]
    if current != name:
        world = client.load_world(name)  # type: ignore[union-attr]
        s = world.get_settings()  # type: ignore[union-attr]
        s.synchronous_mode    = True
        s.fixed_delta_seconds = _DT
        s.no_rendering_mode   = False
        world.apply_settings(s)  # type: ignore[union-attr]
    return world


def _disconnect(world: object) -> None:
    s = world.get_settings()  # type: ignore[union-attr]
    s.synchronous_mode = False
    world.apply_settings(s)  # type: ignore[union-attr]


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
            return v
    raise RuntimeError("Could not spawn hero at any spawn point — all occupied.")


def _spawn_scenario_obstacles(
    world: object,
    hero:  object,
    scenario: MiniScenario,
    spawn_index: int = 8,
) -> List[object]:
    """Spawn static obstacle vehicles relative to spawn point 8.

    Uses the map spawn point directly (not the hero's current transform) so
    the obstacle position is identical whether called from the AV phase or the
    human phase, and is not affected by any physics drift of the hero vehicle.
    The hero is only used to get the correct ground-level z.
    Returns (spawned_actors, no_detour_ids) where no_detour_ids is a set of
    actor IDs that should be excluded from the route-detour calculation.
    Tuple element [6] == False opts an obstacle out of detour (default: included).
    """
    pts   = world.get_map().get_spawn_points()  # type: ignore[union-attr]
    tf    = pts[spawn_index % len(pts)]          # fixed reference — never drifts
    yaw_r = math.radians(tf.rotation.yaw)
    # Unit vectors in CARLA's XY plane (Z-up, yaw measured from +X axis)
    fwd_x, fwd_y =  math.cos(yaw_r),  math.sin(yaw_r)
    rgt_x, rgt_y =  math.sin(yaw_r), -math.cos(yaw_r)   # 90° clockwise = right

    bp_lib       = world.get_blueprint_library()  # type: ignore[union-attr]
    spawned:       List[object] = []
    no_detour_ids: set          = set()

    for entry in scenario.obstacles:
        fwd_m, right_m, bp_id = entry[0], entry[1], entry[2]
        yaw_offset  = float(entry[3]) if len(entry) > 3 else 0.0
        include_det = bool(entry[6]) if len(entry) > 6 else True
        try:
            bp = bp_lib.find(bp_id)
        except Exception:
            print(f"  [spawn] WARNING: blueprint '{bp_id}' not found — skipping.")
            continue

        if len(entry) >= 6 and entry[4] is not None and entry[5] is not None:
            _ox, _oy = float(entry[4]), float(entry[5])
        else:
            _ox = tf.location.x + fwd_m * fwd_x + right_m * rgt_x
            _oy = tf.location.y + fwd_m * fwd_y + right_m * rgt_y
        if len(entry) >= 8 and entry[7] is not None:
            _oz = float(entry[7])  # explicit z override
        else:
            # Project onto the road surface so z is always on the drivable lane
            # regardless of map terrain.  Fallback to hero z if no waypoint found.
            _wpt = world.get_map().get_waypoint(  # type: ignore[union-attr]
                carla.Location(x=_ox, y=_oy, z=hero.get_location().z),  # type: ignore[union-attr]
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            _oz = (_wpt.transform.location.z if _wpt is not None
                   else hero.get_location().z - 0.25)  # type: ignore[union-attr]
        loc = carla.Location(x=_ox, y=_oy, z=_oz + 0.05)  # tiny lift prevents underground spawn; physics disabled so no settling
        obs_tf = carla.Transform(loc, carla.Rotation(yaw=tf.rotation.yaw + yaw_offset))
        actor  = world.try_spawn_actor(bp, obs_tf)  # type: ignore[union-attr]
        if actor is not None:
            actor.set_simulate_physics(False)   # freeze immediately — no drift or fall
            spawned.append(actor)
            if not include_det:
                no_detour_ids.add(actor.id)
        else:
            print(f"  [spawn] WARNING: try_spawn_actor failed at {loc} for '{bp_id}' "
                  f"(wpt={'found' if _wpt else 'none'}, z={_oz:.2f})")

    if spawned:
        world.tick()  # register all actors before the game loop starts  # type: ignore[union-attr]
    return spawned, no_detour_ids


def _spawn_npc_autopilot(
    world:       object,
    hero:        object,
    scenario:    MiniScenario,
    spawn_index: int = 8,
) -> List[object]:
    """Spawn slow-moving NPC vehicles with autopilot enabled.

    Placed using the same forward/right offset system as static obstacles.
    These vehicles drive under CARLA's built-in autopilot — their speed is
    capped to 20 km/h via the traffic manager so the AV can easily overtake.
    Returns the list of spawned actors so the caller can destroy them later.
    """
    if not scenario.npc_autopilot:
        return []

    pts   = world.get_map().get_spawn_points()  # type: ignore[union-attr]
    tf    = pts[spawn_index % len(pts)]
    yaw_r = math.radians(tf.rotation.yaw)
    fwd_x, fwd_y =  math.cos(yaw_r),  math.sin(yaw_r)
    rgt_x, rgt_y =  math.sin(yaw_r), -math.cos(yaw_r)

    bp_lib  = world.get_blueprint_library()  # type: ignore[union-attr]
    spawned: List[object] = []

    try:
        tm = world.get_trafficmanager()  # type: ignore[union-attr]
        tm_port = tm.get_port()
    except Exception:
        tm      = None
        tm_port = 8000

    for fwd_m, right_m, bp_id in scenario.npc_autopilot:
        try:
            bp = bp_lib.find(bp_id)
        except Exception:
            print(f"  [spawn] WARNING: blueprint '{bp_id}' not found — skipping.")
            continue

        _nx = tf.location.x + fwd_m * fwd_x + right_m * rgt_x
        _ny = tf.location.y + fwd_m * fwd_y + right_m * rgt_y
        _wpt = world.get_map().get_waypoint(  # type: ignore[union-attr]
            carla.Location(x=_nx, y=_ny, z=hero.get_location().z),  # type: ignore[union-attr]
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        _nz = (_wpt.transform.location.z if _wpt is not None
               else hero.get_location().z - 0.25)  # type: ignore[union-attr]
        loc    = carla.Location(x=_nx, y=_ny, z=_nz + 0.3)
        npc_tf = carla.Transform(loc, carla.Rotation(yaw=tf.rotation.yaw))
        actor  = world.try_spawn_actor(bp, npc_tf)  # type: ignore[union-attr]
        if actor is not None:
            actor.set_autopilot(True, tm_port)
            if tm is not None:
                tm.vehicle_percentage_speed_difference(actor, 70)  # 30% of speed limit ≈ slow
                tm.auto_lane_change(actor, False)                  # stay in lane
            spawned.append(actor)
        else:
            print(f"  [spawn] WARNING: try_spawn_actor failed at {loc} for '{bp_id}' "
                  f"(wpt={'found' if _wpt else 'none'}, z={_nz:.2f})")

    if spawned:
        world.tick()  # type: ignore[union-attr]
    return spawned


def _spawn_npc_crash(
    world:       object,
    hero:        object,
    scenario:    MiniScenario,
    spawn_index: int = 8,
) -> List[object]:
    """Spawn vehicles that immediately veer off-road and crash.

    Each vehicle is spawned with physics enabled, then a single
    VehicleControl(throttle=0.7, steer=<steer>) is applied.  CARLA keeps that
    control active on every subsequent tick until explicitly changed, so the
    vehicle accelerates while turning hard, leaves the road, and hits a wall,
    kerb, or barrier — creating a wreck the AV/human must steer around.

    The veer starts the moment the game loop's first world.tick() fires, so
    by the time either driver reaches the crash site (~2–5 s at normal speed)
    the vehicle has already left the road.
    """
    if not scenario.npc_crash:
        return []

    pts   = world.get_map().get_spawn_points()  # type: ignore[union-attr]
    tf    = pts[spawn_index % len(pts)]
    yaw_r = math.radians(tf.rotation.yaw)
    fwd_x, fwd_y =  math.cos(yaw_r),  math.sin(yaw_r)
    rgt_x, rgt_y =  math.sin(yaw_r), -math.cos(yaw_r)

    bp_lib  = world.get_blueprint_library()  # type: ignore[union-attr]
    spawned: List[object] = []

    for entry in scenario.npc_crash:
        fwd_m, right_m, bp_id, steer = entry[0], entry[1], entry[2], entry[3]
        try:
            bp = bp_lib.find(bp_id)
        except Exception:
            print(f"  [spawn] WARNING: blueprint '{bp_id}' not found — skipping.")
            continue

        if len(entry) >= 6 and entry[4] is not None and entry[5] is not None:
            _cx, _cy = float(entry[4]), float(entry[5])
        else:
            _cx = tf.location.x + fwd_m * fwd_x + right_m * rgt_x
            _cy = tf.location.y + fwd_m * fwd_y + right_m * rgt_y
        _wpt = world.get_map().get_waypoint(  # type: ignore[union-attr]
            carla.Location(x=_cx, y=_cy, z=hero.get_location().z),  # type: ignore[union-attr]
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        _cz = (_wpt.transform.location.z if _wpt is not None
               else hero.get_location().z - 0.25)  # type: ignore[union-attr]
        loc    = carla.Location(x=_cx, y=_cy, z=_cz + 0.3)
        npc_tf = carla.Transform(loc, carla.Rotation(yaw=tf.rotation.yaw))
        actor  = world.try_spawn_actor(bp, npc_tf)  # type: ignore[union-attr]
        if actor is not None:
            # Physics ON (do not freeze) — the vehicle must drive and crash.
            # Autopilot OFF — we control it manually so it ignores the road.
            actor.set_autopilot(False)
            actor.apply_control(carla.VehicleControl(  # type: ignore[union-attr]
                throttle=0.7,
                steer=float(np.clip(steer, -1.0, 1.0)),
                brake=0.0,
                hand_brake=False,
                reverse=False,
            ))
            spawned.append(actor)
        else:
            print(f"  [spawn] WARNING: try_spawn_actor failed at {loc} for '{bp_id}' "
                  f"(wpt={'found' if _wpt else 'none'}, z={_cz:.2f})")

    if spawned:
        world.tick()  # type: ignore[union-attr]
    return spawned


# ── DEBUG ALERT DRAWING (CARLA UE4 WINDOW) ───────────────────────────────────

def _carla_rgb(rgb: Tuple[int, int, int]) -> object:
    return carla.Color(r=rgb[0], g=rgb[1], b=rgb[2])


def _dist_color(dist_m: float, colorblind: bool = False) -> Tuple[int, int, int]:
    t = float(np.clip(dist_m / _COLOR_DIST_M, 0.0, 1.0))
    if colorblind:
        return (int(t * 230), int(114 - t * 114), int(178 - t * 178))
    return (int(t * 220), int((1 - t) * 200), 0)


def _speed_color(human_speed: float, av_speed: float, colorblind: bool = False) -> Tuple[int, int, int]:
    """Color by human speed relative to AV speed.

    t=0 (green)    → human ≥10 m/s slower than AV
    t=0.5 (yellow) → same speed as AV
    t=1 (red)      → human ≥10 m/s faster than AV
    """
    t = float(np.clip((human_speed - av_speed) / 10.0 + 0.5, 0.0, 1.0))
    if colorblind:
        return (int(t * 230), int(114 - t * 114), int(178 - t * 178))
    return (int(t * 220), int((1.0 - t) * 200), 0)


# Module-level sound cooldown (reset at episode start)
_last_sound_t: float = -999.0


def _draw_av_dot(world: object, traj: AVTrajectory, sim_time: float) -> None:
    """Test mode: draw a red dot at the AV's current position plus fading lookahead dots."""
    pos = traj.get_position_at(sim_time)
    if pos is None:
        return
    world.debug.draw_point(  # type: ignore[union-attr]
        carla.Location(x=pos[0], y=pos[1], z=pos[2] + 0.5),
        size=0.3,
        color=carla.Color(r=255, g=40, b=40),
        life_time=_DRAW_LT,
    )
    for look, sz in ((1.0, 0.2), (2.0, 0.15), (3.0, 0.1)):
        future = traj.get_position_at(sim_time + look)
        if future is not None:
            world.debug.draw_point(  # type: ignore[union-attr]
                carla.Location(x=future[0], y=future[1], z=future[2] + 0.5),
                size=sz,
                color=carla.Color(r=255, g=160, b=40),
                life_time=_DRAW_LT,
            )



def _draw_route(world, alert, hero, traj, sim_time):
    """Draw the AV trajectory as debug lines coloured by human speed vs AV speed."""
    positions = traj.all_positions()
    if len(positions) < 2:
        return

    vel    = hero.get_velocity()
    hspeed = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
    color  = _speed_color(hspeed, traj.get_speed_at(sim_time), bool(alert.color))
    c      = carla.Color(r=color[0], g=color[1], b=color[2])

    # Sample at most ~60 segments so debug draw calls stay cheap
    n    = len(positions)
    step = max(1, n // 60)
    for i in range(0, n - step, step):
        a, b = positions[i], positions[i + step]
        world.debug.draw_line(
            carla.Location(x=a[0], y=a[1], z=a[2] + 0.3),
            carla.Location(x=b[0], y=b[1], z=b[2] + 0.3),
            thickness=0.05,
            color=c,
            life_time=_DRAW_LT,
        )


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
        _draw_route(world, alert, hero, traj, sim_time)
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

    # Distance to AV target (still used for vibration threshold)
    dist = math.sqrt((target[0] - hloc.x)**2 + (target[1] - hloc.y)**2)

    # Speed-based color: faster than AV → red, slower → green
    _hvel      = hero.get_velocity()
    _hspeed    = math.sqrt(_hvel.x ** 2 + _hvel.y ** 2 + _hvel.z ** 2)
    _av_speed  = traj.get_speed_at(sim_time)
    color      = _speed_color(_hspeed, _av_speed, bool(alert.color))

    # Windshield anchor: 1.3 m ahead, centred in the driver's view
    # (no lateral offset — matches the centre of the score GUI on screen).
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



def _tick_sound(alert, hero, traj, sim_time):
    global _last_sound_t
    lat_thresh = float(alert.gui_params[0]) * 4.0     # [0,1] → [0,4] m  (was 10 — reduced for earlier trigger)
    cooldown   = float(alert.gui_params[1]) * 10.0   # [0,1] → [0,10] s
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


# ── SCORE GUI ─────────────────────────────────────────────────────────────────

def _show_score_pygame(
    scenario_name:  str,
    iteration:      int,
    driving_score:  float,
    episode_loss:   float,
    duration:       float = 10.0,
) -> None:
    """Large pygame score screen shown after each scenario (human run + training).

    Shows the human driving score and the alert loss for the iteration just
    completed.  Press SPACE / ENTER to skip early.

    driving_score : mean of the 4 style-reward components (speed, efficiency,
                    aggressiveness, comfort) for the human across all ticks — higher = better
    episode_loss  : mean euclidean distance (m) between human and AV positions — lower = better
    """
    import pygame as pg

    def _score_col(v: float, invert: bool = False) -> tuple:
        """Green = good.  invert=True for loss (lower is better)."""
        if invert:
            if v <= 0.05: return (60, 230, 60)
            if v <= 0.20: return (230, 210, 50)
            return (220, 70, 70)
        else:
            if v >= 0.7: return (60, 230, 60)
            if v >= 0.4: return (230, 210, 50)
            return (220, 70, 70)

    if not pg.get_init():
        pg.init()
    if not pg.font.get_init():
        pg.font.init()

    W, H = 1280, 860
    _info = pg.display.Info()
    _cx = max(0, (_info.current_w - W) // 2)
    _cy = max(0, (_info.current_h - H) // 2)
    os.environ["SDL_VIDEO_WINDOW_POS"] = f"{_cx},{_cy}"
    screen = pg.display.set_mode((W, H), pg.NOFRAME)
    pg.display.set_caption("Score")

    font_title   = pg.font.SysFont("Arial", 56, bold=True)
    font_name    = pg.font.SysFont("Arial", 38)
    font_huge    = pg.font.SysFont("Arial", 150, bold=True)
    font_medium  = pg.font.SysFont("Arial", 90, bold=True)
    font_label   = pg.font.SysFont("Arial", 36)
    font_explain = pg.font.SysFont("Arial", 28)
    font_footer  = pg.font.SysFont("Arial", 30)

    BG     = (12, 14, 26)
    PANEL  = (22, 26, 48)
    BORDER = (60, 80, 140)
    DIM    = (100, 110, 140)
    WHITE  = (210, 215, 255)

    clock = pg.time.Clock()
    start = time.time()

    while True:
        remaining = max(0.0, duration - (time.time() - start))
        for ev in pg.event.get():
            if ev.type == pg.QUIT:
                pg.display.quit()
                return
            if ev.type == pg.KEYDOWN and ev.key in (pg.K_RETURN, pg.K_SPACE, pg.K_ESCAPE):
                pg.display.quit()
                return

        if remaining <= 0.0:
            break

        screen.fill(BG)
        pad = 40
        pg.draw.rect(screen, PANEL,  (pad, pad, W - 2*pad, H - 2*pad), border_radius=18)
        pg.draw.rect(screen, BORDER, (pad, pad, W - 2*pad, H - 2*pad), 3, border_radius=18)

        cy = pad + 36

        # ── header ────────────────────────────────────────────────────────────
        t = font_title.render(f"SCENARIO {iteration} COMPLETE", True, WHITE)
        screen.blit(t, t.get_rect(centerx=W//2, y=cy)); cy += 62

        n = font_name.render(scenario_name.upper().replace("_", " "), True, DIM)
        screen.blit(n, n.get_rect(centerx=W//2, y=cy)); cy += 44

        pg.draw.line(screen, BORDER, (pad + 60, cy), (W - pad - 60, cy), 2); cy += 18

        # ── AV Driving Score ──────────────────────────────────────────────────
        ds_col = _score_col(driving_score, invert=False)
        ds_surf = font_huge.render(f"{driving_score:.3f}", True, ds_col)
        screen.blit(ds_surf, ds_surf.get_rect(centerx=W//2, y=cy)); cy += 158

        lbl1 = font_label.render("HUMAN  DRIVING  SCORE", True, DIM)
        screen.blit(lbl1, lbl1.get_rect(centerx=W//2, y=cy)); cy += 42

        exp1 = font_explain.render(
            "RL reward accumulated by the AV during the pre-run.  Higher = better route completion.",
            True, (140, 150, 180))
        screen.blit(exp1, exp1.get_rect(centerx=W//2, y=cy)); cy += 36

        pg.draw.line(screen, BORDER, (pad + 60, cy), (W - pad - 60, cy), 2); cy += 18

        # ── Alert Model Loss ──────────────────────────────────────────────────
        el_col = _score_col(episode_loss, invert=True)
        el_surf = font_medium.render(f"{episode_loss:.4f}", True, el_col)
        screen.blit(el_surf, el_surf.get_rect(centerx=W//2, y=cy)); cy += 100

        lbl2 = font_label.render("ALERT  MODEL  LOSS", True, DIM)
        screen.blit(lbl2, lbl2.get_rect(centerx=W//2, y=cy)); cy += 42

        exp2 = font_explain.render(
            "Training loss after learning from your driving.  Lower = alert model is improving.",
            True, (140, 150, 180))
        screen.blit(exp2, exp2.get_rect(centerx=W//2, y=cy)); cy += 36

        pg.draw.line(screen, BORDER, (pad + 60, cy), (W - pad - 60, cy), 2); cy += 14

        # ── footer ────────────────────────────────────────────────────────────
        f = font_footer.render(
            f"Next in {remaining:.0f}s  —  press SPACE or ENTER to continue",
            True, (80, 90, 120))
        screen.blit(f, f.get_rect(centerx=W//2, y=cy))

        pg.display.flip()
        clock.tick(30)

    pg.display.quit()


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


def _nearest_av_by_position(steps: List[AVStepData], x: float, y: float) -> AVStepData:
    """Return the AV step whose XY position is closest to (x, y)."""
    if not steps:
        return AVStepData(0, 0, 0, 0, 0, 0, 0, 0, 0, 1)
    best = steps[0]
    best_d2 = float('inf')
    for s in steps:
        d2 = (s.x - x) ** 2 + (s.y - y) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best = s
    return best


def _compute_human_driving_score(
    avg_dist_to_center: float,
    collision_cnt_env:  int,
    collision_cnt_car:  int,
    lane_invasion_cnt:  int,
    speeding_cnt:       int,
    timeout:            float,
    route_completion:   float,
) -> float:
    """Identical formula to carla_gym_env.py adjusted_driving_core.

    Factors (all ≤ 1, multiplicative penalties):
      0.98 ^ avg_dist_to_center  — lane-centre deviation (per-tick average metres)
      0.65 ^ collision_cnt_env   — collisions with static environment
      0.75 ^ collision_cnt_car   — collisions with other vehicles
      0.995^ lane_invasion_cnt   — lane-marking crossings
      0.95 ^ speeding_cnt        — ticks spent above speed limit
      0.75 ^ timeout             — episode ended by timeout (0 or 1)
      route_completion ^ 1.2     — fraction of AV waypoints reached
    """
    score = (
        0.98 ** avg_dist_to_center *
        0.65 ** collision_cnt_env  *
        0.75 ** collision_cnt_car  *
        0.995 ** lane_invasion_cnt *
        0.95 ** speeding_cnt       *
        0.75 ** timeout            *
        route_completion ** 1.2
    )
    return round(float(np.clip(score, 0.0, 1.0)), 3)


# ── TUTORIAL ──────────────────────────────────────────────────────────────────

# Tutorial is now fully action-gated (see run_tutorial).
# _TUTORIAL_DURATION is kept so --tutorial-duration CLI arg still parses.
_TUTORIAL_DURATION = 90  # seconds


def run_tutorial(_client: object, world: object, duration: int = _TUTORIAL_DURATION) -> None:
    """Interactive control tutorial shown before calibration.

    Each stage waits for the participant to actually perform the action before
    advancing — the tutorial does not time out mid-stage.  Reverse detection
    monitors the ControlReader.reverse flag toggling ON.

    Stages (in order):
      0  Welcome          — auto-advances after 6 s
      1  Gas / Throttle   — advance when throttle > 0.25
      2  Brake            — advance when brake    > 0.25
      3  Steer            — advance when |steer|  > 0.25
      4  Reverse (RSB)    — advance when reverse flag turns True
      done → return
    """
    print(f"\n{'='*60}")
    print(f"TUTORIAL  (max {duration}s)  — interactive control check")
    print(f"{'='*60}")

    # Stage definitions: (headline, detail_lines, completion_hint)
    _STAGES = [
        ("WELCOME — LEARN THE CONTROLS",
         ["You are about to drive a simulation.",
          "Follow each prompt and perform the action shown.",
          "The next stage unlocks once you complete the action.",
          "Take your time — the tutorial waits for you."],
         None),   # auto-advance
        ("RIGHT PEDAL  =  GAS / THROTTLE",
         ["Press the RIGHT pedal to accelerate forward.",
          "Release fully to coast.",
          ">>> Press the gas pedal now to continue <<<"],
         "throttle"),
        ("MIDDLE PEDAL  =  BRAKE",
         ["Press the MIDDLE pedal to slow down or stop.",
          "Hold it to come to a complete stop.",
          ">>> Press the brake pedal now to continue <<<"],
         "brake"),
        ("STEERING WHEEL  =  STEER",
         ["Turn the wheel left or right to steer.",
          "Small inputs = gentle changes.",
          ">>> Turn the wheel now to continue <<<"],
         "steer"),
        ("RSB BUTTON  =  REVERSE MODE",
         ["Press RSB (right shoulder button) to enter REVERSE.",
          "Press RSB again to return to FORWARD.",
          "In reverse: gas pedal drives you backward.",
          ">>> Press RSB now to continue <<<"],
         "reverse"),
    ]

    # Clear any leftover actors
    for _a in [a for a in world.get_actors()  # type: ignore[union-attr]
               if a.type_id.startswith("vehicle.") or a.type_id.startswith("walker.")]:
        _a.destroy()
    world.tick()  # type: ignore[union-attr]

    hero        = _spawn_hero(world)
    ctrl_reader = ControlReader()
    clock       = pygame.time.Clock()

    stage_idx        = 0
    stage_start      = time.time()
    stage_done       = False        # flashes "✓ Done!" briefly before advancing
    stage_done_t     = 0.0
    _DONE_FLASH_S    = 1.2          # seconds to show the tick before advancing
    _WELCOME_AUTO_S  = 6.0          # auto-advance welcome stage after this many seconds
    _prev_reverse    = False        # track reverse toggle

    try:
        while stage_idx < len(_STAGES):
            ctrl = ctrl_reader.read()
            if ctrl_reader.quit_request:
                break
            hero.apply_control(ctrl)  # type: ignore[union-attr]
            world.tick()              # type: ignore[union-attr]
            clock.tick(FPS)
            _spectator_follow(world, hero)

            now        = time.time()
            stage_age  = now - stage_start
            headline, detail_lines, action = _STAGES[stage_idx]

            # ── Check completion ──────────────────────────────────────────────
            if not stage_done:
                completed = False
                if action is None:
                    completed = stage_age >= _WELCOME_AUTO_S
                elif action == "throttle":
                    completed = float(ctrl.throttle) > 0.25
                elif action == "brake":
                    completed = float(ctrl.brake) > 0.25
                elif action == "steer":
                    completed = abs(float(ctrl.steer)) > 0.25
                elif action == "reverse":
                    # detect the moment reverse turns ON
                    _cur_rev = bool(ctrl_reader.reverse)
                    completed = _cur_rev and not _prev_reverse
                    _prev_reverse = _cur_rev

                if completed:
                    stage_done   = True
                    stage_done_t = now
            elif now - stage_done_t >= _DONE_FLASH_S:
                stage_idx   += 1
                stage_start  = now
                stage_done   = False
                _prev_reverse = bool(ctrl_reader.reverse)
                continue

            # ── Anchor text near windshield (same as arrow) ───────────────────
            _hloc  = hero.get_location()   # type: ignore[union-attr]
            _hvel  = hero.get_velocity()   # type: ignore[union-attr]
            _htf   = hero.get_transform()  # type: ignore[union-attr]
            _hyaw  = math.radians(_htf.rotation.yaw)
            # 0.20 m forward (closer = larger on screen); 3.5 m left of centre
            _lft_x = -math.sin(_hyaw)
            _lft_y =  math.cos(_hyaw)
            _px    = _hloc.x + _hvel.x * _DT + 0.20 * math.cos(_hyaw) + 3.5 * _lft_x
            _py    = _hloc.y + _hvel.y * _DT + 0.20 * math.sin(_hyaw) + 3.5 * _lft_y
            _wz    = _hloc.z + _hvel.z * _DT + 1.25

            step_lbl = f"Step {stage_idx}/{len(_STAGES)-1}"
            rev_tag  = "  [ REVERSE ]" if ctrl_reader.reverse else ""
            world.debug.draw_string(  # type: ignore[union-attr]
                location=carla.Location(x=_px, y=_py, z=_wz + 0.28),
                text=f"TUTORIAL  {step_lbl}{rev_tag}",
                color=carla.Color(r=180, g=180, b=180),
                life_time=_DRAW_LT,
            )

            if stage_done:
                world.debug.draw_string(  # type: ignore[union-attr]
                    location=carla.Location(x=_px, y=_py, z=_wz + 0.10),
                    text=">>> DONE!  Moving to next step... <<<",
                    color=carla.Color(r=60, g=255, b=60),
                    life_time=_DRAW_LT,
                )
            else:
                world.debug.draw_string(  # type: ignore[union-attr]
                    location=carla.Location(x=_px, y=_py, z=_wz + 0.10),
                    text=f">>> {headline} <<<",
                    color=carla.Color(r=255, g=240, b=0),
                    life_time=_DRAW_LT,
                )
                for _li, _dl in enumerate(detail_lines):
                    world.debug.draw_string(  # type: ignore[union-attr]
                        location=carla.Location(x=_px, y=_py, z=_wz - 0.10 - _li * 0.17),
                        text=_dl,
                        color=carla.Color(r=230, g=230, b=230),
                        life_time=_DRAW_LT,
                    )

    finally:
        ctrl_reader.close()
        hero.destroy()  # type: ignore[union-attr]

    print(f"[tutorial] Done — all {len(_STAGES)} stages completed.")


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

    # Clear any NPC vehicles/walkers so calibration is an empty road.
    _cal_clear = [
        a for a in world.get_actors()
        if a.type_id.startswith("vehicle.") or a.type_id.startswith("walker.")
    ]
    for _a in _cal_clear:
        _a.destroy()
    if _cal_clear:
        world.tick()

    hero      = _spawn_hero(world)
    regressor = HumanStyleRegressor()
    regressor.attach_collision_sensor(world, hero)

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

            # Draw HUD using predicted position to match spectator camera
            loc  = hero.get_location()
            vel  = hero.get_velocity()
            htf  = hero.get_transform()
            _yaw = math.radians(htf.rotation.yaw)
            _px  = loc.x + vel.x * _DT
            _py  = loc.y + vel.y * _DT
            _pz  = loc.z + vel.z * _DT
            ws_x = _px + 0.8 * math.cos(_yaw)
            ws_y = _py + 0.8 * math.sin(_yaw)
            rev_tag = "  [REVERSE]" if ctrl_reader.reverse else ""
            world.debug.draw_string(
                location=carla.Location(x=ws_x, y=ws_y, z=_pz + 1.3),
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

    profile = regressor.get_style_profile()
    print(f"\n[calibration] DONE → "
          + "  ".join(f"{l}={v:.3f}" for l, v in zip(STYLE_LABELS, profile)))
    return profile


# ── ROUTE OBSTACLE DETOUR ─────────────────────────────────────────────────────

def _route_around_obstacles(
    route:      list,
    obs_actors: list,    # spawned obstacle actors (carla.Actor); positions read at call-time
    peak_m:     float = _DETOUR_PEAK_M,
) -> list:
    """Shift waypoints around every spawned obstacle.

    Hard zone  (d ≤ _WP_CLEAR_RADIUS)
        Road-snap the shifted position to the nearest drivable lane.  When
        ``peak_m`` is large enough to cross a lane boundary (≥ half a lane
        width, roughly 1.75 m) this returns a real ``carla.Waypoint`` in the
        adjacent lane — exactly what the AV model was trained to follow.

    Transition zone  (_WP_CLEAR_RADIUS < d ≤ transition radius)
        Uses ``_OffsetWP`` (a thin ``carla.Waypoint``-compatible wrapper) so
        the ramp position is not snapped back to the original lane centre.

    Connecting nodes
        Three linearly-interpolated ``_OffsetWP`` nodes are inserted between
        each adjacent (unshifted, shifted) pair to smooth the entry/exit arc.

    Direction convention
        ``rgt_x, rgt_y = fwd_y, -fwd_x`` (CARLA yaw convention).
        Positive ``peak_m`` → obstacle on right → detour left (−rgt direction).
        Negative ``peak_m`` flips to detour right.
    """
    if not obs_actors:
        return route

    _transition_r = _WP_CLEAR_RADIUS * _WP_TRANSITION_MULT

    # Read actual actor world-positions at call-time.
    obs_locs: List[Tuple[float, float]] = []
    for _a in obs_actors:
        try:
            _al = _a.get_location()
            obs_locs.append((_al.x, _al.y))
        except Exception:
            pass
    if not obs_locs:
        return route

    # _OffsetWP: drop-in for carla.Waypoint that holds a raw shifted position.
    # Used for transition-zone waypoints that sit between two lane centres.
    class _OffsetWP:
        __slots__ = ("transform", "is_junction")
        def __init__(self, orig, nx: float, ny: float, nz: float) -> None:
            self.transform  = carla.Transform(  # type: ignore[union-attr]
                carla.Location(x=nx, y=ny, z=nz),  # type: ignore[union-attr]
                orig.transform.rotation,
            )
            self.is_junction = bool(getattr(orig, "is_junction", False))

    def _wp_xyz(wp) -> Tuple[float, float, float]:
        try:
            return (float(wp.transform.location.x),
                    float(wp.transform.location.y),
                    float(wp.transform.location.z))
        except Exception:
            return 0.0, 0.0, 0.0

    route_pts = [_wp_xyz(wp) for wp in route]
    n = len(route)

    # ── Pass 1: compute offset scalar and hard-zone flag per waypoint ─────────
    offsets:      List[float] = []
    rgt_vecs:     List[Tuple[float, float]] = []
    in_hard_zone: List[bool]  = []

    for i in range(n):
        wx, wy, _ = route_pts[i]
        j_lo = max(0, i - 2)
        j_hi = min(n - 1, i + 2)
        dx = route_pts[j_hi][0] - route_pts[j_lo][0]
        dy = route_pts[j_hi][1] - route_pts[j_lo][1]
        mag = math.sqrt(dx * dx + dy * dy)
        if mag < 1e-6:
            offsets.append(0.0)
            rgt_vecs.append((0.0, 0.0))
            in_hard_zone.append(False)
            continue
        fwd_x = dx / mag
        fwd_y = dy / mag
        rgt_x =  fwd_y
        rgt_y = -fwd_x

        total_lat = 0.0
        hard = False
        for ox, oy in obs_locs:
            d = math.sqrt((wx - ox) ** 2 + (wy - oy) ** 2)
            if d >= _transition_r:
                continue
            rgt_proj  = (ox - wx) * rgt_x + (oy - wy) * rgt_y
            direction = -1.0 if rgt_proj >= 0.0 else 1.0
            if d <= _WP_CLEAR_RADIUS:
                hard = True
                contribution = direction * peak_m
            else:
                t = (d - _WP_CLEAR_RADIUS) / (_transition_r - _WP_CLEAR_RADIUS)
                contribution = direction * peak_m * 0.5 * (1.0 + math.cos(t * math.pi))
            total_lat += contribution

        # Minimum clearance enforcement (hard zone only)
        for ox, oy in obs_locs:
            d = math.sqrt((wx - ox) ** 2 + (wy - oy) ** 2)
            if d >= _WP_CLEAR_RADIUS:
                continue
            sx = wx + total_lat * rgt_x
            sy = wy + total_lat * rgt_y
            lat_dist = (sx - ox) * rgt_x + (sy - oy) * rgt_y
            if abs(lat_dist) < _HALF_LANE_WIDTH:
                deficit = _HALF_LANE_WIDTH - abs(lat_dist)
                sign    = 1.0 if lat_dist >= 0 else -1.0
                total_lat += sign * deficit

        offsets.append(total_lat)
        rgt_vecs.append((rgt_x, rgt_y))
        in_hard_zone.append(hard)

    # ── Pass 2: build shifted waypoints using _OffsetWP throughout ───────────
    # Road-snap is NOT used: get_waypoint(project_to_road=True) always returns
    # the nearest lane *centre*, so any offset smaller than half a lane width
    # snaps straight back — and even larger offsets snap to the wrong centre.
    # _OffsetWP preserves the exact shifted coordinates so both hard-zone and
    # transition-zone dots move together and stay continuous.
    shifted_route: list = []
    for i, wp in enumerate(route):
        lat = offsets[i]
        if abs(lat) < 1e-4:
            shifted_route.append(wp)
            continue
        wx, wy, wz = route_pts[i]
        rx, ry     = rgt_vecs[i]
        shifted_route.append(_OffsetWP(wp, wx + lat * rx, wy + lat * ry, wz))

    # ── Pass 3: insert connecting waypoints wherever the shifted route has gaps ──
    # Check spatial distance between consecutive shifted waypoints (not offset
    # delta): in curved/junction sections, a constant lateral offset applied in
    # a rotating right-direction can create large XY jumps that exceed the AV's
    # 6 m max_route_deviation and terminate the episode.
    _MAX_WP_GAP = 1.5   # max allowed metres between consecutive waypoints
    final_route: list = []
    i = 0
    while i < len(shifted_route):
        final_route.append(shifted_route[i])
        if i < len(shifted_route) - 1:
            ax, ay, az = _wp_xyz(shifted_route[i])
            bx, by, bz = _wp_xyz(shifted_route[i + 1])
            gap = math.sqrt((bx - ax) ** 2 + (by - ay) ** 2)
            if gap > _MAX_WP_GAP:
                n_nodes  = max(1, int(gap / (_MAX_WP_GAP * 0.5)))
                orig_ref = route[min(i + 1, len(route) - 1)]
                for k in range(1, n_nodes + 1):
                    t = k / (n_nodes + 1)
                    final_route.append(_OffsetWP(
                        orig_ref,
                        ax + t * (bx - ax),
                        ay + t * (by - ay),
                        az + t * (bz - az),
                    ))
        i += 1

    n_offset = sum(1 for wp in final_route if isinstance(wp, _OffsetWP))
    print(f"[route_detour] {n_offset} offset-WP / {len(final_route)} total  "
          f"(peak ±{peak_m:.1f} m, +{len(final_route)-n} connecting nodes).")
    return final_route


# ── CUSTOM ROUTE GENERATION ───────────────────────────────────────────────────

def _generate_custom_route(world: object, spawn_tf: object, max_wps: int,
                           mode: str, params: Optional[dict] = None) -> list:
    """Walk CARLA's waypoint graph from spawn_tf up to max_wps waypoints.

    mode="straight_left":
        At each junction prefer the option closest to the current heading
        (smallest absolute yaw delta).  When no option is within 45° of
        straight, pick the leftmost turn (most negative yaw delta).

    mode="line_then_left":
        params: {"target_y": float, "turn_x": float}
        While x < turn_x: at junctions pick the option whose y is closest
        to target_y.  Once x >= turn_x: pick leftmost option at every
        junction (same widening logic as straight_left).
    """
    params = params or {}
    carla_map = world.get_map()  # type: ignore[union-attr]
    wp = carla_map.get_waypoint(
        spawn_tf.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if wp is None:
        return []

    route = [wp]
    step_m = 1.0
    _first_turn_done  = False
    _POST_TURN_SHIFT  = 3.5   # extra metres forward after the first forced turn
    _turn_indices: list = []   # route indices where a forced turn was taken

    while len(route) < max_wps:
        nexts = wp.next(step_m)
        if not nexts:
            break
        if len(nexts) == 1:
            wp = nexts[0]
        else:
            cur_yaw = wp.transform.rotation.yaw
            deltas = []
            for n in nexts:
                d = ((n.transform.rotation.yaw - cur_yaw + 180.0) % 360.0) - 180.0
                deltas.append((abs(d), d, n))
            deltas.sort(key=lambda x: x[0])
            if mode == "straight_left" and deltas[0][0] > 45.0:
                # No straight option — pick leftmost (most negative delta)
                deltas.sort(key=lambda x: x[1])
                wp = deltas[0][2]
                _turn_indices.append(len(route))
                if not _first_turn_done:
                    extra = wp.next(_POST_TURN_SHIFT)
                    if extra:
                        wp = extra[0]
                    _first_turn_done = True
            elif mode == "line_then_left":
                _target_y   = float(params.get("target_y",  0.0))
                _turn_x     = float(params.get("turn_x",    100.0))
                _after_turn = params.get("after_turn", "left")
                cur_x       = wp.transform.location.x
                if cur_x < _turn_x and not _first_turn_done:
                    # Stay near target_y: pick option whose y is closest
                    wp = min(nexts, key=lambda n: abs(n.transform.location.y - _target_y))
                elif not _first_turn_done:
                    # Past turn_x — force left once
                    deltas.sort(key=lambda x: x[1])
                    wp = deltas[0][2]
                    _turn_indices.append(len(route))
                    extra = wp.next(_POST_TURN_SHIFT)
                    if extra:
                        wp = extra[0]
                    _first_turn_done = True
                elif _after_turn == "straight":
                    # After first turn: prefer straightest option
                    wp = deltas[0][2]
                else:
                    # After first turn: keep going left
                    deltas.sort(key=lambda x: x[1])
                    wp = deltas[0][2]
                    _turn_indices.append(len(route))
            else:
                wp = deltas[0][2]  # straightest
        route.append(wp)

    # Widen each turn: shift waypoints in a window around each turn apex outward
    # (right = outside of a left turn) using a cosine-bell profile so the AV
    # takes a wider arc and does not drift off on the exit.
    _WIDEN_M      = 1.2   # peak outward shift in metres
    _WIDEN_RADIUS = 12    # waypoints each side of apex affected

    class _TurnWP:
        __slots__ = ("transform", "is_junction")
        def __init__(self, orig, nx, ny, nz):
            self.transform   = carla.Transform(
                carla.Location(x=nx, y=ny, z=nz),
                orig.transform.rotation,
            )
            self.is_junction = bool(getattr(orig, "is_junction", False))

    for t_idx in _turn_indices:
        for j in range(max(0, t_idx - _WIDEN_RADIUS),
                       min(len(route), t_idx + _WIDEN_RADIUS)):
            t      = abs(j - t_idx) / _WIDEN_RADIUS
            shift  = _WIDEN_M * 0.5 * (1.0 + math.cos(t * math.pi))
            orig   = route[j]
            yaw_r  = math.radians(orig.transform.rotation.yaw)
            rgt_x  =  math.sin(yaw_r)
            rgt_y  = -math.cos(yaw_r)
            loc    = orig.transform.location
            route[j] = _TurnWP(orig, loc.x + shift * rgt_x,
                                      loc.y + shift * rgt_y,
                                      loc.z)

    # Virtual obstacle detour: apply the same cosine-bell offset as
    # _route_around_obstacles but against a list of (x, y) positions in
    # route_params["virtual_obstacles"].  peak_m defaults to _DETOUR_PEAK_M.
    _vobs  = params.get("virtual_obstacles", [])
    _vpeak = float(params.get("peak_m", _DETOUR_PEAK_M))
    if _vobs:
        _tr = _WP_CLEAR_RADIUS * _WP_TRANSITION_MULT
        for i in range(len(route)):
            orig = route[i]
            wx = orig.transform.location.x
            wy = orig.transform.location.y
            wz = orig.transform.location.z
            j_lo = max(0, i - 2); j_hi = min(len(route) - 1, i + 2)
            dx = route[j_hi].transform.location.x - route[j_lo].transform.location.x
            dy = route[j_hi].transform.location.y - route[j_lo].transform.location.y
            mag = math.sqrt(dx * dx + dy * dy)
            if mag < 1e-6:
                continue
            fwd_x, fwd_y = dx / mag, dy / mag
            rgt_x, rgt_y = fwd_y, -fwd_x
            total_lat = 0.0
            for ox, oy in _vobs:
                d = math.sqrt((wx - ox) ** 2 + (wy - oy) ** 2)
                if d >= _tr:
                    continue
                rgt_proj  = (ox - wx) * rgt_x + (oy - wy) * rgt_y
                direction = -1.0 if rgt_proj >= 0.0 else 1.0
                if d <= _WP_CLEAR_RADIUS:
                    total_lat += direction * _vpeak
                else:
                    t = (d - _WP_CLEAR_RADIUS) / (_tr - _WP_CLEAR_RADIUS)
                    total_lat += direction * _vpeak * 0.5 * (1.0 + math.cos(t * math.pi))
            if abs(total_lat) > 1e-4:
                route[i] = _TurnWP(orig, wx + total_lat * rgt_x,
                                         wy + total_lat * rgt_y, wz)
        print(f"[custom_route] Applied virtual obstacle detour: "
              f"{len(_vobs)} obstacle(s), peak={_vpeak}m.")

    print(f"[custom_route] Generated {len(route)} waypoints (mode={mode}, "
          f"{len(_turn_indices)} turn(s) widened ±{_WIDEN_M}m over {_WIDEN_RADIUS*2} wps).")
    return route


# ── PHASE 1: AV HEADLESS RUN ──────────────────────────────────────────────────

def run_av_episode(
    model_path:    str,
    style_profile: np.ndarray,
    scenario:      str,
    port:          int = 2000,
    mini_scenario: Optional[MiniScenario] = None,
    headless:      bool = False,
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

    config                        = Hyperparameters()
    config.scenario               = scenario
    config.next_map               = _SCENARIO_MAPS.get(scenario, "Town01")
    config.evaluate               = True
    config.evaluate_scenarios     = False        # use participant/training routes, not evaluation ones
    config.training_scenarios     = [scenario]   # route type matches the mini scenario being pre-run
    config.client_port            = port
    config.SPECATE                = True    # keep rendering ON so camera produces real images for the model
    config.waypoint_timeout_ticks = 10_000  # ~500 s at 20 FPS — lets AV finish full route

    _av_client = carla.Client("localhost", port)
    _av_client.set_timeout(30.0)

    # Destroy all actors left over from calibration or a previous iteration
    # before CarlaEnv resets the world, so the AV runs in a clean empty world.
    _av_world = _av_client.get_world()
    _leftover = [
        a for a in _av_world.get_actors()
        if a.type_id.startswith(("vehicle.", "walker.", "sensor.", "controller."))
    ]
    for _a in _leftover:
        try:
            _a.destroy()
        except Exception:
            pass
    if _leftover:
        _av_world.tick()

    env = CarlaEnv(_av_client, config)

    # Inject the calibrated style as fixed preference weights.
    # get_pref_weights_step() is called inside get_observation() every tick.
    _w = style_profile.astype(np.float32).copy()
    # Floor the speed weight so the AV always drives at a normal pace regardless
    # of how slowly the human drove during the short calibration phase.
    _w[0] = max(float(_w[0]), 0.6)
    env.pref_weights_round    = _w
    env.get_pref_weights      = lambda: None      # no-op; preserves _w set above
    env.get_pref_weights_step = lambda: _w.copy()

    model = PDMORL_TD3.load(model_path)

    traj:  AVTrajectory     = AVTrajectory()
    steps: List[AVStepData] = []
    sim_t  = 0.0
    done   = False
    tick   = 0

    obs, _ = env.reset()

    # Truncate route AFTER reset — CarlaEnv.reset() always overwrites max_waypoints
    # from its internal scenario table, so the only reliable way to shorten the
    # route is to slice it here after the fact.
    if mini_scenario is not None:
        if mini_scenario.route_type:
            _spawn_pts = _av_world.get_map().get_spawn_points()
            _spawn_tf  = _spawn_pts[mini_scenario.spawn_index % len(_spawn_pts)]
            env.route  = _generate_custom_route(_av_world, _spawn_tf, mini_scenario.route_length, mini_scenario.route_type, mini_scenario.route_params)
        env.route = env.route[:mini_scenario.route_length]
        # waypoint_logic uses config.max_waypoints for the done check and index
        # bounds (including a target_wp_ahead lookahead), so it must match the
        # filtered route length minus the lookahead to stay in bounds.
        env.config.max_waypoints = max(1, len(env.route) - env.config.target_wp_ahead)
    total  = max(len(env.route), 1)

    _hero_id  = env.vehicle.id
    _av_world = env.world
    if headless:
        # Headless mode: keep no_rendering_mode=True for speed.
        # A pygame progress window is shown below inside the game loop.
        try:
            _rs = _av_world.get_settings()
            _rs.no_rendering_mode = True
            _av_world.apply_settings(_rs)
        except Exception:
            pass
    else:
        # Rendered mode: force UE4 window ON so the user can watch the AV run.
        try:
            _rs = _av_world.get_settings()
            _rs.no_rendering_mode = False
            _av_world.apply_settings(_rs)
        except Exception:
            pass

    # Purge every vehicle and walker except the AV hero so the world is clean
    # before mini-scenario actors are placed.  This also removes any random
    # CarlaEnv background traffic whose positions we cannot reproduce for the
    # human run — keeping only the deterministic mini-scenario actors ensures
    # both the AV and the human encounter identical traffic.
    _to_destroy = [
        a for a in _av_world.get_actors()
        if (a.type_id.startswith("vehicle.") or a.type_id.startswith("walker."))
        and a.id != _hero_id
    ]
    if _to_destroy:
        print(f"[av_run] Clearing {len(_to_destroy)} CarlaEnv background actors …")
        for _a in _to_destroy:
            _a.destroy()
        _av_world.tick()

    # Spawn scenario obstacles so both AV and human see exactly the same layout.
    # spawn_index is read from the scenario and MUST match the spawnpoint that
    # CarlaEnv used above (see the scenarios table in carla_gym_env.py).  If they
    # diverge, obstacles land off-route and neither the AV nor the human driver
    # will encounter them.
    _obstacle_actors: List[object] = []
    _npc_actors:      List[object] = []
    _crash_actors:    List[object] = []   # tracked separately for steer-straighten logic
    if mini_scenario is not None:
        _si = mini_scenario.spawn_index
        _obstacle_actors, _no_detour_ids = _spawn_scenario_obstacles(_av_world, env.vehicle, mini_scenario, spawn_index=_si)
        _npc_actors      = _spawn_npc_autopilot(_av_world, env.vehicle, mini_scenario, spawn_index=_si)
        _crash_actors    = _spawn_npc_crash(_av_world, env.vehicle, mini_scenario, spawn_index=_si)
        _npc_actors     += _crash_actors
        print(f"[av_run] Spawned {len(_obstacle_actors)} static obstacle(s) "
              f"and {len(_npc_actors)} NPC(s) for '{mini_scenario.name}'.")
        if len(_obstacle_actors) < len(mini_scenario.obstacles):
            print(f"  WARNING: only {len(_obstacle_actors)}/{len(mini_scenario.obstacles)} "
                  f"static obstacles spawned — check spawn_index={_si} and blueprint IDs.")
        if len(_npc_actors) < len(mini_scenario.npc_autopilot) + len(mini_scenario.npc_crash):
            print(f"  WARNING: only {len(_npc_actors)}/"
                  f"{len(mini_scenario.npc_autopilot)+len(mini_scenario.npc_crash)} "
                  f"NPCs spawned — locations may be blocked.")
        # Apply smooth detour AFTER actors are physically placed so we read their
        # actual snapped positions (not recomputed from the scenario definition).
        # Obstacles with entry[6]==False are excluded from detour.
        _detour_actors = [a for a in _obstacle_actors if a.id not in _no_detour_ids] + _crash_actors
        if _detour_actors:
            env.route = _route_around_obstacles(
                env.route, _detour_actors,
                peak_m=mini_scenario.detour_peak_m
                       if mini_scenario.detour_peak_m is not None
                       else _DETOUR_PEAK_M,
            )
            env.config.max_waypoints = max(1, len(env.route) - env.config.target_wp_ahead)


    # ── Headless loading screen setup ─────────────────────────────────────────
    _loading_surf:  object = None   # pygame.Surface or None
    _loading_font:  object = None
    _loading_font_sm: object = None
    if headless and pygame is not None:
        try:
            if not pygame.get_init():
                pygame.init()
            _lw, _lh = 640, 180
            import os as _os
            _os.environ.setdefault("SDL_VIDEO_WINDOW_POS", "0,60")
            _loading_surf = pygame.display.set_mode((_lw, _lh))
            pygame.display.set_caption("AV Pre-run — Loading …")
            _loading_font    = pygame.font.SysFont("consolas", 20, bold=True)
            _loading_font_sm = pygame.font.SysFont("consolas", 14)
        except Exception:
            _loading_surf = None

    _av_driving_score  = 0.0
    _speed_print_time  = time.time()
    _prev_collision_cnt = 0          # for crash detection
    _stopped_ticks      = 0          # consecutive ticks at near-zero speed
    _was_stopped        = False
    _STOP_SPEED_MS      = 0.3        # m/s — below this counts as stopped
    _STOP_TICKS_THRESH  = FPS // 2   # must be stopped for ≥0.5 s before printing
    try:
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = env.step(action)
            sim_t += config.fixed_delta_seconds

            # ── Crash detection ───────────────────────────────────────────────
            _cur_col_cnt = int(getattr(env, "collision_cnt", 0))
            if _cur_col_cnt > _prev_collision_cnt:
                _col_type = info.get("collision_type", "unknown") if isinstance(info, dict) else "unknown"
                print(f"\n[av_run] *** AV CRASHED ***  t={sim_t:.1f}s  "
                      f"collision_type={_col_type}  "
                      f"wp={steps[-1].wp_index if steps else 0}/{total}")
            _prev_collision_cnt = _cur_col_cnt

            # ── Stop detection ────────────────────────────────────────────────
            _vel = env.vehicle.get_velocity()
            _speed_ms = (_vel.x**2 + _vel.y**2 + _vel.z**2) ** 0.5
            if _speed_ms < _STOP_SPEED_MS:
                _stopped_ticks += 1
                if _stopped_ticks == _STOP_TICKS_THRESH and not _was_stopped:
                    _was_stopped = True
                    print(f"\n[av_run] *** AV STOPPED ***  t={sim_t:.1f}s  "
                          f"wp={steps[-1].wp_index if steps else 0}/{total}")
            else:
                if _was_stopped:
                    print(f"[av_run] AV moving again  t={sim_t:.1f}s  "
                          f"speed={_speed_ms:.2f} m/s")
                _was_stopped   = False
                _stopped_ticks = 0

            _now = time.time()
            if _now - _speed_print_time >= 1.0:
                _speed_print_time = _now
                _speed_kmh = _speed_ms * 3.6
                _limit_kmh = env.vehicle.get_speed_limit()
                if not headless:
                    print(f"AV Speed: {_speed_kmh:.1f} km/h  (limit: {_limit_kmh:.0f} km/h)")

            # ── Update headless loading bar ────────────────────────────────
            if _loading_surf is not None:
                try:
                    pygame.event.pump()
                    _ratio = min(len(steps) / max(total, 1), 1.0)
                    _lw2, _lh2 = _loading_surf.get_size()  # type: ignore[union-attr]
                    _loading_surf.fill((18, 18, 18))  # type: ignore[union-attr]
                    _sc_name = mini_scenario.name if mini_scenario else scenario
                    _title = _loading_font.render(  # type: ignore[union-attr]
                        f"AV Pre-run: {_sc_name}", True, (220, 220, 220))
                    _loading_surf.blit(_title, (20, 18))  # type: ignore[union-attr]
                    # Progress bar background
                    _bx, _by, _bw, _bh = 20, 70, _lw2 - 40, 36
                    pygame.draw.rect(_loading_surf, (45, 45, 45), (_bx, _by, _bw, _bh), border_radius=6)
                    # Filled portion
                    _fill = max(4, int(_bw * _ratio))
                    pygame.draw.rect(_loading_surf, (0, 170, 90), (_bx, _by, _fill, _bh), border_radius=6)
                    # Percentage label
                    _pct_txt = _loading_font.render(  # type: ignore[union-attr]
                        f"{_ratio * 100:.0f}%", True, (255, 255, 255))
                    _loading_surf.blit(_pct_txt, (_bx + _bw // 2 - _pct_txt.get_width() // 2, _by + 5))  # type: ignore[union-attr]
                    # Step counter
                    _step_txt = _loading_font_sm.render(  # type: ignore[union-attr]
                        f"step {len(steps)} / {total}  |  sim {sim_t:.0f}s", True, (140, 140, 140))
                    _loading_surf.blit(_step_txt, (20, 124))  # type: ignore[union-attr]
                    pygame.display.flip()
                except Exception:
                    pass

            if done:
                _av_driving_score = float(info.get('adjusted_driving_core', 0.0))
                wp_now = steps[-1].wp_index if steps else 0
                print(f"\n[av_run] *** EPISODE TERMINATED ***")
                print(f"  sim_time          : {sim_t:.1f}s")
                print(f"  waypoint          : {wp_now} / {total}  ({wp_now/total*100:.1f}%)")
                print(f"  waypoint_logic    : {info.get('waypoint_logic_value', '?')}")
                print(f"  collision         : {info.get('collision', '?')}")
                print(f"  collision_type    : {info.get('collision_type', '?')}")
                print(f"  timeout_flag      : {getattr(env, 'timeout', '?')}")
                print(f"  timeout_ticks     : {getattr(env, 'timeout_ticks', '?')}")
                print(f"  episode_ticks     : {getattr(env, 'episode_ticks', '?')}")
                print(f"  collision_cnt     : {getattr(env, 'collision_cnt', '?')}")

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
            # After 1 s of sim time, straighten crash NPCs so they drive off in a
            # straight line instead of looping in circles.
            if tick == FPS and _crash_actors:
                for _ca in _crash_actors:
                    try:
                        _ca.apply_control(carla.VehicleControl(  # type: ignore[union-attr]
                            throttle=0.7, steer=0.0, brake=0.0))
                    except Exception:
                        pass
                print(f"[av_run] Crash NPCs straightened after 1 s.")
            if tick % FPS == 0:
                print(f"  [av_run] t={sim_t:.1f}s  wp={steps[-1].wp_index}/{total}"
                      f"  speed={steps[-1].speed:.1f}m/s")
    finally:
        for _obs in _obstacle_actors + _npc_actors:
            try:
                _obs.destroy()
            except Exception:
                pass
        env.close()
        # Comprehensive post-episode actor wipe — env.close() does not guarantee
        # all sensors/controllers/walkers are gone, causing accumulation over runs.
        import gc
        try:
            _post_world = _av_client.get_world()
            _post_actors = [
                a for a in _post_world.get_actors()
                if a.type_id.startswith(("vehicle.", "walker.", "sensor.", "controller."))
            ]
            for _a in _post_actors:
                try:
                    _a.destroy()
                except Exception:
                    pass
            if _post_actors:
                _post_world.tick()
        except Exception:
            pass
        gc.collect()

    prog = steps[-1].progress() * 100 if steps else 0.0
    print(f"\n[av_run] DONE → {len(steps)} steps, route completion {prog:.1f}%")
    return traj, steps, _av_driving_score


# ── PHASE 2: HUMAN RUN WITH ALERTS ────────────────────────────────────────────

def run_human_episode(
    client:           object,
    world:            object,
    traj:             AVTrajectory,
    av_steps:         List[AVStepData],
    alert_model:      MoEAlertModel,
    max_duration:     float = 300.0,
    force_arrow:      bool  = False,
    test_mode:        bool  = False,
    mini_scenario:    Optional[MiniScenario] = None,
    session_deadline: Optional[float] = None,   # wall-clock time.time() deadline; None = no limit
) -> Tuple[List[HumanStepData], float, float]:
    """Human drives with alert overlays drawn into the UE4 window.

    Alert model is sampled ONCE at episode start.
    GUI type and all parameters are fixed for the entire episode.
    Only positional rendering updates (arrow direction, colour, sound trigger).
    """
    print(f"\n{'='*60}")
    print(f"PHASE 2  HUMAN RUN WITH ALERTS  (max {int(max_duration)}s)")
    print("Watch the CARLA UE4 window. Follow the route shown by the alert.")
    print(f"{'='*60}")

    # Wipe all vehicles and walkers before spawning hero so the map is empty.
    _pre_clear = [
        a for a in world.get_actors()
        if a.type_id.startswith("vehicle.") or a.type_id.startswith("walker.")
    ]
    for _a in _pre_clear:
        _a.destroy()
    if _pre_clear:
        world.tick()

    _si   = mini_scenario.spawn_index if mini_scenario is not None else 8
    hero  = _spawn_hero(world, spawn_index=_si)
    regressor = HumanStyleRegressor()
    regressor.attach_collision_sensor(world, hero)

    # ── Sensors for human driving-score accumulators ───────────────────────
    _world_map = world.get_map()
    _bp_lib    = world.get_blueprint_library()

    _collision_cnt_env:    int   = 0
    _collision_cnt_car:    int   = 0
    _lane_invasion_cnt:    int   = 0
    _total_dist_to_center: float = 0.0
    _speeding_cnt:         int   = 0
    _episode_ticks:        int   = 0
    _timeout_flag:         float = 0.0
    _final_x:              float = 0.0
    _final_y:              float = 0.0

    def _on_collision_typed(event) -> None:
        nonlocal _collision_cnt_env, _collision_cnt_car
        if 'vehicle' in event.other_actor.type_id:
            _collision_cnt_car += 1
        else:
            _collision_cnt_env += 1

    def _on_lane_invasion(_event) -> None:
        nonlocal _lane_invasion_cnt
        _lane_invasion_cnt += 1

    _typed_collision_sensor = world.spawn_actor(
        _bp_lib.find('sensor.other.collision'),
        carla.Transform(), attach_to=hero,
    )
    _typed_collision_sensor.listen(_on_collision_typed)

    _lane_inv_sensor = world.spawn_actor(
        _bp_lib.find('sensor.other.lane_invasion'),
        carla.Transform(), attach_to=hero,
    )
    _lane_inv_sensor.listen(_on_lane_invasion)
    # ──────────────────────────────────────────────────────────────────────

    world.tick()  # settle hero physics so get_location().z is accurate for obstacle z

    _obstacle_actors: List[object] = []
    _npc_actors:      List[object] = []
    _crash_actors:    List[object] = []
    if mini_scenario is not None:
        _obstacle_actors, _ = _spawn_scenario_obstacles(world, hero, mini_scenario, spawn_index=_si)
        _npc_actors      = _spawn_npc_autopilot(world, hero, mini_scenario, spawn_index=_si)
        _crash_actors    = _spawn_npc_crash(world, hero, mini_scenario, spawn_index=_si)
        _npc_actors     += _crash_actors

    # ── Sample alert ONCE ─────────────────────────────────────────────────
    # Use the calibrated style profile as the initial state.  Since style is
    # constant for the episode, the model sees a stable distribution.
    first_av  = av_steps[0] if av_steps else AVStepData(0,0,0,0,0,0,0,0,0,1)
    init_meas = CarlaEnvUtils.get_vehicle_measurements(hero)
    init_state = _build_state(hero, init_meas, first_av, 0.0, 0.0)
    alert, log_prob = alert_model.sample(init_state)
    if force_arrow:
        _rng = np.random.default_rng()
        alert.gui_type   = 0                                          # arrow
        alert.color      = 0                                          # standard RGB (red/green)
        alert.location   = int(_rng.integers(0, 2))                   # random windshield/panel
        alert.vibration  = int(_rng.integers(0, 2))                   # random on/off
        alert.lag        = float(_rng.uniform(0.5, MAX_LAG))          # random 0.5–2.0 s
        alert.gui_params = _rng.random(3).astype(np.float32)          # random scale/opacity/vib_dist
        print(f"[human_run] force_arrow=True — fixed arrow, random params, lag={alert.lag:.2f}s")
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
    step_log:   List[HumanStepData] = []
    _ep_dists:  List[float]         = []   # euclidean distance human↔AV each tick
    sim_t = 0.0
    tick  = 0

    try:
        while True:
            if sim_t >= av_end_t:
                print(f"[human_run] AV reached last waypoint — ending human run ({sim_t:.1f}s).")
                break
            if ctrl_reader.quit_request:
                print("[human_run] User ended episode early (Q pressed).")
                break
            if session_deadline is not None and time.time() >= session_deadline:
                print(f"[human_run] Session time limit reached — stopping script.")
                raise SystemExit(0)

            ctrl = ctrl_reader.read()
            hero.apply_control(ctrl)
            world.tick()
            clock.tick(FPS)   # cap to FPS so sim time ≈ real time
            sim_t += _DT
            tick  += 1

            # Straighten crash NPCs after 1 s so they drive off straight
            # instead of looping in circles with constant steer.
            if tick == FPS and _crash_actors:
                for _ca in _crash_actors:
                    try:
                        _ca.apply_control(carla.VehicleControl(
                            throttle=0.7, steer=0.0, brake=0.0))
                    except Exception:
                        pass

            meas        = CarlaEnvUtils.get_vehicle_measurements(hero)
            human_style = regressor.tick(hero, world, _DT)
            av_step     = _nearest_av(av_steps, sim_t)
            progress    = float(np.clip(sim_t / max(av_end_t, 1e-9), 0.0, 1.0))

            loc  = hero.get_location()

            # ── Human driving-score accumulators (same metrics as AV env) ─
            _road_wp = _world_map.get_waypoint(loc, project_to_road=True)
            if _road_wp is not None:
                _wl = _road_wp.transform.location
                _total_dist_to_center += math.sqrt(
                    (loc.x - _wl.x) ** 2 + (loc.y - _wl.y) ** 2
                )
            if float(meas[0]) > float(CarlaEnvUtils.get_speed_limit_ms(hero)):
                _speeding_cnt += 1
            _episode_ticks += 1
            _final_x = loc.x
            _final_y = loc.y
            # ──────────────────────────────────────────────────────────────

            next_pos = traj.get_position_at(sim_t + 1.0)
            if next_pos is not None:
                hyaw   = math.radians(hero.get_transform().rotation.yaw)
                wp_ang = (math.atan2(next_pos[1] - loc.y,
                                     next_pos[0] - loc.x) - hyaw)
            else:
                wp_ang = 0.0

            # Per-tick reward: 1 - euclidean distance between human and AV positions.
            # Closer to the AV route → smaller distance → higher reward.
            _dist = math.sqrt((loc.x - av_step.x) ** 2 + (loc.y - av_step.y) ** 2)
            _ep_dists.append(_dist)
            score = 1.0 - _dist

            # ── CARLA debug rendering ─────────────────────────────────────
            if test_mode:
                _draw_av_dot(world, traj, sim_t)
            else:
                draw_alert(world, alert, hero, traj, sim_t)
            _spectator_follow(world, hero)
            vel  = hero.get_velocity()
            htf2 = hero.get_transform()
            _h2y = math.radians(htf2.rotation.yaw)
            _hx2 = loc.x + vel.x * _DT + 0.8 * math.cos(_h2y)
            _hy2 = loc.y + vel.y * _DT + 0.8 * math.sin(_h2y)
            _hz2 = loc.z + vel.z * _DT + 1.3
            world.debug.draw_string(
                location=carla.Location(x=_hx2, y=_hy2, z=_hz2 + 0.5),
                text=(f"t={sim_t:.0f}s  align={score:.2f}  "
                      + "  ".join(f"{l[0]}:{v:.2f}"
                                  for l, v in zip(STYLE_LABELS, human_style))),
                color=carla.Color(r=200, g=230, b=200),
                life_time=_DRAW_LT,
            )
            world.debug.draw_string(
                location=carla.Location(x=_hx2, y=_hy2, z=_hz2 + 0.1),
                text=f"x={loc.x:.1f}  y={loc.y:.1f}  z={loc.z:.1f}",
                color=carla.Color(r=255, g=255, b=100),
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
        for _obs in _obstacle_actors + _npc_actors:
            try:
                _obs.destroy()
            except Exception:
                pass
        for _s in (_typed_collision_sensor, _lane_inv_sensor):
            try:
                _s.stop()
                _s.destroy()
            except Exception:
                pass
        ctrl_reader.close()
        regressor.destroy_collision_sensor()
        hero.destroy()
        print("[human_run] Hero destroyed.")

    mean_dist = float(np.mean(_ep_dists)) if _ep_dists else 0.0

    # ── Human driving score — identical formula to AV adjusted_driving_core ──
    _total_wps     = av_steps[-1].total_waypoints if av_steps else 1
    _final_av_step = _nearest_av_by_position(av_steps, _final_x, _final_y)
    _route_comp    = float(_final_av_step.wp_index) / max(float(_total_wps - 1), 1.0)
    _avg_dist_ctr  = _total_dist_to_center / max(_episode_ticks, 1)
    human_driving_score = _compute_human_driving_score(
        _avg_dist_ctr, _collision_cnt_env, _collision_cnt_car,
        _lane_invasion_cnt, _speeding_cnt / FPS, _timeout_flag, _route_comp,
    )

    print(f"\n[human_run] DONE → {len(step_log)} steps  "
          f"mean_euclidean_dist={mean_dist:.3f}m  "
          f"human_driving_score={human_driving_score:.3f}")
    print(f"  route_completion={_route_comp:.3f}  avg_lane_dist={_avg_dist_ctr:.3f}m  "
          f"collisions(env={_collision_cnt_env} car={_collision_cnt_car})  "
          f"lane_inv={_lane_invasion_cnt}  speeding_ticks={_speeding_cnt}")
    return step_log, mean_dist, human_driving_score


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
        max_iterations:       int  = 25,
        host:                 str  = "localhost",
        port:                 int  = 2000,
        save_dir:             str   = "./pipeline_runs",
        participant_number:   int   = 0,
        session_duration:     float = 25 * 60,
        alert_mode:           str   = "adaptive",
        av_speedup:           float = 2.0,
        tutorial_duration:    int   = _TUTORIAL_DURATION,
        start_scenario:       int   = 0,
        headless_av:          bool  = False,
    ) -> None:
        if alert_mode not in ("fixed", "adaptive", "test"):
            raise ValueError(f"alert_mode must be 'fixed', 'adaptive', or 'test', got {alert_mode!r}")
        self.model_path           = model_path
        self.scenario             = scenario
        self.calibration_duration = calibration_duration
        self.max_iterations       = max_iterations
        self.host                 = host
        self.port                 = port
        self.save_dir             = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.participant_number   = participant_number
        self.session_duration     = session_duration
        self.alert_mode           = alert_mode
        self.av_speedup           = av_speedup
        self.tutorial_duration    = tutorial_duration
        self.start_scenario       = start_scenario
        self.headless_av          = headless_av

        self.alert_model    = MoEAlertModel(state_dim=DEFAULT_STATE_DIM)
        self.style_profile: Optional[np.ndarray] = None
        self._loss_history: deque = deque(maxlen=_CONV_WINDOW)

        # Per-participant CSV log
        self._participant_csv = self.save_dir / f"participant_{participant_number}_log.csv"
        with open(self._participant_csv, "w", newline="") as _f:
            csv.writer(_f).writerow([
                "participant_number", "scenario", "iteration",
                "driving_score", "episode_loss",
                "style_speed", "style_efficiency", "style_aggressiveness", "style_comfort",
                "gui_type", "gui_location", "gui_color", "gui_vibration", "gui_lag",
                "gui_p0_name", "gui_p0_val",
                "gui_p1_name", "gui_p1_val",
                "gui_p2_name", "gui_p2_val",
            ])

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
        # ── Phase 0: Tutorial + Calibration ──────────────────────────────────
        # Calibration runs FIRST so the AV pre-runs use the participant's real
        # style profile rather than a generic default.
        cal_map = _SCENARIO_MAPS.get(MINI_SCENARIOS[0].base_scenario, "Town01")
        client, world = _connect(self.host, self.port, render=True)
        world = _load_map(client, world, cal_map)
        try:
            if self.tutorial_duration > 0:
                run_tutorial(client, world, duration=self.tutorial_duration)
            print(f"\n[pipeline] Starting calibration ({self.calibration_duration}s) …")
            self.style_profile = run_calibration(client, world, self.calibration_duration)
        finally:
            _disconnect(world)
        print(f"[pipeline] Style profile: {self.style_profile.tolist()}")

        # ── Phase 1: Pre-run every scenario's AV trajectory ───────────────────
        # Always recalculated fresh after calibration so trajectories reflect
        # the participant's actual style profile.
        _active_scenarios = MINI_SCENARIOS[self.start_scenario:]
        if self.start_scenario:
            print(f"\n[pipeline] Starting from scenario {self.start_scenario} "
                  f"({MINI_SCENARIOS[self.start_scenario].name}) — "
                  f"{len(_active_scenarios)} scenario(s) active.")
        _prerun_cache: List[tuple] = []
        print(f"\n[pipeline] Pre-running {len(_active_scenarios)} AV scenarios "
              f"with participant style (speedup={self.av_speedup:.1f}x) …", flush=True)
        for _i, _ms in enumerate(_active_scenarios):
            _sid = self.start_scenario + _i
            print(f"  [{_i+1}/{len(_active_scenarios)}]  scenario {_sid}: {_ms.name}", flush=True)
            try:
                _traj, _av_st, _av_sc = run_av_episode(
                    self.model_path, self.style_profile, _ms.base_scenario, self.port,
                    mini_scenario=_ms,
                    headless=self.headless_av,
                )
                for _pt in _traj._points:
                    _pt.t     /= self.av_speedup
                    _pt.speed *= self.av_speedup
                for _st in _av_st:
                    _st.sim_time /= self.av_speedup
                _prerun_cache.append((_ms, _traj, _av_st, _av_sc))
            except Exception:
                import traceback
                print(f"  WARNING: {_ms.name} pre-run failed — skipped.\n"
                      + traceback.format_exc())
        print(f"[pipeline] {len(_prerun_cache)}/{len(_active_scenarios)} scenarios pre-run.")

        if not _prerun_cache:
            raise RuntimeError("No AV scenarios pre-computed successfully — cannot continue.")

        # ── Scenario loop ─────────────────────────────────────────────────────
        # Randomly pick from pre-computed trajectories.  Only human-drive time
        # counts against the session clock.
        print(f"\n[pipeline] Alert mode: {self.alert_mode.upper()}")
        print(f"[pipeline] Session: {self.session_duration/60:.1f} min of human drive time")
        session_human_elapsed = 0.0
        _session_wall_start   = time.time()
        _session_wall_end     = _session_wall_start + self.session_duration
        scenario_idx = 0
        _plot_av_scores:    List[float] = []   # AV driving reward per iteration
        _plot_align_scores: List[float] = []   # mean human-AV alignment per iteration
        while (session_human_elapsed < self.session_duration
               and scenario_idx < self.max_iterations):
            elapsed_min   = session_human_elapsed / 60
            remaining_min = (self.session_duration - session_human_elapsed) / 60
            mini, traj, av_steps, _ = random.choice(_prerun_cache)
            scenario_map  = _SCENARIO_MAPS.get(mini.base_scenario, "Town01")

            _scenario_id = next((i for i, ms in enumerate(MINI_SCENARIOS) if ms is mini), -1)
            _w = 72
            _sep = "=" * _w
            _tag = f"  SCENARIO {_scenario_id} : {mini.name.upper()}  "
            _tag = _tag.center(_w)
            _iter = f"  iteration {scenario_idx+1}  |  {elapsed_min:.1f} min elapsed  |  {remaining_min:.1f} min left  "
            _iter = _iter.center(_w)
            print(f"\n{_sep}", flush=True)
            print(_sep, flush=True)
            print(_tag, flush=True)
            print(_iter, flush=True)
            print(_sep, flush=True)
            print(_sep, flush=True)

            try:
                # Phase 2 — human run (counted against session clock)
                client, world = _connect(self.host, self.port, render=True)
                # Force a full world reload every 5 iterations to flush CARLA's
                # accumulated actor/resource state and prevent server degradation.
                if scenario_idx > 0 and scenario_idx % 5 == 0:
                    print(f"[pipeline] Reloading world to flush CARLA state "
                          f"(iteration {scenario_idx + 1}) …", flush=True)
                    world = client.load_world(scenario_map)
                    s = world.get_settings()
                    s.synchronous_mode    = True
                    s.fixed_delta_seconds = _DT
                    s.no_rendering_mode   = False
                    world.apply_settings(s)
                else:
                    world = _load_map(client, world, scenario_map)
                _human_start = time.time()
                try:
                    step_log, mean_euclidean_dist, human_driving_score = run_human_episode(
                        client, world, traj, av_steps,
                        self.alert_model,
                        force_arrow=(self.alert_mode == "fixed"),
                        test_mode=(self.alert_mode == "test"),
                        mini_scenario=mini,
                        session_deadline=_session_wall_end,
                    )
                finally:
                    # Destroy all remaining actors before releasing sync mode so
                    # nothing accumulates across episodes.
                    import gc
                    try:
                        _leftover_h = [
                            a for a in world.get_actors()
                            if a.type_id.startswith(
                                ("vehicle.", "walker.", "sensor.", "controller."))
                        ]
                        for _a in _leftover_h:
                            try:
                                _a.destroy()
                            except Exception:
                                pass
                        if _leftover_h:
                            world.tick()
                    except Exception:
                        pass
                    gc.collect()
                    _disconnect(world)

                session_human_elapsed += time.time() - _human_start

                # Phase 3 — train (must run before score display so we have the loss)
                flush_training(self.alert_model, step_log)

                # Alert loss: mean euclidean distance (metres) between the human's
                # position and the synchronised AV position throughout the run.
                alert_loss = mean_euclidean_dist

                # Track per-iteration metrics for end-of-session plots
                _plot_av_scores.append(human_driving_score)
                _plot_align_scores.append(alert_loss)

                # Large pygame score screen (shown after training completes)
                if step_log:
                    _show_score_pygame(
                        mini.name, scenario_idx + 1,
                        human_driving_score, alert_loss,
                        duration=10.0,
                    )
                self._loss_history.append(alert_loss)
                self._save_csv(step_log, scenario_idx)
                self._save(scenario_idx)
                with open(self._participant_csv, "a", newline="") as _f:
                    # Mean style scores across all ticks of this episode
                    _ep_styles = np.mean(
                        [s.style_scores for s in step_log], axis=0
                    ).tolist() if step_log else [0.0, 0.0, 0.0, 0.0]
                    # Alert vector (fixed for episode — read from first step)
                    _ep_av = AlertVector.from_raw(
                        np.array(step_log[0].alert_raw, dtype=np.float32)
                    ) if step_log else None
                    _pnames = _ep_av.param_names if _ep_av else ["p0", "p1", "p2"]
                    _pvals  = _ep_av.gui_params.tolist() if _ep_av else [0.0, 0.0, 0.0]
                    csv.writer(_f).writerow([
                        self.participant_number, mini.name, scenario_idx,
                        human_driving_score, alert_loss,
                        # style
                        round(_ep_styles[0], 4), round(_ep_styles[1], 4),
                        round(_ep_styles[2], 4), round(_ep_styles[3], 4),
                        # alert GUI metadata
                        _ep_av.gui_name if _ep_av else "",
                        "windshield" if (_ep_av and _ep_av.location) else "panel",
                        "colorblind" if (_ep_av and _ep_av.color) else "RGB",
                        bool(_ep_av.vibration) if _ep_av else False,
                        round(_ep_av.lag, 3) if _ep_av else 0.0,
                        _pnames[0], round(_pvals[0], 4),
                        _pnames[1], round(_pvals[1], 4),
                        _pnames[2], round(_pvals[2], 4),
                    ])

                _style_str = "  ".join(
                    f"{l}={v:.3f}" for l, v in zip(STYLE_LABELS, _ep_styles)
                )
                _gui_str = (
                    f"{_ep_av.gui_name}  lag={_ep_av.lag:.2f}s  "
                    f"loc={'windshield' if _ep_av.location else 'panel'}  "
                    f"color={'colorblind' if _ep_av.color else 'RGB'}  "
                    f"vib={bool(_ep_av.vibration)}  "
                    + "  ".join(
                        f"{n}={v:.3f}" for n, v in zip(_pnames, _pvals)
                    )
                ) if _ep_av else "N/A"
                print(f"\n[pipeline] Iteration {scenario_idx+1} ('{mini.name}') done.  "
                      f"HumanScore={human_driving_score:.4f}  AlertLoss={alert_loss:.4f}m  "
                      f"History={[f'{v:.4f}' for v in self._loss_history]}")
                print(f"  HumanStyle  : {_style_str}")
                print(f"  Alert GUI   : {_gui_str}")

            except Exception:
                import traceback
                print(f"\n[pipeline] WARNING — iteration {scenario_idx+1} ('{mini.name}') "
                      f"failed and will be skipped:\n{traceback.format_exc()}")

            scenario_idx += 1

        print(f"\n[pipeline] Session complete after {scenario_idx} iteration(s) "
              f"({session_human_elapsed/60:.1f} min driven).")

        # ── Auto-generate session plots ────────────────────────────────────────
        if _plot_av_scores:
            try:
                import matplotlib
                matplotlib.use("Agg")           # non-interactive — saves to file, no GUI
                import matplotlib.pyplot as plt  # type: ignore[import]

                iters = list(range(1, len(_plot_av_scores) + 1))

                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
                fig.suptitle(f"Participant {self.participant_number} — Session Summary",
                             fontsize=14, fontweight="bold")

                ax1.plot(iters, _plot_av_scores, "b-o", linewidth=2, markersize=6)
                ax1.set_ylabel("Human Driving Score (0–1)")
                ax1.set_title("Human Driving Score per Iteration")
                ax1.set_ylim(0, max(1.0, max(_plot_av_scores) * 1.1))
                ax1.grid(True, alpha=0.3)

                ax2.plot(iters, _plot_align_scores, "r-o", linewidth=2, markersize=6)
                ax2.set_ylabel("Mean Euclidean Distance (m)")
                ax2.set_title("Alert Loss (Mean Human↔AV Euclidean Distance) per Iteration")
                ax2.set_xlabel("Iteration")
                ax2.set_ylim(0, max(1.0, max(_plot_align_scores) * 1.1))
                ax2.grid(True, alpha=0.3)

                plt.tight_layout()
                _plot_path = self.save_dir / f"session_plots_p{self.participant_number}.png"
                plt.savefig(str(_plot_path), dpi=150)
                plt.close(fig)
                print(f"[pipeline] Plots saved → {_plot_path}")
            except Exception as _pe:
                print(f"[pipeline] WARNING: could not generate plots: {_pe}")


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


# ── TEST-MODE RUNNER ──────────────────────────────────────────────────────────

def _run_test_mode(
    model_path:           str,
    scenario_id:          int   = 0,
    host:                 str   = "localhost",
    port:                 int   = 2000,
    alert_mode:           str   = "adaptive",
    av_speedup:           float = 1.0,
    calibration_duration: int   = 0,
) -> None:
    """Single-scenario debug loop: (optional calibration →) AV run → human run.

    Use ``--test-mode --scenario-id N`` on the command line.
    If ``--calibration-duration`` is provided (> 0) the participant drives freely
    first and the resulting style profile is used for the AV.  Otherwise a
    neutral profile is used (speed=0.6, all others=0.5).
    The human run uses a freshly-initialised MoEAlertModel so the prompt
    system fires exactly as it would in a real session.
    """
    if scenario_id < 0 or scenario_id >= len(MINI_SCENARIOS):
        raise SystemExit(
            f"--scenario-id {scenario_id} is out of range "
            f"(0 – {len(MINI_SCENARIOS)-1}).  "
            f"Run with --list-scenarios to see valid IDs."
        )

    ms = MINI_SCENARIOS[scenario_id]
    print(f"\n{'='*60}")
    print(f"TEST MODE  scenario_id={scenario_id}  name='{ms.name}'")
    print(f"  base_scenario : {ms.base_scenario}")
    print(f"  spawn_index   : {ms.spawn_index}")
    print(f"  obstacles     : {ms.obstacles}")
    print(f"  npc_crash     : {ms.npc_crash}")
    print(f"  route_length  : {ms.route_length}")
    print(f"{'='*60}\n")

    # ── Calibration (optional) ────────────────────────────────────────────────
    if calibration_duration > 0:
        print(f"[test_mode] Running calibration ({calibration_duration}s) …")
        cal_map = _SCENARIO_MAPS.get(ms.base_scenario, "Town01")
        _cal_client, _cal_world = _connect(host, port, render=True)
        _cal_world = _load_map(_cal_client, _cal_world, cal_map)
        try:
            style = run_calibration(_cal_client, _cal_world, calibration_duration)
        finally:
            _disconnect(_cal_world)
        print(f"[test_mode] Style profile from calibration: {style.tolist()}")
    else:
        # Neutral style — speed floored so the AV actually moves
        style = np.array([0.6, 0.5, 0.5, 0.5], dtype=np.float32)
        print(f"[test_mode] Using neutral style profile (no calibration).")

    # ── Phase 1: AV run ───────────────────────────────────────────────────────
    print("[test_mode] Running AV episode …")
    traj, av_steps, av_score = run_av_episode(
        model_path=model_path,
        style_profile=style,
        scenario=ms.base_scenario,
        port=port,
        mini_scenario=ms,
        headless=False,
    )
    # Apply speedup (default 1× in test mode so the human sees real AV speed)
    if av_speedup != 1.0:
        for pt in traj._points:
            pt.t     /= av_speedup
            pt.speed *= av_speedup
        for st in av_steps:
            st.sim_time /= av_speedup

    print(f"[test_mode] AV done — {len(av_steps)} steps, score={av_score:.3f}")

    # ── Phase 2: Human run ────────────────────────────────────────────────────
    print("[test_mode] Starting human episode …")
    scenario_map = _SCENARIO_MAPS.get(ms.base_scenario, "Town01")
    client, world = _connect(host, port, render=True)
    world = _load_map(client, world, scenario_map)

    alert_model = MoEAlertModel(state_dim=DEFAULT_STATE_DIM)
    force_arrow = (alert_mode == "fixed")

    try:
        step_log, _, _ = run_human_episode(
            client=client,
            world=world,
            traj=traj,
            av_steps=av_steps,
            alert_model=alert_model,
            max_duration=300.0,
            force_arrow=force_arrow,
            test_mode=(alert_mode == "test"),
            mini_scenario=ms,
            session_deadline=None,
        )
    finally:
        _disconnect(world)

    print(f"[test_mode] Human done — {len(step_log)} steps recorded.")
    if step_log:
        scores = [s.alert_score for s in step_log]
        print(f"[test_mode] Mean alert score: {np.mean(scores):.3f}  "
              f"min={min(scores):.3f}  max={max(scores):.3f}")


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
    p.add_argument("--participant-number", type=int, default=0,
                   help="Participant ID — used to name the per-participant CSV log")
    p.add_argument("--session-duration", type=float, default=25.0,
                   help="Total session duration in minutes (default: 25)")
    p.add_argument("--av-speedup", type=float, default=2.0,
                   help="Multiply AV trajectory speed by this factor during human runs (default: 2.0)")
    p.add_argument("--alert-mode", default="adaptive", choices=["fixed", "adaptive", "test"],
                   help="'fixed' forces arrow alert every episode; "
                        "'adaptive' lets the model sample the alert type; "
                        "'test' shows a moving dot along the AV trajectory (no alert) (default: adaptive)")
    p.add_argument("--tutorial-duration", type=int, default=_TUTORIAL_DURATION,
                   help=f"Tutorial length in seconds (default: {_TUTORIAL_DURATION}). "
                        "Set to 0 to skip the tutorial entirely.")
    p.add_argument("--start-scenario", type=int, default=0,
                   help="Scenario index to start from; that scenario and all following ones "
                        "will be included in the session pool (default: 0 = all scenarios). "
                        "Run with --list-scenarios to see available IDs.")
    p.add_argument("--headless-av", action="store_true",
                   help="Run AV pre-run episodes without CARLA rendering (faster). "
                        "The participant will not see the AV drive during pre-run.")
    p.add_argument("--test", action="store_true",
                   help="Run smoke tests (no CARLA or model needed) then exit")
    p.add_argument("--test-mode", action="store_true",
                   help="Debug mode: run AV then human for a single mini-scenario "
                        "(no calibration, no training loop). "
                        "Use --scenario-id to pick the scenario.")
    p.add_argument("--scenario-id", type=int, default=0,
                   help="Index into MINI_SCENARIOS list for --test-mode (default: 0). "
                        "Run with --list-scenarios to see available IDs.")
    p.add_argument("--list-scenarios", action="store_true",
                   help="Print all available mini-scenario IDs and names then exit")
    args = p.parse_args()

    if args.list_scenarios:
        print("Available mini-scenarios:")
        for _idx, _ms in enumerate(MINI_SCENARIOS):
            print(f"  {_idx:2d}  {_ms.name}  "
                  f"(base={_ms.base_scenario}, spawn={_ms.spawn_index}, "
                  f"obstacles={len(_ms.obstacles)})")
        return

    if args.test:
        _run_smoke_test()
        return

    model_path = find_model(args.model)   # resolves or raises with a clear message

    if args.test_mode:
        _run_test_mode(
            model_path=model_path,
            scenario_id=args.scenario_id,
            host=args.host,
            port=args.port,
            alert_mode=args.alert_mode,
            av_speedup=args.av_speedup,
            calibration_duration=args.calibration_duration,
        )
        return

    AlertPipeline(
        model_path=model_path,
        scenario=args.scenario,
        calibration_duration=args.calibration_duration,
        max_iterations=args.max_iterations,
        host=args.host,
        port=args.port,
        save_dir=args.save_dir,
        participant_number=args.participant_number,
        session_duration=args.session_duration * 60,
        alert_mode=args.alert_mode,
        av_speedup=args.av_speedup,
        tutorial_duration=args.tutorial_duration,
        start_scenario=args.start_scenario,
        headless_av=args.headless_av,
    ).run()


if __name__ == "__main__":
    main()
