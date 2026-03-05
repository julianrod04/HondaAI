"""human_style_regression.py

Continuously estimates a human driver's driving style from live CARLA telemetry
using the EXACT same reward functions used to train the autonomous vehicle
(sb3/carla_env_utils.py: CarlaEnvUtils).

The 4 AV training objectives map to a driver style profile vector in [0, 1]:

  AV reward fn              Style dim  Meaning
  ─────────────────────────────────────────────────────────────────────────────
  CarlaEnvUtils.speed_reward()     [0] speed          0=slow,    1=at speed limit
  CarlaEnvUtils.efficiency()       [1] efficiency     0=erratic, 1=smooth/fast
  CarlaEnvUtils.aggressiveness()   [2] aggressiveness 0=passive, 1=aggressive
  CarlaEnvUtils.comfort()          [3] comfort/safety 0=jerky,   1=smooth/safe

Each raw reward is normalised to [0, 1] using the same output ranges produced
by the AV training environment, so the human's style scores sit on the same
scale as the AV's reward signal.

The profile vector is compatible with MoEAlertModel / GaussianAlertModel as the
driver-profile state input, enabling the alert system to personalise outputs to
match the human's driving style and generate a digital twin AV.

Usage
─────
Standalone (CARLA must be running):
    python human_style_regression.py
    python human_style_regression.py --npc          # spawn traffic ahead
    python human_style_regression.py --buffer 300   # shorter averaging window

Library:
    from human_style_regression import HumanStyleRegressor

    regressor = HumanStyleRegressor()
    regressor.attach_collision_sensor(world, ego_vehicle)

    # Inside the game loop (pass current world for traffic detection):
    regressor.tick(ego_vehicle, world)
    profile = regressor.get_style_profile()  # np.ndarray shape (4,) in [0, 1]
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

# ── CARLA optional import ─────────────────────────────────────────────────────
try:
    import carla
except ImportError:
    carla = None

# ── sb3 package: exact reward functions used by the AV ───────────────────────
# sb3/ lives two levels above kw_sandbox (under HondaAI/).
_SB3_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent / "sb3")
if _SB3_ROOT not in sys.path:
    sys.path.insert(0, _SB3_ROOT)

from carla_env_utils import CarlaEnvUtils
from config import Hyperparameters  # for fixed_delta_seconds used by comfort()

# Sibling-directory imports for manual_drive helpers
sys.path.insert(0, str(Path(__file__).parent))


# ─────────────────────────────────────────────────────────────────────────────
# Constants — normalisation ranges matching the sb3 AV reward functions
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_BUFFER_SIZE = 500_000   # large enough to hold every tick of any session;
                                # ensures get_style_profile() is a true mean of ALL samples

# Speed reward: Path C (normal driving) spans [-2.25, 1.75]
_SPEED_REWARD_MIN =  -2.25
_SPEED_REWARD_MAX =   1.75
_SPEED_REWARD_RANGE = _SPEED_REWARD_MAX - _SPEED_REWARD_MIN  # 4.0

# Efficiency: already [0, 1] — no additional normalisation needed

# Aggressiveness: max = long_max + lat_max + yaw_max - 0 = 1.1 + 1.1 + 0.075 = 2.275
# (clip(11,0,11)*0.1 + clip(11,0,11)*0.1 + clip(0.25,0,0.25)*0.3)
_AGGR_MAX = 2.275

# Comfort: base jerk bonus = 1.2; max penalties ≈ 0.05+0.1+0.075+0.09+0.15 = 0.465
# Practical range: approximately [-1.0, 1.2]
_COMFORT_MIN = -1.0
_COMFORT_MAX =  1.2
_COMFORT_RANGE = _COMFORT_MAX - _COMFORT_MIN  # 2.2

# Style vector indices
SPEED_IDX          = 0
EFFICIENCY_IDX     = 1
AGGRESSIVENESS_IDX = 2
COMFORT_IDX        = 3
STYLE_DIM          = 4

STYLE_LABELS = ["speed", "efficiency", "aggressiveness", "comfort"]


# ─────────────────────────────────────────────────────────────────────────────
# HumanStyleRegressor
# ─────────────────────────────────────────────────────────────────────────────

class HumanStyleRegressor:
    """Grades a human driver using the exact reward pipeline that trains the AV.

    Calls CarlaEnvUtils.speed_reward(), efficiency(), aggressiveness(), and
    comfort() each tick — the same four objectives the AV is optimised on.
    Raw reward values are normalised to [0, 1] using the same output ranges
    produced by the AV training environment.

    The rolling mean of the buffer converges to a stable style profile that
    can be fed directly into MoEAlertModel / GaussianAlertModel as the
    driver-profile state.

    Args:
        buffer_size:  number of recent tick scores to average (ring buffer)
        config:       Hyperparameters instance matching the AV training config;
                      if None the default config is used (target_speed_perc=0.95,
                      fixed_delta_seconds=1/20).
    """

    def __init__(
        self,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        config: Optional[Hyperparameters] = None,
    ) -> None:
        self._buffer: deque = deque(maxlen=buffer_size)
        self._cfg = config or Hyperparameters()
        # Cumulative sum / count so the final profile is always the mean of
        # every single sample collected, regardless of buffer_size.
        self._cumsum: np.ndarray = np.zeros(STYLE_DIM, dtype=np.float64)
        self._cumcount: int = 0

        # Previous-tick state required by aggressiveness() and comfort()
        self._prev_steering:         float = 0.0
        self._prev_throttle:         float = 0.0
        self._prev_acc_longitudinal: float = 0.0
        self._prev_velo:             float = 0.0
        self._prev_yaw:              float = 0.0
        self._prev_acc_vec                 = None  # carla.Vector3D or None

        # Collision counter and per-tick flag for aggressiveness boosting
        self._collision_cnt: int = 0
        self._collision_this_tick: bool = False  # set by sensor callback, cleared each tick
        self._collision_sensor       = None

        self._prev_time: float = time.time()
        self._n_ticks:   int   = 0

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def attach_collision_sensor(self, world, ego_vehicle) -> None:
        """Spawn a CARLA collision sensor so aggressiveness() sees collisions.

        Call once after spawning the ego vehicle.
        """
        bp = world.get_blueprint_library().find("sensor.other.collision")
        self._collision_sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=ego_vehicle
        )
        self._collision_sensor.listen(self._on_collision)

    def destroy_collision_sensor(self) -> None:
        """Stop and destroy the attached collision sensor."""
        if self._collision_sensor is not None:
            self._collision_sensor.stop()
            self._collision_sensor.destroy()
            self._collision_sensor = None

    def tick(self, ego_vehicle, world, dt: Optional[float] = None) -> np.ndarray:
        """Compute per-tick style scores using the sb3 AV reward functions.

        Call once per simulation tick inside the game loop.

        Args:
            ego_vehicle: carla.Vehicle — the human-driven hero car
            world:       carla.World   — used to detect nearby traffic
            dt:          elapsed seconds since last tick (auto-measured if None)

        Returns:
            np.ndarray shape (4,) dtype float32 — per-tick
            [speed, efficiency, aggressiveness, comfort], all in [0, 1].
        """
        now = time.time()
        if dt is None:
            dt = max(now - self._prev_time, 1e-4)
        self._prev_time = now

        # ── Measurements (identical to sb3 AV environment) ────────────────────
        measurements = CarlaEnvUtils.get_vehicle_measurements(ego_vehicle)

        # ── Traffic list for blocked/reduced detection ─────────────────────────
        # Include all vehicles except the ego so CarlaEnvUtils.closest_car()
        # can compute whether the human is blocked by / following another car.
        traffic_list = [
            v for v in world.get_actors().filter("vehicle.*")
            if v.id != ego_vehicle.id
        ]
        _, blocked, reduced = CarlaEnvUtils.closest_car(
            ego_vehicle, traffic_list
        )

        # ── Jerk (required by comfort()) ──────────────────────────────────────
        curr_acc_vec = ego_vehicle.get_acceleration()
        if self._prev_acc_vec is not None:
            _, jerk_magnitude = CarlaEnvUtils.get_jerk(
                self._cfg.fixed_delta_seconds, curr_acc_vec, self._prev_acc_vec
            )
        else:
            jerk_magnitude = 0.0

        # ── Extract current control inputs ────────────────────────────────────
        control  = ego_vehicle.get_control()
        steering = float(control.steer)
        throttle = float(control.throttle)

        # ── Call the exact sb3 reward functions ───────────────────────────────
        raw_speed = CarlaEnvUtils.speed_reward(
            ego_vehicle, measurements,
            self._cfg.target_speed_perc,
            blocked, reduced,
        )

        raw_efficiency = CarlaEnvUtils.efficiency(
            ego_vehicle, measurements, throttle
        )

        # Always pass collision_cnt=0 so physics (acceleration/yaw) are computed every tick.
        # Then spike aggressiveness if a collision occurred this tick.
        raw_aggr = CarlaEnvUtils.aggressiveness(
            ego_vehicle, measurements,
            0,
            self._prev_acc_longitudinal,
            done=False,
        )
        if self._collision_this_tick:
            raw_aggr += _AGGR_MAX  # collision → max aggressiveness for this tick
        self._collision_this_tick = False

        raw_comfort = CarlaEnvUtils.comfort(
            ego_vehicle, measurements,
            self._prev_steering, self._prev_throttle,
            steering, throttle,
            self._prev_acc_longitudinal,
            self._prev_velo,
            self._prev_yaw,
            jerk_magnitude,
            self._cfg,
        )

        # ── Normalise to [0, 1] ───────────────────────────────────────────────
        scores = np.array([
            float(np.clip((raw_speed - _SPEED_REWARD_MIN) / _SPEED_REWARD_RANGE, 0.0, 1.0)),
            float(np.clip(raw_efficiency,                                         0.0, 1.0)),
            float(np.clip(raw_aggr / _AGGR_MAX,                                  0.0, 1.0)),
            float(np.clip((raw_comfort - _COMFORT_MIN) / _COMFORT_RANGE,         0.0, 1.0)),
        ], dtype=np.float32)

        self._buffer.append(scores)
        self._cumsum   += scores.astype(np.float64)
        self._cumcount += 1
        self._n_ticks  += 1

        # ── Update previous-tick state ────────────────────────────────────────
        self._prev_steering         = steering
        self._prev_throttle         = throttle
        self._prev_acc_longitudinal = float(measurements[5])
        self._prev_velo             = float(measurements[0])
        self._prev_yaw              = float(measurements[7])
        self._prev_acc_vec          = curr_acc_vec

        return scores

    def get_style_profile(self) -> np.ndarray:
        """Return the mean style profile over ALL samples ever collected.

        Uses a cumulative sum/count so the result is always the true session
        mean, not just the last window of the rolling buffer.

        Returns:
            np.ndarray shape (4,) dtype float32 in [0, 1].
            Returns [0.5, 0.5, 0.5, 0.5] before any ticks are observed.
        """
        if self._cumcount == 0:
            return np.full(STYLE_DIM, 0.5, dtype=np.float32)
        return (self._cumsum / self._cumcount).astype(np.float32)

    def get_stats(self) -> dict:
        """Return diagnostic statistics including the current style profile."""
        profile = self.get_style_profile()
        return {
            "n_ticks":        self._n_ticks,
            "buffer_fill":    len(self._buffer),
            "speed":          float(profile[SPEED_IDX]),
            "efficiency":     float(profile[EFFICIENCY_IDX]),
            "aggressiveness": float(profile[AGGRESSIVENESS_IDX]),
            "comfort":        float(profile[COMFORT_IDX]),
        }

    def reset(self) -> None:
        """Clear the buffer (e.g., when a new driver takes the wheel)."""
        self._buffer.clear()
        self._cumsum[:]  = 0.0
        self._cumcount   = 0
        self._prev_steering         = 0.0
        self._prev_throttle         = 0.0
        self._prev_acc_longitudinal = 0.0
        self._prev_velo             = 0.0
        self._prev_yaw              = 0.0
        self._prev_acc_vec          = None
        self._collision_cnt         = 0
        self._collision_this_tick   = False
        self._n_ticks               = 0

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _on_collision(self, _event) -> None:
        """CARLA collision sensor callback — increments the collision counter and flags this tick."""
        self._collision_cnt += 1
        self._collision_this_tick = True


# ─────────────────────────────────────────────────────────────────────────────
# Standalone main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import pygame
    from steering_control import get_keyboard_control, get_wheel_control
    from waypoint import find_forward_waypoint

    parser = argparse.ArgumentParser(description="Human style regression with CARLA")
    parser.add_argument("--npc",            action="store_true",
                        help="Spawn an AI vehicle ahead")
    parser.add_argument("--buffer",         type=int,   default=DEFAULT_BUFFER_SIZE,
                        help="Style buffer size in ticks")
    parser.add_argument("--print-interval", type=float, default=5.0,
                        help="Seconds between printed style profile updates")
    args = parser.parse_args()

    WINDOW_W, WINDOW_H = 1280, 720

    os.environ["SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS"] = "1"
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("CARLA Human Style Regression — ESC to quit")
    pygame.joystick.init()

    for _ in range(5):
        pygame.event.pump()
        time.sleep(0.1)

    wheel = None
    if pygame.joystick.get_count() > 0:
        wheel = pygame.joystick.Joystick(0)
        wheel.init()
        print(f"Using wheel: {wheel.get_name()}")
    else:
        print("No joystick found — using keyboard (WASD).")

    if carla is None:
        print("ERROR: carla module not found.", file=sys.stderr)
        sys.exit(1)

    client    = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world     = client.get_world()
    world_map = world.get_map()
    bp_lib    = world.get_blueprint_library()

    ego_bp    = bp_lib.filter("model3")[0]
    spawn_pts = world_map.get_spawn_points()
    ego_spawn = random.choice(spawn_pts)

    ego_vehicle = world.try_spawn_actor(ego_bp, ego_spawn)
    if ego_vehicle is None:
        raise RuntimeError("Failed to spawn ego vehicle.")
    ego_vehicle.set_autopilot(False)
    print(f"Spawned ego vehicle id={ego_vehicle.id}")

    npc_vehicle = None
    if args.npc:
        npc_spawn_tf = find_forward_waypoint(world_map, ego_spawn)
        if npc_spawn_tf is None:
            npc_spawn_tf = world_map.get_waypoint(ego_spawn.location).next(10.0)[0].transform
        npc_spawn_tf.location.x += 1.0
        npc_spawn_tf.location.z += 0.5
        npc_vehicle = world.try_spawn_actor(bp_lib.filter("model3")[0], npc_spawn_tf)
        if npc_vehicle is not None:
            tm = client.get_trafficmanager()
            npc_vehicle.set_autopilot(True, tm.get_port())
            tm.set_synchronous_mode(False)
            print(f"Spawned NPC id={npc_vehicle.id}")

    cfg = Hyperparameters()
    regressor = HumanStyleRegressor(buffer_size=args.buffer, config=cfg)
    regressor.attach_collision_sensor(world, ego_vehicle)

    # RGB camera
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(WINDOW_W))
    cam_bp.set_attribute("image_size_y", str(WINDOW_H))
    cam_bp.set_attribute("fov", "100")
    camera = world.spawn_actor(
        cam_bp,
        carla.Transform(carla.Location(x=0.2, y=-0.36, z=1.2),
                        carla.Rotation(pitch=-5.0)),
        attach_to=ego_vehicle,
    )
    latest_image = None
    def on_image(img):
        nonlocal latest_image
        arr = np.frombuffer(img.raw_data, dtype=np.uint8)
        arr = arr.reshape((img.height, img.width, 4))
        latest_image = arr[:, :, :3][:, :, ::-1]
    camera.listen(on_image)

    control      = carla.VehicleControl()
    reverse_mode = False
    clock        = pygame.time.Clock()
    spectator    = world.get_spectator()
    last_print   = time.time()

    print("\n" + "=" * 60)
    print("HUMAN STYLE REGRESSION  (using sb3 AV reward functions)")
    print("=" * 60)
    print(f"AV target speed  : {cfg.target_speed_perc*100:.0f}% of speed limit")
    print(f"Buffer           : {args.buffer} ticks")
    print("Controls         : WASD / wheel to drive, R=reverse, ESC=quit")
    print("=" * 60 + "\n")

    try:
        while True:
            clock.tick(60)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return
                    elif event.key == pygame.K_r:
                        reverse_mode = not reverse_mode
                if event.type == pygame.JOYBUTTONDOWN and event.button == 4:
                    reverse_mode = not reverse_mode

            if wheel is not None:
                control = get_wheel_control(wheel, control, reverse_mode)
            else:
                control = get_keyboard_control(pygame.key.get_pressed(), control, reverse_mode)

            ego_vehicle.apply_control(control)

            # ── Style regression tick ─────────────────────────────────────────
            regressor.tick(ego_vehicle, world)
            profile = regressor.get_style_profile()
            stats   = regressor.get_stats()

            now = time.time()
            if now - last_print >= args.print_interval:
                last_print = now
                print(f"\n[tick={stats['n_ticks']}  buffer={stats['buffer_fill']}  collisions={regressor._collision_cnt}]")
                for label, val in zip(STYLE_LABELS, profile):
                    filled = int(round(float(val) * 20))
                    bar = "[" + "█" * filled + "░" * (20 - filled) + f"] {float(val):.3f}"
                    print(f"  {label:<14} {bar}")

            # ── Spectator follow camera ───────────────────────────────────────
            tf  = ego_vehicle.get_transform()
            loc = tf.location
            yaw = math.radians(tf.rotation.yaw)
            spectator.set_transform(carla.Transform(
                carla.Location(
                    loc.x - 6.0 * math.cos(yaw),
                    loc.y - 6.0 * math.sin(yaw),
                    loc.z + 3.0,
                ),
                carla.Rotation(pitch=-15.0, yaw=tf.rotation.yaw),
            ))

            # ── Render ───────────────────────────────────────────────────────
            if latest_image is not None:
                screen.blit(pygame.surfarray.make_surface(latest_image.swapaxes(0, 1)), (0, 0))
            else:
                screen.fill((30, 30, 30))

            pygame.display.flip()

    finally:
        print("\n" + "=" * 60)
        print("FINAL STYLE PROFILE")
        print("=" * 60)
        profile = regressor.get_style_profile()
        stats   = regressor.get_stats()
        print(f"Total ticks  : {stats['n_ticks']}")
        print(f"Collisions   : {regressor._collision_cnt}")
        for label, val in zip(STYLE_LABELS, profile):
            filled = int(round(float(val) * 20))
            bar = "[" + "█" * filled + "░" * (20 - filled) + f"] {float(val):.3f}"
            print(f"  {label:<14} {bar}")
        print("\nProfile array (pass directly to alert model as state):")
        print(f"  {profile.tolist()}")
        print("=" * 60)

        regressor.destroy_collision_sensor()
        camera.stop()
        camera.destroy()
        if npc_vehicle is not None:
            npc_vehicle.destroy()
        ego_vehicle.destroy()
        pygame.quit()


if __name__ == "__main__":
    main()
