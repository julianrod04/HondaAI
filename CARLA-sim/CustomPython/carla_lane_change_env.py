"""
Gymnasium-compatible CARLA lane-change environment for RL training.
The environment wraps the prototype scenario (ego + NPC + RGB camera) and
exposes reset/step/render compatible with PPO/SAC style agents.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import carla
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from carla_display import CarlaDisplay
from carla_respawn import AutoRespawnMonitor


@dataclass(frozen=True)
class StyleConfig:
    """Reward/termination parameters for different driving styles."""
    target_speed: float
    progress_weight: float
    lane_weight: float
    jerk_weight: float
    collision_penalty: float
    success_bonus: float
    timeout_seconds: float
    lane_tolerance: float
    allowed_lane_deviation: float
    min_success_progress: float


DEFAULT_STYLE_CONFIGS: Dict[str, StyleConfig] = {
    "aggressive": StyleConfig(
        target_speed=28.0,
        progress_weight=2.0,
        lane_weight=1.2,
        jerk_weight=0.05,
        collision_penalty=120.0,
        success_bonus=200.0,
        timeout_seconds=14.0,
        lane_tolerance=0.25,
        allowed_lane_deviation=3.0,
        min_success_progress=110.0,
    ),
    "normal": StyleConfig(
        target_speed=22.0,
        progress_weight=1.5,
        lane_weight=1.5,
        jerk_weight=0.1,
        collision_penalty=150.0,
        success_bonus=180.0,
        timeout_seconds=18.0,
        lane_tolerance=0.30,
        allowed_lane_deviation=3.3,
        min_success_progress=105.0,
    ),
    "conservative": StyleConfig(
        target_speed=17.0,
        progress_weight=1.2,
        lane_weight=1.8,
        jerk_weight=0.15,
        collision_penalty=200.0,
        success_bonus=150.0,
        timeout_seconds=22.0,
        lane_tolerance=0.40,
        allowed_lane_deviation=3.5,
        min_success_progress=95.0,
    ),
}


def _style_index(style_name: str) -> float:
    """Map style names to a normalized scalar to feed observation."""
    mapping = {"conservative": 0.0, "normal": 0.5, "aggressive": 1.0}
    return mapping.get(style_name, 0.5)


class CarlaLaneChangeEnv(gym.Env):
    """Lane-change scenario for CARLA with Gymnasium API."""

    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(
        self,
        host: str = "localhost",
        port: int = 2000,
        map_name: str = "Town06",
        style: str = "normal",
        observation_type: str = "state",
        camera_resolution: Tuple[int, int] = (240, 135),
        enable_camera_sensor: bool = True,
        autopilot_npc: bool = True,
        synchronous_mode: bool = False,
        fixed_delta_seconds: float = 0.05,
        max_episode_steps: int = 400,
        display: Optional[CarlaDisplay] = None,
        style_configs: Optional[Dict[str, StyleConfig]] = None,
    ):
        super().__init__()
        self.host = host
        self.port = port
        self.map_name = map_name
        self.style_name = style
        self.style_config = (style_configs or DEFAULT_STYLE_CONFIGS)[style]
        self.autopilot_npc = autopilot_npc
        self.camera_resolution = camera_resolution
        self.synchronous_mode = synchronous_mode
        self.fixed_delta_seconds = fixed_delta_seconds
        self.max_episode_steps = max_episode_steps
        self.observation_type = observation_type
        self.enable_camera_sensor = enable_camera_sensor or observation_type == "rgb"

        self._client = carla.Client(self.host, self.port)
        self._client.set_timeout(5.0)
        if self.map_name:
            self.world = self._client.load_world(self.map_name)
        else:
            self.world = self._client.get_world()
        self._original_settings = self.world.get_settings()
        if self.synchronous_mode:
            settings = carla.WorldSettings(
                no_rendering_mode=self._original_settings.no_rendering_mode,
                synchronous_mode=True,
                fixed_delta_seconds=self.fixed_delta_seconds,
            )
            self.world.apply_settings(settings)

        self.blueprint_library = self.world.get_blueprint_library()
        self.ego_blueprint = self.blueprint_library.find("vehicle.dodge.charger_2020")
        self.ego_blueprint.set_attribute("role_name", "hero")
        self.npc_blueprint = self.blueprint_library.find("vehicle.dodge.charger_police_2020")

        self.camera_blueprint = self.blueprint_library.find("sensor.camera.rgb")
        self.camera_blueprint.set_attribute("image_size_x", str(camera_resolution[0]))
        self.camera_blueprint.set_attribute("image_size_y", str(camera_resolution[1]))
        self.camera_blueprint.set_attribute("fov", "120")

        self.collision_blueprint = self.blueprint_library.find("sensor.other.collision")

        self.display = display
        self._owns_display = display is None

        self.ego_vehicle: Optional[carla.Vehicle] = None
        self.npc_vehicle: Optional[carla.Vehicle] = None
        self.camera: Optional[carla.Sensor] = None
        self.collision_sensor: Optional[carla.Sensor] = None

        self.monitor = AutoRespawnMonitor(
            world=self.world,
            ego_spawn_point=self._default_ego_spawn(),
            npc_spawn_point=self._default_npc_spawn(),
            ego_blueprint=self.ego_blueprint,
            npc_blueprint=self.npc_blueprint,
            camera_blueprint=self.camera_blueprint,
            camera_transform=self._default_camera_transform(),
            display=None,
            camera_callback=self._handle_camera_image if self.enable_camera_sensor else None,
            spawn_x_threshold=1e6,
            time_threshold=1e6,
        )

        self.spawn_origin_x = self.monitor.ego_spawn_point.location.x
        self.start_lane_center_y = self.monitor.ego_spawn_point.location.y
        # Target lane is 1 lane to the left by default (negative y direction in this map)
        self.target_lane_center_y = self.start_lane_center_y - 3.5

        self._latest_image = None
        self._latest_rgb = None
        self._collided = False
        self._episode_elapsed = 0.0
        self.step_count = 0
        self._prev_progress = 0.0
        self._last_action = np.zeros(3, dtype=np.float32)

        self._init_spaces()

    # --- World / spawn helpers -------------------------------------------------
    @staticmethod
    def _default_ego_spawn() -> carla.Transform:
        location = carla.Location(x=21.0, y=244.485397, z=0.5)
        rotation = carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
        return carla.Transform(location, rotation)

    @staticmethod
    def _default_npc_spawn() -> carla.Transform:
        location = carla.Location(x=21.0, y=244.485397 - 3.5, z=0.5)
        rotation = carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
        return carla.Transform(location, rotation)

    @staticmethod
    def _default_camera_transform() -> carla.Transform:
        return carla.Transform(
            carla.Location(x=0.2, y=-0.36, z=1.2),
            carla.Rotation(pitch=-10.0, yaw=0.0, roll=0.0),
        )

    def _init_spaces(self):
        """Configure Gymnasium action/observation spaces."""
        # steer, throttle, brake
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        if self.observation_type == "state":
            low = np.array(
                [0.0, -6.0, -2.0, -200.0, -10.0, 0.0, 0.0, 0.0, 0.0],
                dtype=np.float32,
            )
            high = np.array(
                [50.0, 6.0, 2.0, 200.0, 10.0, 50.0, 200.0, self.style_config.timeout_seconds, 1.0],
                dtype=np.float32,
            )
            self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        elif self.observation_type == "rgb":
            width, height = self.camera_resolution
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(height, width, 3),
                dtype=np.uint8,
            )
        else:
            raise ValueError(f"Unsupported observation_type={self.observation_type}")

    # --- Gymnasium API ---------------------------------------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._destroy_episode_actors()

        self._collided = False
        self._latest_image = None
        self._latest_rgb = None
        self._episode_elapsed = 0.0
        self.step_count = 0
        self._prev_progress = 0.0
        self._last_action[:] = 0.0

        ego, npc, camera = self.monitor.force_respawn(
            ego_autopilot=False, npc_autopilot=self.autopilot_npc
        )
        self.ego_vehicle = ego
        self.npc_vehicle = npc
        if self.enable_camera_sensor:
            self.camera = camera
        else:
            if camera:
                camera.stop()
                camera.destroy()
            self.camera = None
            self.monitor.camera = None

        if self.enable_camera_sensor:
            self._wait_for_camera_frame()

        self._attach_collision_sensor()
        self._tick_world()

        obs = self._build_observation()
        info = {"style": self.style_name, "event": "reset"}
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        control = carla.VehicleControl(
            throttle=float(action[1]),
            brake=float(action[2]),
            steer=float(action[0]),
            hand_brake=False,
            reverse=False,
        )
        if not self.ego_vehicle:
            raise RuntimeError("Environment not reset before step.")
        self.ego_vehicle.apply_control(control)

        snapshot = self._tick_world()
        dt = snapshot.delta_seconds if snapshot else self.fixed_delta_seconds
        self._episode_elapsed += dt
        self.step_count += 1

        obs = self._build_observation()
        reward, terminated, truncated, info = self._compute_reward_and_done(obs, action, dt)
        self._last_action = action.astype(np.float32)
        return obs, reward, terminated, truncated, info

    def render(self, mode: str = "human"):
        if mode != "human":
            raise NotImplementedError("Only human rendering is supported.")
        if not self.enable_camera_sensor:
            raise RuntimeError("Camera sensor is disabled for this environment.")
        self._ensure_display()
        # Images are pushed to the display asynchronously inside _handle_camera_image
        if self._latest_rgb is None:
            return None
        return self._latest_rgb.copy()

    def close(self):
        self._destroy_episode_actors()
        if self.monitor:
            self.monitor.stop()
        if self.display and self._owns_display:
            self.display.stop()
        if self.synchronous_mode and self._original_settings:
            self.world.apply_settings(self._original_settings)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # --- Reward / observation helpers -----------------------------------------
    def _build_observation(self):
        if self.observation_type == "rgb":
            if self._latest_rgb is None:
                self._wait_for_camera_frame()
            if self._latest_rgb is None:
                return np.zeros(self.observation_space.shape, dtype=np.uint8)
            return self._latest_rgb.copy()

        assert self.ego_vehicle is not None
        ego_tf = self.ego_vehicle.get_transform()
        ego_location = ego_tf.location
        ego_velocity = self.ego_vehicle.get_velocity()
        ego_speed = math.sqrt(
            ego_velocity.x ** 2 + ego_velocity.y ** 2 + ego_velocity.z ** 2
        )
        ego_ang_vel = self.ego_vehicle.get_angular_velocity()

        npc_rel_x, npc_rel_y, npc_speed = 0.0, 0.0, 0.0
        if self.npc_vehicle:
            npc_location = self.npc_vehicle.get_transform().location
            npc_rel_x = npc_location.x - ego_location.x
            npc_rel_y = npc_location.y - ego_location.y
            npc_vel = self.npc_vehicle.get_velocity()
            npc_speed = math.sqrt(npc_vel.x ** 2 + npc_vel.y ** 2 + npc_vel.z ** 2)

        lane_error = ego_location.y - self.target_lane_center_y
        progress = ego_location.x - self.spawn_origin_x

        obs = np.array(
            [
                np.clip(ego_speed, 0.0, 60.0),
                np.clip(lane_error, -6.0, 6.0),
                np.clip(ego_ang_vel.z, -2.0, 2.0),
                np.clip(npc_rel_x, -200.0, 200.0),
                np.clip(npc_rel_y, -10.0, 10.0),
                np.clip(npc_speed, 0.0, 50.0),
                np.clip(progress, 0.0, 220.0),
                np.clip(self._episode_elapsed, 0.0, self.style_config.timeout_seconds),
                _style_index(self.style_name),
            ],
            dtype=np.float32,
        )
        return obs

    def _compute_reward_and_done(self, obs, action, dt):
        style = self.style_config
        lane_error = float(obs[1])
        ego_speed = float(obs[0])
        progress = float(obs[6])

        progress_delta = progress - self._prev_progress
        self._prev_progress = progress

        lane_penalty = -style.lane_weight * abs(lane_error)
        progress_reward = style.progress_weight * progress_delta
        speed_penalty = -0.15 * abs(ego_speed - style.target_speed)
        jerk_penalty = -style.jerk_weight * float(np.mean(np.square(action - self._last_action)))

        reward = progress_reward + lane_penalty + speed_penalty + jerk_penalty

        terminated = False
        truncated = False
        event = None

        if self._collided:
            reward -= style.collision_penalty
            terminated = True
            event = "collision"
        elif abs(lane_error) > style.allowed_lane_deviation:
            reward -= 50.0
            terminated = True
            event = "off-lane"

        success = (
            not self._collided
            and abs(lane_error) <= style.lane_tolerance
            and progress >= style.min_success_progress
        )
        if success:
            reward += style.success_bonus
            terminated = True
            event = "success"

        if self.step_count >= self.max_episode_steps:
            truncated = True
            event = event or "step-limit"
        elif self._episode_elapsed >= style.timeout_seconds:
            truncated = True
            event = event or "timeout"

        info = {
            "style": self.style_name,
            "event": event or "running",
            "progress": progress,
            "lane_error": lane_error,
            "speed_mps": ego_speed,
            "elapsed": self._episode_elapsed,
        }
        return reward, terminated, truncated, info

    # --- Sensors / callbacks ---------------------------------------------------
    def _handle_camera_image(self, image: carla.Image):
        self._latest_image = image
        # Convert to numpy for RL observations / logging
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        self._latest_rgb = array[:, :, :3][:, :, ::-1]
        if self.display and self.display.running:
            self.display.on_image(image)

    def _wait_for_camera_frame(self, timeout: float = 2.0):
        if not self.enable_camera_sensor:
            return
        start = time.time()
        while self._latest_rgb is None and (time.time() - start) < timeout:
            time.sleep(0.01)

    def _attach_collision_sensor(self):
        self._destroy_collision_sensor()
        if not self.ego_vehicle:
            return
        self.collision_sensor = self.world.spawn_actor(
            self.collision_blueprint, carla.Transform(), attach_to=self.ego_vehicle
        )
        self.collision_sensor.listen(self._on_collision)

    def _on_collision(self, event: carla.CollisionEvent):
        self._collided = True

    # --- Actor / world cleanup -------------------------------------------------
    def _destroy_episode_actors(self):
        if self.camera:
            self.camera.stop()
            self.camera.destroy()
            self.camera = None
            self.monitor.camera = None
        self._destroy_collision_sensor()
        if self.ego_vehicle:
            self.ego_vehicle.destroy()
            self.ego_vehicle = None
            self.monitor.ego_vehicle = None
        if self.npc_vehicle:
            self.npc_vehicle.destroy()
            self.npc_vehicle = None
            self.monitor.npc_vehicle = None

    def _destroy_collision_sensor(self):
        if self.collision_sensor:
            self.collision_sensor.stop()
            self.collision_sensor.destroy()
            self.collision_sensor = None

    def _tick_world(self):
        if self.synchronous_mode:
            self.world.tick()
            return self.world.get_snapshot()
        return self.world.wait_for_tick()

    def _ensure_display(self):
        if not self.display:
            self.display = CarlaDisplay(
                width=self.camera_resolution[0], height=self.camera_resolution[1]
            )
            self._owns_display = True
        if not self.display.running:
            self.display.start()
