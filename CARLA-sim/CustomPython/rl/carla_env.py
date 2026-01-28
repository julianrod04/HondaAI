"""
Gymnasium-compatible CARLA Reinforcement Learning Environment.

This environment wraps CARLA for training autonomous driving agents.
It supports:
- Hybrid observations (state vector + camera)
- Continuous action space (throttle/brake + steering)
- Vectorized parallel training across multiple CARLA instances
"""

import math
import time
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import carla

from rl.config import (
    CarlaConfig,
    ScenarioConfig,
    VehicleConfig,
    CameraConfig,
    RewardConfig,
    FullConfig,
    DEFAULT_CONFIG,
)
from rl.utils import (
    connect_to_carla,
    configure_world_settings,
    restore_world_settings,
    create_spawn_transform,
    spawn_ego_vehicle,
    spawn_npc_vehicle,
    spawn_camera_sensor,
    spawn_collision_sensor,
    destroy_actors,
    cleanup_world,
    get_vehicle_speed,
    teleport_vehicle,
)
from rl.observations import (
    extract_episode_state,
    create_observation,
    create_observation_space,
    ImageBuffer,
    EpisodeState,
)
from rl.rewards import (
    RewardShaper,
    get_termination_conditions,
)


class CarlaRLEnv(gym.Env):
    """
    Gymnasium environment for CARLA autonomous driving.
    
    Observation Space (Dict):
        - "state": Box(12,) normalized state vector
        - "camera": Box(84, 84, 3) RGB image (if include_camera=True)
    
    Action Space: Box(2,) continuous
        - action[0]: throttle/brake (-1 to 1)
          - Positive values: throttle (0 to 1 mapped to 0 to 1)
          - Negative values: brake (-1 to 0 mapped to 0 to 1)
        - action[1]: steering (-1 to 1)
          - Negative: left, Positive: right
    
    Args:
        config: Full configuration (uses DEFAULT_CONFIG if None)
        instance_id: CARLA server instance ID (for port selection)
        include_camera: Whether to include camera observations
        include_npc: Whether to spawn an NPC vehicle
        render_mode: Gymnasium render mode ("human" or None)
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 20}
    
    def __init__(
        self,
        config: Optional[FullConfig] = None,
        instance_id: int = 0,
        include_camera: bool = True,
        include_npc: bool = True,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        
        self.config = config or DEFAULT_CONFIG
        self.instance_id = instance_id
        self.include_camera = include_camera
        self.include_npc = include_npc
        self.render_mode = render_mode
        
        # CARLA objects (initialized in reset)
        self._client: Optional[carla.Client] = None
        self._world: Optional[carla.World] = None
        self._world_map: Optional[carla.Map] = None
        self._original_settings: Optional[carla.WorldSettings] = None
        
        # Actors
        self._ego: Optional[carla.Vehicle] = None
        self._npc: Optional[carla.Vehicle] = None
        self._camera: Optional[carla.Sensor] = None
        self._collision_sensor: Optional[carla.Sensor] = None
        
        # State tracking
        self._image_buffer: Optional[ImageBuffer] = None
        self._collision_occurred: bool = False
        self._step_count: int = 0
        self._prev_state: Optional[EpisodeState] = None
        
        # Reward shaper
        self._reward_shaper = RewardShaper(
            self.config.reward,
            self.config.scenario,
            use_potential_shaping=False
        )
        
        # Define action space: [throttle_brake, steering]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Define observation space
        self.observation_space = create_observation_space(
            self.config.camera,
            include_camera=self.include_camera
        )
        
        # Track if connected
        self._connected = False
    
    def _connect(self) -> None:
        """Connect to CARLA server and configure world."""
        if self._connected:
            return
        
        self._client, self._world = connect_to_carla(
            self.config.carla,
            instance_id=self.instance_id,
            load_map=self.config.scenario.map_name
        )
        
        self._world_map = self._world.get_map()
        self._original_settings = configure_world_settings(
            self._world,
            self.config.carla
        )
        
        # Initialize image buffer if using camera
        if self.include_camera:
            self._image_buffer = ImageBuffer(
                target_size=(self.config.camera.width, self.config.camera.height)
            )
        
        self._connected = True
    
    def _spawn_actors(self) -> bool:
        """
        Spawn all actors (ego, NPC, sensors).
        
        Returns:
            True if all required actors spawned successfully
        """
        # Spawn ego vehicle
        self._ego = spawn_ego_vehicle(
            self._world,
            self.config.vehicle,
            self.config.scenario
        )
        
        if self._ego is None:
            return False
        
        # Spawn NPC if requested
        if self.include_npc:
            self._npc = spawn_npc_vehicle(
                self._world,
                self.config.vehicle,
                self.config.scenario,
                self._client
            )
        
        # Spawn camera sensor
        if self.include_camera:
            self._camera = spawn_camera_sensor(
                self._world,
                self._ego,
                self.config.camera
            )
            self._camera.listen(self._image_buffer.on_image)
        
        # Spawn collision sensor
        self._collision_sensor = spawn_collision_sensor(self._world, self._ego)
        self._collision_sensor.listen(self._on_collision)
        
        return True
    
    def _destroy_actors(self) -> None:
        """Destroy all spawned actors."""
        actors_to_destroy = []
        
        if self._collision_sensor:
            actors_to_destroy.append(self._collision_sensor)
        if self._camera:
            actors_to_destroy.append(self._camera)
        if self._npc:
            actors_to_destroy.append(self._npc)
        if self._ego:
            actors_to_destroy.append(self._ego)
        
        destroy_actors(actors_to_destroy)
        
        self._collision_sensor = None
        self._camera = None
        self._npc = None
        self._ego = None
    
    def _on_collision(self, event: carla.CollisionEvent) -> None:
        """Callback for collision sensor."""
        self._collision_occurred = True
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation."""
        state = extract_episode_state(
            self._ego,
            self._npc,
            self._world_map,
            self.config.scenario
        )
        
        camera_image = None
        if self.include_camera and self._image_buffer:
            camera_image = self._image_buffer.get_image()
            
            # If no image yet, create a blank one
            if camera_image is None:
                camera_image = np.zeros(
                    (self.config.camera.height, self.config.camera.width, 3),
                    dtype=np.uint8
                )
        
        self._prev_state = state
        
        return create_observation(
            state,
            camera_image,
            self.config.scenario,
            include_camera=self.include_camera
        )
    
    def _apply_action(self, action: np.ndarray) -> None:
        """
        Apply action to the ego vehicle.
        
        Args:
            action: [throttle_brake, steering] in range [-1, 1]
        """
        throttle_brake = float(action[0])
        steering = float(action[1])
        
        control = carla.VehicleControl()
        
        # Convert combined throttle/brake to separate values
        if throttle_brake >= 0:
            control.throttle = throttle_brake
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = -throttle_brake
        
        # Steering
        control.steer = np.clip(steering, -1.0, 1.0)
        
        # Apply control
        self._ego.apply_control(control)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset the environment for a new episode.
        
        Args:
            seed: Random seed (not used currently)
            options: Additional options (not used currently)
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        # Connect to CARLA if not already connected
        self._connect()
        
        # Destroy existing actors
        self._destroy_actors()
        
        # Reset state
        self._collision_occurred = False
        self._step_count = 0
        self._prev_state = None
        self._reward_shaper.reset()
        
        if self._image_buffer:
            self._image_buffer.clear()
        
        # Spawn new actors
        if not self._spawn_actors():
            raise RuntimeError(
                f"Failed to spawn actors in CARLA instance {self.instance_id}"
            )
        
        # Let the simulation settle
        for _ in range(5):
            self._world.tick()
        
        # Get initial observation
        observation = self._get_observation()
        
        info = {
            "instance_id": self.instance_id,
            "spawn_x": self.config.scenario.spawn_x,
            "goal_x": self.config.scenario.goal_x,
        }
        
        return observation, info
    
    def step(
        self,
        action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step.
        
        Args:
            action: [throttle_brake, steering] in range [-1, 1]
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Apply action
        self._apply_action(action)
        
        # Advance simulation
        self._world.tick()
        
        # Increment step counter
        self._step_count += 1
        
        # Get new state
        curr_state = extract_episode_state(
            self._ego,
            self._npc,
            self._world_map,
            self.config.scenario
        )
        
        # Check termination conditions
        terminated, truncated, termination_reason = get_termination_conditions(
            curr_state,
            self._collision_occurred,
            self._step_count,
            self.config.scenario
        )
        
        goal_reached = termination_reason == "goal_reached"
        
        # Compute reward
        reward, reward_info = self._reward_shaper.compute(
            curr_state,
            action,
            self._collision_occurred,
            goal_reached
        )
        
        # Get observation
        camera_image = None
        if self.include_camera and self._image_buffer:
            camera_image = self._image_buffer.get_image()
            if camera_image is None:
                camera_image = np.zeros(
                    (self.config.camera.height, self.config.camera.width, 3),
                    dtype=np.uint8
                )
        
        observation = create_observation(
            curr_state,
            camera_image,
            self.config.scenario,
            include_camera=self.include_camera
        )
        
        # Update previous state
        self._prev_state = curr_state
        
        # Build info dict
        info = {
            "step": self._step_count,
            "ego_x": curr_state.ego.x,
            "ego_speed": curr_state.ego.speed,
            "lane_offset": curr_state.ego.lane_offset,
            "progress": curr_state.progress,
            "collision": self._collision_occurred,
            "termination_reason": termination_reason,
            "reward_breakdown": {
                "total": reward_info.total,
                "progress": reward_info.progress,
                "lane_keeping": reward_info.lane_keeping,
                "speed": reward_info.speed,
                "collision": reward_info.collision,
                "goal": reward_info.goal,
            },
        }
        
        if self.include_npc and curr_state.npc:
            info["npc_x"] = curr_state.npc.x
            info["npc_speed"] = curr_state.npc.speed
            info["distance_to_npc"] = curr_state.distance_to_npc
        
        return observation, reward, terminated, truncated, info
    
    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Returns:
            RGB array if render_mode is "rgb_array", None otherwise
        """
        if self.render_mode == "human":
            # For human rendering, the CARLA server window is the display
            pass
        elif self.render_mode == "rgb_array":
            if self._image_buffer:
                return self._image_buffer.get_image()
        return None
    
    def close(self) -> None:
        """Clean up resources."""
        self._destroy_actors()
        
        if self._original_settings and self._world:
            restore_world_settings(self._world, self._original_settings)
        
        self._connected = False
        self._client = None
        self._world = None
        self._world_map = None


def make_carla_env(
    instance_id: int = 0,
    config: Optional[FullConfig] = None,
    include_camera: bool = True,
    include_npc: bool = True
) -> CarlaRLEnv:
    """
    Factory function to create a CARLA RL environment.
    
    Args:
        instance_id: CARLA server instance ID
        config: Configuration (uses default if None)
        include_camera: Include camera observations
        include_npc: Include NPC vehicle
        
    Returns:
        Configured CarlaRLEnv instance
    """
    return CarlaRLEnv(
        config=config,
        instance_id=instance_id,
        include_camera=include_camera,
        include_npc=include_npc
    )


def make_vec_env_factory(
    num_envs: int = 4,
    config: Optional[FullConfig] = None,
    include_camera: bool = True,
    include_npc: bool = True
) -> List:
    """
    Create a list of environment factory functions for vectorized training.
    
    Usage with Stable-Baselines3:
        env = SubprocVecEnv(make_vec_env_factory(4))
    
    Args:
        num_envs: Number of parallel environments
        config: Configuration (uses default if None)
        include_camera: Include camera observations
        include_npc: Include NPC vehicle
        
    Returns:
        List of factory functions, one per environment
    """
    def _make_env(instance_id: int):
        def _init():
            return make_carla_env(
                instance_id=instance_id,
                config=config,
                include_camera=include_camera,
                include_npc=include_npc
            )
        return _init
    
    return [_make_env(i) for i in range(num_envs)]

