"""
Observation space builders and state extraction for the CARLA RL environment.

Provides hybrid observations combining:
- State vector: normalized numerical features
- Camera: preprocessed RGB images
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import carla

from rl.config import ScenarioConfig, CameraConfig


@dataclass
class VehicleState:
    """Container for extracted vehicle state."""
    
    x: float
    y: float
    z: float
    yaw: float  # radians
    speed: float  # m/s
    velocity_x: float  # m/s
    velocity_y: float  # m/s
    lane_offset: float  # meters from lane center
    lane_id: int
    lane_width: float


@dataclass
class EpisodeState:
    """Container for full episode state including ego, NPC, and derived features."""
    
    ego: VehicleState
    npc: Optional[VehicleState]
    
    # Derived features
    distance_to_goal: float
    progress: float  # 0.0 to 1.0
    
    # Relative NPC features (if NPC exists)
    npc_rel_x: float = 0.0  # positive = NPC ahead
    npc_rel_y: float = 0.0  # positive = NPC to left
    npc_rel_speed: float = 0.0  # positive = NPC faster
    distance_to_npc: float = -1.0  # -1 if no NPC


def extract_vehicle_state(
    vehicle: carla.Vehicle,
    world_map: carla.Map
) -> VehicleState:
    """
    Extract state features from a CARLA vehicle.
    
    Args:
        vehicle: CARLA vehicle actor
        world_map: CARLA map for lane info
        
    Returns:
        VehicleState with all extracted features
    """
    transform = vehicle.get_transform()
    location = transform.location
    rotation = transform.rotation
    velocity = vehicle.get_velocity()
    
    # Speed calculation
    speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    
    # Lane offset calculation
    waypoint = world_map.get_waypoint(
        location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )
    
    lane_yaw = math.radians(waypoint.transform.rotation.yaw)
    normal_x = -math.sin(lane_yaw)
    normal_y = math.cos(lane_yaw)
    
    dx = location.x - waypoint.transform.location.x
    dy = location.y - waypoint.transform.location.y
    lane_offset = dx * normal_x + dy * normal_y
    
    return VehicleState(
        x=location.x,
        y=location.y,
        z=location.z,
        yaw=math.radians(rotation.yaw),
        speed=speed,
        velocity_x=velocity.x,
        velocity_y=velocity.y,
        lane_offset=lane_offset,
        lane_id=waypoint.lane_id,
        lane_width=waypoint.lane_width
    )


def extract_episode_state(
    ego: carla.Vehicle,
    npc: Optional[carla.Vehicle],
    world_map: carla.Map,
    scenario_config: ScenarioConfig
) -> EpisodeState:
    """
    Extract full episode state including ego, NPC, and derived features.
    
    Args:
        ego: Ego vehicle actor
        npc: NPC vehicle actor (can be None)
        world_map: CARLA map
        scenario_config: Scenario configuration
        
    Returns:
        Complete EpisodeState
    """
    ego_state = extract_vehicle_state(ego, world_map)
    
    # Progress and distance to goal
    track_length = scenario_config.goal_x - scenario_config.spawn_x
    distance_traveled = ego_state.x - scenario_config.spawn_x
    progress = np.clip(distance_traveled / track_length, 0.0, 1.0)
    distance_to_goal = max(0.0, scenario_config.goal_x - ego_state.x)
    
    # NPC state and relative features
    npc_state = None
    npc_rel_x = 0.0
    npc_rel_y = 0.0
    npc_rel_speed = 0.0
    distance_to_npc = -1.0
    
    if npc is not None:
        npc_state = extract_vehicle_state(npc, world_map)
        
        # Relative position (in ego's frame)
        npc_rel_x = npc_state.x - ego_state.x
        npc_rel_y = npc_state.y - ego_state.y
        
        # Relative speed
        npc_rel_speed = npc_state.speed - ego_state.speed
        
        # Euclidean distance
        distance_to_npc = math.sqrt(npc_rel_x**2 + npc_rel_y**2)
    
    return EpisodeState(
        ego=ego_state,
        npc=npc_state,
        distance_to_goal=distance_to_goal,
        progress=progress,
        npc_rel_x=npc_rel_x,
        npc_rel_y=npc_rel_y,
        npc_rel_speed=npc_rel_speed,
        distance_to_npc=distance_to_npc
    )


def state_to_vector(
    state: EpisodeState,
    scenario_config: ScenarioConfig
) -> np.ndarray:
    """
    Convert episode state to a normalized feature vector.
    
    The vector contains:
    - Ego position (normalized to track)
    - Ego heading (sin/cos encoded)
    - Ego speed (normalized)
    - Lane offset (normalized)
    - NPC relative position
    - NPC relative speed
    - Progress to goal
    
    Args:
        state: Episode state
        scenario_config: Scenario configuration for normalization
        
    Returns:
        Numpy array of shape (12,) with normalized features
    """
    track_length = scenario_config.goal_x - scenario_config.spawn_x
    half_lane = scenario_config.lane_width / 2.0
    max_speed = 20.0  # m/s (~72 km/h) for normalization
    max_rel_distance = 50.0  # meters
    
    features = np.array([
        # Ego position (normalized 0-1 for x, centered for y)
        (state.ego.x - scenario_config.spawn_x) / track_length,
        (state.ego.y - scenario_config.spawn_y) / (2 * scenario_config.lane_width),
        
        # Ego heading (sin/cos encoding for continuity)
        math.sin(state.ego.yaw),
        math.cos(state.ego.yaw),
        
        # Ego speed (normalized)
        state.ego.speed / max_speed,
        
        # Lane offset (normalized, clipped)
        np.clip(state.ego.lane_offset / half_lane, -2.0, 2.0),
        
        # NPC relative position (normalized)
        np.clip(state.npc_rel_x / max_rel_distance, -1.0, 1.0),
        np.clip(state.npc_rel_y / max_rel_distance, -1.0, 1.0),
        
        # NPC relative speed (normalized)
        np.clip(state.npc_rel_speed / max_speed, -1.0, 1.0),
        
        # Distance to NPC (normalized, -1 if no NPC)
        state.distance_to_npc / max_rel_distance if state.distance_to_npc >= 0 else -1.0,
        
        # Progress to goal
        state.progress,
        
        # Distance to goal (normalized)
        state.distance_to_goal / track_length,
        
    ], dtype=np.float32)
    
    return features


def process_camera_image(
    image: carla.Image,
    target_size: Tuple[int, int] = (84, 84)
) -> np.ndarray:
    """
    Process a CARLA camera image for use as observation.
    
    Args:
        image: CARLA image from camera sensor
        target_size: Target (width, height) for resizing
        
    Returns:
        Numpy array of shape (height, width, 3) with uint8 RGB values
    """
    # Convert CARLA image to numpy array
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    
    # Extract RGB (drop alpha channel) and convert from BGRA to RGB
    rgb = array[:, :, :3][:, :, ::-1].copy()
    
    # Resize if needed
    if (image.width, image.height) != target_size:
        import cv2
        rgb = cv2.resize(rgb, target_size, interpolation=cv2.INTER_AREA)
    
    return rgb


def create_observation_space(
    camera_config: CameraConfig,
    include_camera: bool = True
) -> spaces.Dict:
    """
    Create the Gymnasium observation space.
    
    Args:
        camera_config: Camera configuration for image dimensions
        include_camera: Whether to include camera observations
        
    Returns:
        Gymnasium Dict space with "state" and optionally "camera" keys
    """
    state_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(12,),
        dtype=np.float32
    )
    
    if include_camera:
        camera_space = spaces.Box(
            low=0,
            high=255,
            shape=(camera_config.height, camera_config.width, 3),
            dtype=np.uint8
        )
        return spaces.Dict({
            "state": state_space,
            "camera": camera_space
        })
    else:
        return spaces.Dict({
            "state": state_space
        })


def create_observation(
    state: EpisodeState,
    camera_image: Optional[np.ndarray],
    scenario_config: ScenarioConfig,
    include_camera: bool = True
) -> Dict[str, np.ndarray]:
    """
    Create a full observation dictionary.
    
    Args:
        state: Episode state
        camera_image: Preprocessed camera image (or None)
        scenario_config: Scenario configuration
        include_camera: Whether to include camera in observation
        
    Returns:
        Dictionary with "state" and optionally "camera" keys
    """
    state_vector = state_to_vector(state, scenario_config)
    
    if include_camera and camera_image is not None:
        return {
            "state": state_vector,
            "camera": camera_image
        }
    else:
        return {
            "state": state_vector
        }


class ImageBuffer:
    """
    Thread-safe buffer for camera images from CARLA sensor callbacks.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (84, 84)):
        self.target_size = target_size
        self._latest_image: Optional[np.ndarray] = None
        self._lock = None
        
        # Import threading only when needed
        import threading
        self._lock = threading.Lock()
    
    def on_image(self, image: carla.Image) -> None:
        """Callback for CARLA camera sensor."""
        processed = process_camera_image(image, self.target_size)
        with self._lock:
            self._latest_image = processed
    
    def get_image(self) -> Optional[np.ndarray]:
        """Get the latest processed image."""
        with self._lock:
            if self._latest_image is not None:
                return self._latest_image.copy()
            return None
    
    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self._latest_image = None

