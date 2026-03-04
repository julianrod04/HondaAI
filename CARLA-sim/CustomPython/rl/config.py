"""
Centralized configuration for the CARLA RL environment.

All hardcoded values are consolidated here for maintainability.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class CarlaConfig:
    """CARLA server connection settings."""
    
    host: str = "localhost"
    base_port: int = 2000
    timeout: float = 30.0  # Increased timeout for slow connections
    sync_mode: bool = True
    fixed_delta: float = 0.05  # 20 FPS simulation (0.05s per tick)
    
    def get_port(self, instance_id: int = 0) -> int:
        """Get port for a specific CARLA instance (ports increment by 2)."""
        return self.base_port + (instance_id * 2)


@dataclass
class ScenarioConfig:
    """Scenario/track configuration for the highway driving task."""
    
    map_name: str = "Town06"
    
    # Spawn location (start of the 200m straight section)
    spawn_x: float = 21.0
    spawn_y: float = 244.485397
    spawn_z: float = 0.5
    spawn_yaw: float = 0.0  # Facing east (+X direction)
    
    # Goal location (end of track)
    goal_x: float = 221.0  # 200m from spawn
    
    # Lane geometry
    lane_width: float = 3.5  # meters
    
    # Episode limits
    max_episode_steps: int = 600  # 30 seconds at 20 FPS
    max_episode_time: float = 30.0  # seconds
    
    # NPC offset (adjacent lane)
    npc_lane_offset: float = -3.5  # One lane to the right (negative Y)
    npc_x_offset: float = 0.0  # Same X as ego initially
    
    # Target speed for reward shaping
    target_speed_mps: float = 8.33  # ~30 km/h
    target_speed_kmh: float = 30.0


@dataclass
class VehicleConfig:
    """Vehicle blueprint configuration."""
    
    ego_blueprint: str = "vehicle.dodge.charger_2020"
    ego_role_name: str = "hero"
    
    npc_blueprint: str = "vehicle.dodge.charger_police_2020"
    npc_role_name: str = "npc"
    
    # NPC autopilot settings
    npc_autopilot: bool = True
    npc_target_speed_kmh: float = 30.0


@dataclass
class CameraConfig:
    """Camera sensor configuration for visual observations."""
    
    # Image dimensions (84x84 is standard for RL vision)
    width: int = 84
    height: int = 84
    fov: int = 90  # Field of view in degrees
    
    # Camera mounting position relative to vehicle
    # Driver POV: positioned at driver's head, looking forward
    location_x: float = 0.2   # Forward/back from vehicle center
    location_y: float = -0.36  # Left/right (negative = driver side)
    location_z: float = 1.2   # Height (head level)
    
    # Camera rotation
    pitch: float = -10.0  # Slight downward tilt for natural driving gaze
    yaw: float = 0.0
    roll: float = 0.0


@dataclass
class RewardConfig:
    """Reward function weights and parameters."""
    
    # Progress reward (per meter forward)
    progress_weight: float = 0.1
    
    # Lane keeping penalty (per meter offset from center)
    lane_deviation_weight: float = 0.5
    
    # Speed maintenance penalty (per m/s deviation from target)
    speed_deviation_weight: float = 0.1
    
    # Collision penalty (terminal)
    collision_penalty: float = 100.0
    
    # Goal reached bonus (terminal)
    goal_bonus: float = 50.0
    
    # Smoothness penalty (for jerky steering/throttle)
    action_smoothness_weight: float = 0.01


@dataclass 
class TrainingConfig:
    """Training hyperparameters."""
    
    # Number of parallel environments
    num_envs: int = 4
    
    # Total training steps
    total_timesteps: int = 1_000_000
    
    # PPO-specific
    learning_rate: float = 3e-4
    n_steps: int = 2048  # Steps per environment before update
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda
    clip_range: float = 0.2
    
    # Logging
    tensorboard_log: str = "./tb_logs/"
    save_freq: int = 10_000  # Save model every N steps
    model_save_path: str = "./models/"


@dataclass
class AlertDisplayConfig:
    """Global alert overlay appearance and layout configuration.

    Two independent panels:
      FOV panel        – large center overlay (navigation / critical alerts)
      Dashboard panel  – compact banner above the HUD bar (speed / lane alerts)
    """

    # ---- FOV panel (field-of-vision, Panel 1) --------------------------------
    width: int = 450                                    # panel width in pixels
    height: int = 80                                    # panel height in pixels
    position: str = "top-center"                            # "center", "top-center", "top-left",
                                                        # "top-right", "bottom-left", "bottom-right"
    custom_x: Optional[int] = None                      # pixel override (None = use position string)
    custom_y: Optional[int] = None
    text_color: Tuple[int, int, int] = (255, 255, 255)
    bg_color_override: Optional[Tuple[int, int, int]] = None  # None = use per-alert color
    alpha: int = 100                                    # 0-255 background transparency

    # ---- Dashboard alert panel (Panel 2) -------------------------------------
    dashboard_position: str = "bottom-right"            # "bottom-left", "bottom-center", "bottom-right"
    dashboard_alpha: int = 200                          # 0-255 background transparency
    dashboard_text_color: Optional[Tuple[int, int, int]] = None       # None = use per-alert color
    dashboard_bg_color_override: Optional[Tuple[int, int, int]] = None  # None = use per-alert color
    dashboard_padding_x: int = 24                       # horizontal padding around text (px)
    dashboard_padding_y: int = 12                       # vertical padding around text (px)

    # ---- General -------------------------------------------------------------
    show_diagnostics_bar: bool = True                   # toggle bottom HUD bar (Tab key)


@dataclass
class FullConfig:
    """Complete configuration combining all sub-configs."""

    carla: CarlaConfig = field(default_factory=CarlaConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    alert_display: AlertDisplayConfig = field(default_factory=AlertDisplayConfig)


# Default configuration instance
DEFAULT_CONFIG = FullConfig()

