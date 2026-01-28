"""
Action space definitions for the CARLA RL environment.

Current implementation: Continuous control
Future extension: High-level alert-based actions for hierarchical control
"""

from enum import IntEnum
from typing import Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import carla


class ContinuousAction:
    """
    Continuous action space representation.
    
    Action vector: [throttle_brake, steering]
    - throttle_brake: -1 (full brake) to +1 (full throttle)
    - steering: -1 (full left) to +1 (full right)
    """
    
    @staticmethod
    def get_space() -> spaces.Box:
        """Get the Gymnasium action space."""
        return spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
    
    @staticmethod
    def to_vehicle_control(action: np.ndarray) -> carla.VehicleControl:
        """
        Convert action array to CARLA VehicleControl.
        
        Args:
            action: [throttle_brake, steering] array
            
        Returns:
            CARLA VehicleControl object
        """
        throttle_brake = float(action[0])
        steering = float(action[1])
        
        control = carla.VehicleControl()
        
        # Split throttle/brake
        if throttle_brake >= 0:
            control.throttle = throttle_brake
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = -throttle_brake
        
        # Steering
        control.steer = np.clip(steering, -1.0, 1.0)
        
        # Defaults
        control.hand_brake = False
        control.reverse = False
        
        return control
    
    @staticmethod
    def from_vehicle_control(control: carla.VehicleControl) -> np.ndarray:
        """
        Convert CARLA VehicleControl to action array.
        
        Args:
            control: CARLA VehicleControl object
            
        Returns:
            [throttle_brake, steering] array
        """
        if control.brake > 0:
            throttle_brake = -control.brake
        else:
            throttle_brake = control.throttle
        
        return np.array([throttle_brake, control.steer], dtype=np.float32)


class AlertAction(IntEnum):
    """
    High-level alert-based actions for future hierarchical control.
    
    These represent the external commands the car should respond to:
    - MAINTAIN: Keep current behavior (lane, speed)
    - LANE_CHANGE_LEFT/RIGHT: Execute lane change maneuver
    - SPEED_UP/SLOW_DOWN: Adjust speed
    - EMERGENCY_STOP: Immediate stopping
    - OVERTAKE: Pass the vehicle ahead
    """
    
    MAINTAIN = 0
    LANE_CHANGE_LEFT = 1
    LANE_CHANGE_RIGHT = 2
    SPEED_UP = 3
    SLOW_DOWN = 4
    EMERGENCY_STOP = 5
    OVERTAKE = 6


class HierarchicalAction:
    """
    Hierarchical action space for alert-based control.
    
    High level: Discrete alert selection
    Low level: Continuous control parameters
    
    This is for Phase 2 implementation.
    """
    
    @staticmethod
    def get_space() -> spaces.Dict:
        """Get the Gymnasium action space."""
        return spaces.Dict({
            "alert": spaces.Discrete(len(AlertAction)),
            "control": spaces.Box(
                low=np.array([-1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                dtype=np.float32
            )
        })
    
    @staticmethod
    def get_alert_name(alert_id: int) -> str:
        """Get human-readable name for an alert."""
        return AlertAction(alert_id).name


class ActionScaler:
    """
    Utility for scaling and clipping actions.
    """
    
    def __init__(
        self,
        max_throttle: float = 1.0,
        max_brake: float = 1.0,
        max_steer: float = 1.0,
    ):
        self.max_throttle = max_throttle
        self.max_brake = max_brake
        self.max_steer = max_steer
    
    def scale(self, action: np.ndarray) -> np.ndarray:
        """
        Scale action from [-1, 1] to actual limits.
        
        Args:
            action: Raw action in [-1, 1]
            
        Returns:
            Scaled action
        """
        throttle_brake = action[0]
        steering = action[1]
        
        if throttle_brake >= 0:
            throttle_brake *= self.max_throttle
        else:
            throttle_brake *= self.max_brake
        
        steering *= self.max_steer
        
        return np.array([throttle_brake, steering], dtype=np.float32)


def compute_action_smoothness(
    current_action: np.ndarray,
    previous_action: np.ndarray
) -> float:
    """
    Compute action smoothness penalty.
    
    Large changes in action indicate jerky control.
    
    Args:
        current_action: Current action
        previous_action: Previous action
        
    Returns:
        Smoothness penalty (lower is smoother)
    """
    diff = np.abs(current_action - previous_action)
    return float(np.sum(diff))

