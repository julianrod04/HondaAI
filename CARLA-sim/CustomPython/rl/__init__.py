"""
CARLA Reinforcement Learning Module.

This module provides a Gymnasium-compatible environment for training
autonomous driving agents in the CARLA simulator.

Quick Start:
    from rl import CarlaRLEnv, DEFAULT_CONFIG
    
    env = CarlaRLEnv(config=DEFAULT_CONFIG, instance_id=0)
    obs, info = env.reset()
    
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()

For parallel training:
    from rl import make_vec_env_factory
    from stable_baselines3.common.vec_env import SubprocVecEnv
    
    env = SubprocVecEnv(make_vec_env_factory(num_envs=4))
"""

from rl.config import (
    CarlaConfig,
    ScenarioConfig,
    VehicleConfig,
    CameraConfig,
    RewardConfig,
    TrainingConfig,
    FullConfig,
    DEFAULT_CONFIG,
)
from rl.carla_env import CarlaRLEnv, make_carla_env, make_vec_env_factory
from rl.actions import ContinuousAction, AlertAction, HierarchicalAction
from rl.observations import EpisodeState, VehicleState
from rl.rewards import RewardShaper, RewardInfo

__all__ = [
    # Configuration
    "CarlaConfig",
    "ScenarioConfig",
    "VehicleConfig",
    "CameraConfig",
    "RewardConfig",
    "TrainingConfig",
    "FullConfig",
    "DEFAULT_CONFIG",
    # Environment
    "CarlaRLEnv",
    "make_carla_env",
    "make_vec_env_factory",
    # Actions
    "ContinuousAction",
    "AlertAction",
    "HierarchicalAction",
    # Observations
    "EpisodeState",
    "VehicleState",
    # Rewards
    "RewardShaper",
    "RewardInfo",
]

