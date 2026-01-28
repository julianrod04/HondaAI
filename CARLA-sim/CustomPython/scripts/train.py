#!/usr/bin/env python3
"""
Training script for CARLA RL autonomous driving.

This script trains a PPO agent using Stable-Baselines3 with:
- Vectorized parallel environments (4 CARLA instances by default)
- Hybrid observations (state vector + camera)
- TensorBoard logging
- Model checkpointing

Prerequisites:
    1. Start CARLA servers: python scripts/launch_servers.py -n 4
    2. Install dependencies: pip install stable-baselines3[extra] tensorboard

Usage:
    python scripts/train.py
    python scripts/train.py --num-envs 4 --timesteps 1000000
    python scripts/train.py --resume models/ppo_carla_latest.zip
    python scripts/train.py --light  # Lightweight mode (no camera, smaller batches)
"""

import argparse
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import psutil

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor

from rl.config import FullConfig, DEFAULT_CONFIG, TrainingConfig
from rl.carla_env import CarlaRLEnv


class ResourceMonitorCallback(BaseCallback):
    """
    Callback for monitoring system resources and logging to TensorBoard.
    """
    
    def __init__(self, log_freq: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._has_gpu = False
        
        # Try to import GPU monitoring
        try:
            import torch
            self._has_gpu = torch.cuda.is_available()
        except ImportError:
            pass
    
    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            memory_used_gb = memory.used / (1024 ** 3)
            memory_percent = memory.percent
            
            self.logger.record("resources/cpu_percent", cpu_percent)
            self.logger.record("resources/memory_used_gb", memory_used_gb)
            self.logger.record("resources/memory_percent", memory_percent)
            
            # GPU if available
            if self._has_gpu:
                try:
                    import torch
                    gpu_memory = torch.cuda.memory_allocated() / (1024 ** 3)
                    self.logger.record("resources/gpu_memory_gb", gpu_memory)
                except Exception:
                    pass
            
            # Log to console periodically
            if self.n_calls % 1000 == 0:
                print(f"  [Resources] CPU: {cpu_percent:.1f}%, RAM: {memory_used_gb:.1f}GB ({memory_percent:.1f}%)")
        
        return True


class TensorBoardCallback(BaseCallback):
    """
    Custom callback for logging additional metrics to TensorBoard.
    """
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._episode_rewards = []
        self._episode_lengths = []
        self._episode_progresses = []
    
    def _on_step(self) -> bool:
        # Log episode statistics when episodes complete
        if self.locals.get("dones") is not None:
            for i, done in enumerate(self.locals["dones"]):
                if done:
                    info = self.locals["infos"][i]
                    
                    # Log progress if available
                    if "progress" in info:
                        self._episode_progresses.append(info["progress"])
                    
                    # Log termination reason
                    if "termination_reason" in info:
                        reason = info["termination_reason"]
                        # Convert to numeric for logging
                        reason_map = {
                            "goal_reached": 1.0,
                            "collision": -1.0,
                            "timeout": 0.0,
                            "off_road": -0.5,
                            "reversed": -0.5,
                        }
                        self.logger.record(
                            "episode/termination_value",
                            reason_map.get(reason, 0.0)
                        )
        
        # Log average progress every 1000 steps
        if self.n_calls % 1000 == 0 and self._episode_progresses:
            self.logger.record(
                "episode/mean_progress",
                np.mean(self._episode_progresses[-100:])
            )
        
        return True


def make_env(
    instance_id: int,
    config: FullConfig,
    include_camera: bool = True,
    include_npc: bool = True,
    seed: int = 0,
    use_monitor: bool = True,
) -> Callable[[], CarlaRLEnv]:
    """
    Create an environment factory function.
    
    Args:
        instance_id: CARLA server instance ID
        config: Environment configuration
        include_camera: Include camera observations
        include_npc: Include NPC vehicle
        seed: Random seed
        use_monitor: Whether to wrap with Monitor (disable for DummyVecEnv)
        
    Returns:
        Factory function that creates the environment
    """
    def _init() -> CarlaRLEnv:
        env = CarlaRLEnv(
            config=config,
            instance_id=instance_id,
            include_camera=include_camera,
            include_npc=include_npc,
        )
        if use_monitor:
            env = Monitor(env)  # Wrap with Monitor for logging
        return env
    
    return _init


def create_vec_env(
    num_envs: int,
    config: FullConfig,
    include_camera: bool = True,
    include_npc: bool = True,
    seed: int = 0,
):
    """
    Create a vectorized environment with multiple CARLA instances.
    
    Args:
        num_envs: Number of parallel environments
        config: Environment configuration
        include_camera: Include camera observations
        include_npc: Include NPC vehicle
        seed: Base random seed
        
    Returns:
        Vectorized environment
    """
    # Use DummyVecEnv for single environment (more stable on Windows)
    # Use SubprocVecEnv for multiple environments (true parallelism)
    if num_envs == 1:
        # For DummyVecEnv, don't use Monitor wrapper (VecMonitor will handle it)
        env_fns = [
            make_env(
                instance_id=0,
                config=config,
                include_camera=include_camera,
                include_npc=include_npc,
                seed=seed,
                use_monitor=False,  # Don't double-wrap
            )
        ]
        vec_env = DummyVecEnv(env_fns)
    else:
        # For SubprocVecEnv, use Monitor in each subprocess
        env_fns = [
            make_env(
                instance_id=i,
                config=config,
                include_camera=include_camera,
                include_npc=include_npc,
                seed=seed + i,
                use_monitor=True,
            )
            for i in range(num_envs)
        ]
        vec_env = SubprocVecEnv(env_fns, start_method="spawn")
    
    # Wrap with VecMonitor for logging
    vec_env = VecMonitor(vec_env)
    
    return vec_env


def train(
    num_envs: int = 4,
    total_timesteps: int = 1_000_000,
    include_camera: bool = True,
    include_npc: bool = True,
    resume_path: Optional[str] = None,
    config: Optional[FullConfig] = None,
    seed: int = 42,
    light_mode: bool = False,
    n_steps: int = 2048,
    batch_size: int = 64,
    progress_bar: bool = True,
) -> PPO:
    """
    Train a PPO agent on the CARLA environment.
    
    Args:
        num_envs: Number of parallel environments
        total_timesteps: Total training steps
        include_camera: Include camera observations
        include_npc: Include NPC vehicle
        resume_path: Path to model checkpoint to resume from
        config: Environment/training configuration
        seed: Random seed
        light_mode: Use lightweight settings (smaller batches, less memory)
        n_steps: Steps per environment before update
        batch_size: Minibatch size for PPO updates
        
    Returns:
        Trained PPO model
    """
    # Setup logging
    log_file = Path("logs") / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file.parent.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Light mode overrides
    if light_mode:
        include_camera = False
        n_steps = 512
        batch_size = 32
        logger.info("LIGHT MODE enabled: no camera, smaller batches")
    
    config = config or DEFAULT_CONFIG
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppo_carla_{timestamp}"
    
    # Create directories
    log_dir = Path(config.training.tensorboard_log) / run_name
    model_dir = Path(config.training.model_save_path)
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("CARLA RL Training")
    print("=" * 60)
    print(f"Run name: {run_name}")
    print(f"Number of environments: {num_envs}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Include camera: {include_camera}")
    print(f"Include NPC: {include_npc}")
    print(f"TensorBoard log: {log_dir}")
    print(f"Model save path: {model_dir}")
    print("=" * 60)
    
    # Create vectorized environment
    print("\nCreating vectorized environment...")
    env = create_vec_env(
        num_envs=num_envs,
        config=config,
        include_camera=include_camera,
        include_npc=include_npc,
        seed=seed,
    )
    print(f"Environment created with {num_envs} parallel instances")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Create or load model
    if resume_path:
        print(f"\nLoading model from {resume_path}...")
        model = PPO.load(resume_path, env=env)
        print("Model loaded successfully")
    else:
        print("\nCreating new PPO model...")
        
        # Policy network architecture
        policy_kwargs = dict(
            net_arch=dict(
                pi=[256, 256],  # Policy network
                vf=[256, 256],  # Value network
            ),
        )
        
        model = PPO(
            policy="MultiInputPolicy",  # Handles Dict observation space
            env=env,
            learning_rate=config.training.learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=config.training.n_epochs,
            gamma=config.training.gamma,
            gae_lambda=config.training.gae_lambda,
            clip_range=config.training.clip_range,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=str(log_dir.parent),
            seed=seed,
        )
        print(f"Model created (n_steps={n_steps}, batch_size={batch_size})")
    
    # Set up callbacks
    callbacks = []
    
    # Checkpoint callback - save model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=max(config.training.save_freq // num_envs, 1),
        save_path=str(model_dir),
        name_prefix=run_name,
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    callbacks.append(checkpoint_callback)
    
    # Resource monitoring callback
    resource_callback = ResourceMonitorCallback(log_freq=100)
    callbacks.append(resource_callback)
    
    # Custom TensorBoard callback
    tb_callback = TensorBoardCallback()
    callbacks.append(tb_callback)
    
    # Train the model
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    print(f"\nTensorBoard: tensorboard --logdir {log_dir.parent}")
    print("Press Ctrl+C to stop training early\n")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList(callbacks),
            tb_log_name=run_name,
            progress_bar=progress_bar,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        logger.info("Training interrupted by user")
    except Exception as e:
        # Log the crash with full traceback
        logger.error(f"Training crashed: {e}")
        logger.error(traceback.format_exc())
        
        # Log system state at crash
        memory = psutil.virtual_memory()
        logger.error(f"System state at crash:")
        logger.error(f"  CPU: {psutil.cpu_percent()}%")
        logger.error(f"  RAM: {memory.used / (1024**3):.1f}GB / {memory.total / (1024**3):.1f}GB ({memory.percent}%)")
        
        print(f"\n\nTraining CRASHED: {e}")
        print(f"See log file for details: {log_file}")
    
    # Save final model
    final_model_path = model_dir / f"{run_name}_final.zip"
    model.save(str(final_model_path))
    print(f"\nFinal model saved: {final_model_path}")
    
    # Also save as "latest" for easy access
    latest_path = model_dir / "ppo_carla_latest.zip"
    model.save(str(latest_path))
    print(f"Latest model saved: {latest_path}")
    
    # Cleanup
    env.close()
    
    return model


def evaluate(
    model_path: str,
    num_episodes: int = 10,
    include_camera: bool = True,
    include_npc: bool = True,
    render: bool = False,
    config: Optional[FullConfig] = None,
) -> dict:
    """
    Evaluate a trained model.
    
    Args:
        model_path: Path to trained model
        num_episodes: Number of evaluation episodes
        include_camera: Include camera observations
        include_npc: Include NPC vehicle
        render: Whether to render (requires display)
        config: Environment configuration
        
    Returns:
        Dictionary with evaluation statistics
    """
    config = config or DEFAULT_CONFIG
    
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    
    # Create single environment for evaluation
    env = CarlaRLEnv(
        config=config,
        instance_id=0,
        include_camera=include_camera,
        include_npc=include_npc,
        render_mode="human" if render else None,
    )
    
    print(f"\nEvaluating for {num_episodes} episodes...")
    
    episode_rewards = []
    episode_lengths = []
    episode_progresses = []
    successes = 0
    collisions = 0
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            if render:
                env.render()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_progresses.append(info.get("progress", 0.0))
        
        if info.get("termination_reason") == "goal_reached":
            successes += 1
        elif info.get("termination_reason") == "collision":
            collisions += 1
        
        print(f"  Episode {ep + 1}/{num_episodes}: "
              f"reward={episode_reward:.2f}, "
              f"length={episode_length}, "
              f"progress={info.get('progress', 0.0):.2%}, "
              f"reason={info.get('termination_reason', 'unknown')}")
    
    env.close()
    
    results = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "mean_progress": np.mean(episode_progresses),
        "success_rate": successes / num_episodes,
        "collision_rate": collisions / num_episodes,
    }
    
    print("\n" + "=" * 40)
    print("Evaluation Results")
    print("=" * 40)
    print(f"Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Mean length: {results['mean_length']:.1f}")
    print(f"Mean progress: {results['mean_progress']:.2%}")
    print(f"Success rate: {results['success_rate']:.2%}")
    print(f"Collision rate: {results['collision_rate']:.2%}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train RL agent for CARLA autonomous driving",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["train", "eval"],
        default="train",
        help="Run mode: train a new model or evaluate an existing one"
    )
    
    # Training arguments
    parser.add_argument(
        "--num-envs", "-n",
        type=int,
        default=4,
        help="Number of parallel environments"
    )
    
    parser.add_argument(
        "--timesteps", "-t",
        type=int,
        default=1_000_000,
        help="Total training timesteps"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to model checkpoint to resume training from"
    )
    
    # Environment arguments
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="Disable camera observations (state-only)"
    )
    
    parser.add_argument(
        "--no-npc",
        action="store_true",
        help="Disable NPC vehicle"
    )
    
    parser.add_argument(
        "--light",
        action="store_true",
        help="Lightweight mode: no camera, smaller batches, less memory"
    )
    
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="Steps per environment before PPO update"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Minibatch size for PPO updates"
    )
    
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Disable progress bar (can help with stability)"
    )
    
    # Evaluation arguments
    parser.add_argument(
        "--model",
        type=str,
        default="models/ppo_carla_latest.zip",
        help="Model path for evaluation"
    )
    
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes"
    )
    
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render during evaluation"
    )
    
    # General arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU mode (use if GPU is not compatible)"
    )
    
    args = parser.parse_args()
    
    # Force CPU mode if requested (must be done before PPO is created)
    if args.cpu:
        import torch
        torch.cuda.is_available = lambda: False
        print("Forcing CPU mode (GPU disabled)")
    
    if args.mode == "train":
        train(
            num_envs=args.num_envs,
            total_timesteps=args.timesteps,
            include_camera=not args.no_camera,
            include_npc=not args.no_npc,
            resume_path=args.resume,
            seed=args.seed,
            light_mode=args.light,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            progress_bar=not args.no_progress_bar,
        )
    else:  # eval
        evaluate(
            model_path=args.model,
            num_episodes=args.episodes,
            include_camera=not args.no_camera,
            include_npc=not args.no_npc,
            render=args.render,
        )


if __name__ == "__main__":
    main()

