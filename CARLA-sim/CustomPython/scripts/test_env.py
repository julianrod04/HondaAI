#!/usr/bin/env python3
"""
Simple test of the CARLA RL environment without SB3 wrappers.

This helps diagnose if crashes are from the environment or the training setup.

Usage:
    python scripts/test_env.py
    python scripts/test_env.py --steps 100
    python scripts/test_env.py --no-camera
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).parent.parent))

from rl.carla_env import CarlaRLEnv
from rl.config import DEFAULT_CONFIG


def print_resources():
    """Print current system resources."""
    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory()
    print(f"  [Resources] CPU: {cpu:.1f}%, RAM: {mem.used / (1024**3):.1f}GB ({mem.percent:.1f}%)")


def test_env(
    num_steps: int = 100,
    include_camera: bool = True,
    include_npc: bool = True,
):
    """Test the environment with random actions."""
    
    print("=" * 60)
    print("CARLA Environment Test")
    print("=" * 60)
    print(f"Steps: {num_steps}")
    print(f"Camera: {include_camera}")
    print(f"NPC: {include_npc}")
    print("=" * 60)
    
    print_resources()
    
    print("\nCreating environment...")
    try:
        env = CarlaRLEnv(
            config=DEFAULT_CONFIG,
            instance_id=0,
            include_camera=include_camera,
            include_npc=include_npc,
        )
        print("Environment created!")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
    except Exception as e:
        print(f"FAILED to create environment: {e}")
        traceback.print_exc()
        return False
    
    print_resources()
    
    print("\nResetting environment...")
    try:
        obs, info = env.reset()
        print(f"Reset successful!")
        print(f"  State shape: {obs['state'].shape}")
        if include_camera:
            print(f"  Camera shape: {obs['camera'].shape}")
        print(f"  Info: {info}")
    except Exception as e:
        print(f"FAILED to reset: {e}")
        traceback.print_exc()
        env.close()
        return False
    
    print_resources()
    
    print(f"\nRunning {num_steps} steps with random actions...")
    episode_reward = 0
    step_times = []
    
    try:
        for step in range(num_steps):
            start_time = time.time()
            
            # Random action
            action = env.action_space.sample()
            
            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            
            step_time = time.time() - start_time
            step_times.append(step_time)
            episode_reward += reward
            
            # Print progress
            if (step + 1) % 10 == 0:
                avg_time = sum(step_times[-10:]) / min(10, len(step_times))
                fps = 1.0 / avg_time if avg_time > 0 else 0
                print(f"  Step {step + 1}/{num_steps}: "
                      f"reward={reward:.3f}, "
                      f"progress={info.get('progress', 0):.1%}, "
                      f"fps={fps:.1f}")
            
            # Print resources every 50 steps
            if (step + 1) % 50 == 0:
                print_resources()
            
            # Check for episode end
            if terminated or truncated:
                print(f"\n  Episode ended at step {step + 1}: {info.get('termination_reason', 'unknown')}")
                print(f"  Total reward: {episode_reward:.2f}")
                print("\n  Resetting...")
                obs, info = env.reset()
                episode_reward = 0
        
        print(f"\nTest completed successfully!")
        print(f"  Average step time: {sum(step_times) / len(step_times):.4f}s")
        print(f"  Average FPS: {len(step_times) / sum(step_times):.1f}")
        
    except Exception as e:
        print(f"\nCRASHED at step {step + 1}: {e}")
        traceback.print_exc()
        print_resources()
        env.close()
        return False
    
    print("\nClosing environment...")
    env.close()
    print("Done!")
    
    print_resources()
    return True


def main():
    parser = argparse.ArgumentParser(description="Test CARLA RL environment")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps to run")
    parser.add_argument("--no-camera", action="store_true", help="Disable camera")
    parser.add_argument("--no-npc", action="store_true", help="Disable NPC")
    args = parser.parse_args()
    
    success = test_env(
        num_steps=args.steps,
        include_camera=not args.no_camera,
        include_npc=not args.no_npc,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()


