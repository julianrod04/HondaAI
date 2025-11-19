"""
Training harness for CarlaLaneChangeEnv with PPO (stable-baselines3).
Provides CLI entry-point and a callable run_training(config) helper so it can
be imported into notebooks.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from carla_lane_change_env import CarlaLaneChangeEnv, DEFAULT_STYLE_CONFIGS


@dataclass
class TrainingConfig:
    host: str = "localhost"
    port: int = 2000
    map_name: str = "Town06"
    style: str = "normal"
    observation_type: str = "state"
    total_timesteps: int = 200_000
    chunk_steps: int = 25_000
    eval_episodes: int = 2
    render_demos: bool = True
    log_dir: Optional[str] = None
    model_path: str = "ppo_lane_change"
    n_steps: int = 512
    batch_size: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    enable_camera_for_training: bool = False


def _make_env(env_kwargs: Dict):
    return lambda: CarlaLaneChangeEnv(**env_kwargs)


def run_demo_episodes(model: PPO, env_kwargs: Dict, n_episodes: int, render: bool):
    """Run deterministic evaluation episodes with optional rendering."""
    if n_episodes <= 0:
        return []
    eval_kwargs = env_kwargs.copy()
    eval_kwargs["enable_camera_sensor"] = True
    env = CarlaLaneChangeEnv(**eval_kwargs)
    stats: List[Dict] = []
    for ep in range(n_episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0.0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += float(reward)
            if render:
                env.render()
        info = info or {}
        info.update({"episode_reward": episode_reward})
        stats.append(info)
        print(
            f"[DEMO {ep}] style={info.get('style')} outcome={info.get('event')} "
            f"reward={episode_reward:.1f} progress={info.get('progress', 0):.1f}"
        )
    env.close()
    return stats


def run_training(config: TrainingConfig):
    """High-level PPO training loop with periodic evaluation demos."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    env_kwargs = dict(
        host=config.host,
        port=config.port,
        map_name=config.map_name,
        style=config.style,
        observation_type=config.observation_type,
        enable_camera_sensor=config.enable_camera_for_training or config.render_demos,
        autopilot_npc=True,
        synchronous_mode=False,
    )

    vec_env = DummyVecEnv([_make_env(env_kwargs)])
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        tensorboard_log=config.log_dir,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        device=device,
    )

    total_steps = 0
    iteration = 0
    while total_steps < config.total_timesteps:
        chunk = min(config.chunk_steps, config.total_timesteps - total_steps)
        print(f"Starting PPO learn chunk={chunk} (total so far={total_steps})")
        model.learn(total_timesteps=chunk, reset_num_timesteps=(iteration == 0))
        total_steps += chunk
        vec_env.close()  # Free CARLA actors before evaluation
        if config.eval_episodes > 0:
            stats = run_demo_episodes(model, env_kwargs, config.eval_episodes, config.render_demos)
            successes = sum(1 for s in stats if s.get("event") == "success")
            failures = len(stats) - successes
            print(
                f"[EVAL] Steps={total_steps} successes={successes} failures={failures} "
                f"avg_reward={np.mean([s['episode_reward'] for s in stats]) if stats else 0:.1f}"
            )
        if total_steps < config.total_timesteps:
            vec_env = DummyVecEnv([_make_env(env_kwargs)])
            model.set_env(vec_env)
        iteration += 1

    model.save(config.model_path)
    print(f"Training complete. Model saved to {config.model_path}")
    vec_env.close()
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on CarlaLaneChangeEnv.")
    parser.add_argument("--style", type=str, default="normal", choices=list(DEFAULT_STYLE_CONFIGS.keys()))
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--map", type=str, default="Town06")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--chunk-steps", type=int, default=25_000)
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--no-render", action="store_true", help="Disable demo rendering.")
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--model-path", type=str, default="ppo_lane_change")
    parser.add_argument("--obs-type", type=str, default="state", choices=["state", "rgb"])
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = TrainingConfig(
        host=args.host,
        port=args.port,
        map_name=args.map,
        style=args.style,
        observation_type=args.obs_type,
        total_timesteps=args.timesteps,
        chunk_steps=args.chunk_steps,
        eval_episodes=args.eval_episodes,
        render_demos=not args.no_render,
        log_dir=args.log_dir,
        model_path=args.model_path,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
    )
    run_training(cfg)
