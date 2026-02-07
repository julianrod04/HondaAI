import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# Train
env_id = "CartPole-v1"
env = make_vec_env(env_id, n_envs=8) # Vectorized env is recommended for PPO (faster + more stable training)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    n_steps=2048,        # rollout length per env (per update)
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    learning_rate=3e-4,
)

model.learn(total_timesteps=300_000)
model.save("ppo_cartpole")

# Test
test_env = gym.make(env_id)
mean_reward, std_reward = evaluate_policy(model, test_env, n_eval_episodes=20, deterministic=True)
print(mean_reward, std_reward)

obs, info = test_env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = test_env.step(action)
    if terminated or truncated:
        obs, info = test_env.reset()
