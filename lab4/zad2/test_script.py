import gymnasium as gym
from stable_baselines3 import DDPG
import numpy as np

def test_ddpg_agent(model_path='ddpg_lunar', episodes=10, render_mode=False):
    env = gym.make("LunarLanderContinuous-v3", render_mode='human' if render_mode else None)
    model = DDPG.load(model_path)

    total_rewards = []
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        total_rewards.append(total_reward)
        print(f"Test Episode {episode + 1}: Reward = {total_reward:.2f}")

    avg_reward = np.mean(total_rewards)
    success_rate = np.mean(np.array(total_rewards) > 200) * 100
    print(f"\nAverage Reward over {episodes} episodes: {avg_reward:.2f}")
    print(f"Success rate: {success_rate:.2f}%")

    env.close()
