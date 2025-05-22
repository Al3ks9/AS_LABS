import gymnasium as gym
from models import *
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_agent(file_name='actor.pt', episodes=100, render_mode=False):
    env = gym.make("LunarLander-v3", render_mode='human' if render_mode else None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    actor = Actor(state_dim, action_dim).to(device)
    actor.load_state_dict(torch.load(file_name, map_location=device))
    actor.eval()

    print()
    total_rewards = []
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
            with torch.no_grad():
                probs = actor(state_tensor)
            action = torch.argmax(probs).item()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        total_rewards.append(total_reward)
        print(f"Test Episode {episode + 1}: Reward = {total_reward:.2f}")

    avg_reward = np.mean(total_rewards)
    print(f"\nAverage Reward over {episodes} episodes: {avg_reward:.2f}")
    print(f"Success rate: {np.mean(np.array(total_rewards) > 200) * 100:.2f}%")
