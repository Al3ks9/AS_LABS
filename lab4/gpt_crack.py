import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from tqdm import trange


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.model(x)


class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    values.append(0)
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    return advantages


def evaluate(env, actor, episodes=5):
    rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        done, total_reward = False, 0
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            with torch.no_grad():
                action = torch.argmax(actor(state_tensor)).item()
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
    return np.mean(rewards)


def train_loop(env_name="LunarLander-v3", episodes=500, success_threshold=200):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    actor = Actor(state_dim, action_dim)
    critic = Critic(state_dim)
    opt_actor = optim.Adam(actor.parameters(), lr=1e-4)
    opt_critic = optim.Adam(critic.parameters(), lr=1e-3)

    best_eval = -float('inf')

    for ep in trange(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        states, actions, rewards, dones = [], [], [], []

        while not done:
            s_tensor = torch.tensor(state, dtype=torch.float32)
            probs = actor(s_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()

            next_state, reward, terminated, truncated, _ = env.step(action)

            if state[6] == 1.0 and state[7] == 1.0 and action == 0:
                reward += 10

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(float(terminated or truncated))

            total_reward += reward
            state = next_state
            done = terminated or truncated

        if total_reward > -150:
            states_tensor = torch.tensor(states, dtype=torch.float32)
            actions_tensor = torch.tensor(actions, dtype=torch.long)

            with torch.no_grad():
                values = critic(states_tensor).squeeze().cpu().tolist()

            advantages = compute_gae(rewards, values.copy(), dones)
            advantages = torch.tensor(advantages, dtype=torch.float32)
            returns = advantages + torch.tensor(values, dtype=torch.float32)

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            probs = actor(states_tensor)
            dist = torch.distributions.Categorical(probs)
            log_probs = dist.log_prob(actions_tensor)
            entropy = dist.entropy().mean()

            actor_loss = -(log_probs * advantages).mean() - 0.01 * entropy
            critic_loss = nn.MSELoss()(critic(states_tensor).squeeze(), returns)

            opt_actor.zero_grad()
            actor_loss.backward()
            opt_actor.step()

            opt_critic.zero_grad()
            critic_loss.backward()
            opt_critic.step()

        if ep % 20 == 0:
            avg = evaluate(env, actor)
            if avg > best_eval:
                best_eval = avg
                torch.save(actor.state_dict(), "best_actor.pth")
                torch.save(critic.state_dict(), "best_critic.pth")

    torch.save(actor.state_dict(), "final_actor.pth")
    torch.save(critic.state_dict(), "final_critic.pth")
    env.close()


train_loop(episodes=1000)
