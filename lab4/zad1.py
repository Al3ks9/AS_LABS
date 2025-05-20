import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from torch.distributions import Categorical

# =================== Set device ===================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =================== Actor Network ===================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Actor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.model(state)

    def train_step(self, optimizer, state, action, advantage):
        probs = self.forward(state)
        dist = Categorical(probs)
        log_prob = dist.log_prob(action)
        loss = -(log_prob * advantage.detach()).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# =================== Critic Network ===================
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.model(state).squeeze(-1)

    def train_step(self, optimizer, state, target):
        value = self.forward(state)
        loss = nn.MSELoss()(value, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# =================== Training Function ===================
def train_agent(episodes=1000, gamma=0.99):
    env = gym.make("LunarLander-v3")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    actor = Actor(state_dim, action_dim).to(device)
    critic = Critic(state_dim).to(device)

    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-4)
    critic_optimizer = optim.Adam(critic.parameters(), lr=5e-4)

    start_time = time.time()
    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        done = False
        total_reward = 0
        transitions = []

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
            probs = actor(state_tensor)
            dist = Categorical(probs)
            action = dist.sample().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            transitions.append((state, action, reward, next_state, done))
            state = next_state

        for t in range(len(transitions)):
            s, a, r, s_next, d = transitions[t]

            state_tensor = torch.tensor(s, dtype=torch.float32).to(device)
            next_state_tensor = torch.tensor(s_next, dtype=torch.float32).to(device)
            reward_tensor = torch.tensor(r, dtype=torch.float32).to(device)
            done_tensor = torch.tensor(d, dtype=torch.float32).to(device)

            value = critic(state_tensor)
            next_value = critic(next_state_tensor)
            target = reward_tensor + (1 - done_tensor) * gamma * next_value
            advantage = target - value

            actor.train_step(actor_optimizer, state_tensor, torch.tensor(a).to(device), advantage)
            critic.train_step(critic_optimizer, state_tensor, target)

        elapsed = time.time() - start_time
        avg_time = elapsed / episode
        remaining = avg_time * (episodes - episode)
        eta = time.strftime("%H:%M:%S", time.gmtime(remaining))
        print(f"\rEpisode {episode}/{episodes} - Reward: {total_reward:.2f} - ETA: {eta}", end='')

    torch.save(actor.state_dict(), "actor.pt")
    torch.save(critic.state_dict(), 'critic.pt')


# =================== Evaluation Function ===================
def test_agent(episodes=100, render_mode=False):
    env = gym.make("LunarLander-v3", render_mode='human' if render_mode else None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    actor = Actor(state_dim, action_dim).to(device)
    actor.load_state_dict(torch.load("actor.pt", map_location=device))
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
        print(f"Test Episode {episode+1}: Reward = {total_reward:.2f}")

    avg_reward = np.mean(total_rewards)
    print(f"\nAverage Reward over {episodes} episodes: {avg_reward:.2f}")
    return avg_reward


# =================== Main ===================
if __name__ == "__main__":
    # train_agent(episodes=1000)
    test_agent(5, render_mode=True)
