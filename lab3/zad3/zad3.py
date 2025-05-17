import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import ale_py
from PIL import Image
from collections import deque
import time

gym.register_envs(ale_py)


def preprocess_state(state):
    grayscale_img = np.array(state, dtype=np.float32)
    grayscale_img = grayscale_img / 255.0
    preprocessed = grayscale_img[np.newaxis, :, :]
    return preprocessed


class DQN(nn.Module):
    def __init__(self, input_shape, output_dim):
        super(DQN, self).__init__()
        input_dim = input_shape[0] * input_shape[1] * input_shape[2]
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


BATCH_SIZE = 32
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.0001
TARGET_UPDATE = 10
MEMORY_CAPACITY = 10000


def train_dqn(episodes=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make('ALE/MsPacman-v5', render_mode=None, obs_type='grayscale')
    input_shape = (1, 210, 160)
    output_dim = env.action_space.n

    policy_net = DQN(input_shape, output_dim).to(device)
    target_net = DQN(input_shape, output_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayMemory(MEMORY_CAPACITY)

    epsilon = EPSILON_START
    start_time = time.time()  # Start timing the training process

    for episode in range(episodes):
        state = env.reset()[0]
        frame = torch.FloatTensor(preprocess_state(state)).unsqueeze(0).to(device)
        total_reward = 0
        terminated, truncated = False, False

        while not terminated and not truncated:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = policy_net(frame).argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            next_frame = torch.FloatTensor(preprocess_state(next_state)).unsqueeze(0).to(device)

            memory.push((frame, action, reward, next_frame, terminated))

            frame = next_frame
            total_reward += reward

            if len(memory) >= BATCH_SIZE:
                batch = memory.sample(BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.cat(states).to(device)
                actions = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                next_states = torch.cat(next_states).to(device)
                dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

                q_values = policy_net(states).gather(1, actions)
                with torch.no_grad():
                    next_q_values = target_net(next_states).max(1, keepdim=True)[0]
                    target_q_values = rewards + (GAMMA * next_q_values * (1 - dones))

                loss = nn.MSELoss()(q_values, target_q_values)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Calculate and display ETA
        elapsed_time = time.time() - start_time
        episodes_remaining = episodes - (episode + 1)
        avg_time_per_episode = elapsed_time / (episode + 1)
        eta = avg_time_per_episode * episodes_remaining
        eta_minutes, eta_seconds = divmod(int(eta), 60)
        print(f"\rEpisode {episode + 1}/{episodes} completed. ETA: {eta_minutes}m {eta_seconds}s", end='')

    print("\nTraining complete.")
    torch.save(policy_net.state_dict(), 'dqn_mspacman.pth')
    env.close()


def test_dqn(model_path, episodes=10, render=False):
    env = gym.make('ALE/MsPacman-v5', render_mode='human' if render else None, obs_type='grayscale')
    policy_net = DQN((1, 210, 160), 9)
    policy_net.load_state_dict(torch.load(model_path))
    policy_net.eval()

    total_rewards = []

    for episode in range(episodes):
        state = env.reset()[0]
        state = torch.FloatTensor(preprocess_state(state)).unsqueeze(0)
        total_reward = 0
        terminated, truncated = False, False

        while not terminated and not truncated:
            with torch.no_grad():
                action = policy_net(state).argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = torch.FloatTensor(preprocess_state(next_state)).unsqueeze(0)

            state = next_state
            total_reward += reward

            if render:
                env.render()

        total_rewards.append(total_reward)
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    env.close()

    print(f"Average Reward over {episodes} episodes: {np.mean(total_rewards):.2f}")


if __name__ == '__main__':
    # Uncomment the following lines to train or test the model
    # train_dqn(500)
    # test_dqn('dqn_mspacman.pth', episodes=5, render=True)
    test_dqn('dqn_mspacman.pth', episodes=100, render=False)
    # 207 avg reward after 100 episodes
    # 603.9 avg reward after 500 episodes
