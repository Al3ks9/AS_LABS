import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from PIL import Image
import ale_py

gym.register_envs(ale_py)


# Preprocess state function
def preprocess_state(state):
    img = Image.fromarray(state)  # Convert to PIL Image
    img = img.convert('L')  # Convert to grayscale
    img = img.resize((84, 84), Image.Resampling.LANCZOS)  # Resize to 84x84
    grayscale_img = np.array(img, dtype=np.float32)  # Convert to numpy array
    grayscale_img = grayscale_img / 255.0  # Normalize pixel values to [0, 1]
    preprocessed = grayscale_img[np.newaxis, :, :]  # Add channel dimension
    return preprocessed

# Replay memory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Dueling DQN model
class DuelingDQN(nn.Module):
    def __init__(self, input_shape, output_dim):
        super(DuelingDQN, self).__init__()
        input_dim = input_shape[0] * input_shape[1] * input_shape[2]
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU()
        )
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.fc(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        # Combine value and advantage streams
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

# Hyperparameters
BATCH_SIZE = 32
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.0001
TARGET_UPDATE = 10
MEMORY_CAPACITY = 10000

# Training loop
def train_dueling_dqn(episodes=1000):
    env = gym.make('ALE/MsPacman-v5', render_mode=None)
    input_shape = (1, 84, 84)  # Based on the preprocess_state function
    output_dim = env.action_space.n

    policy_net = DuelingDQN(input_shape, output_dim)
    target_net = DuelingDQN(input_shape, output_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayMemory(MEMORY_CAPACITY)

    epsilon = EPSILON_START
    for episode in range(episodes):
        state = env.reset()[0]
        frame = torch.FloatTensor(preprocess_state(state)).unsqueeze(0)  # Add batch dimension
        total_reward = 0
        terminated, truncated = False, False

        while not terminated and not truncated:
            # Use the preprocessed frame as input
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = policy_net(frame).argmax().item()

            # Take action in the environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_frame = torch.FloatTensor(preprocess_state(next_state)).unsqueeze(0)

            # Store experience in replay memory
            memory.push((frame, action, reward, next_frame, terminated))

            frame = next_frame
            total_reward += reward

            # Train the network
            if len(memory) >= BATCH_SIZE:
                batch = memory.sample(BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.cat(states)
                actions = torch.LongTensor(actions).unsqueeze(1)
                rewards = torch.FloatTensor(rewards).unsqueeze(1)
                next_states = torch.cat(next_states)
                dones = torch.FloatTensor(dones).unsqueeze(1)

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

        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    torch.save(policy_net.state_dict(), 'dueling_dqn_mspacman.pth')
    env.close()


def test_dqn(model_path, episodes=10, render=False):
    env = gym.make('ALE/MsPacman-v5', render_mode='human' if render else None)
    # Load the trained model
    policy_net = DuelingDQN((1, 84, 84), 9)  # Adjust input/output dimensions as needed
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
    train_dueling_dqn(100)
    test_dqn('dueling_dqn_mspacman.pth', episodes=100, render=False)
    # after 100 episodes of training, dueling dqn achieved an average reward of 210,
    # which is significantly better than the normal dqn with an average of 112.