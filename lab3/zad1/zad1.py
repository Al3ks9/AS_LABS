import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()
        self.fc1 = nn.Linear(in_states, h1_nodes)
        self.out = nn.Linear(h1_nodes, out_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return x


class ReplayMemory:
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)


class MountainCarDQN:

    learning_rate_a = 0.001
    discount_factor_g = 0.8
    network_sync_rate = 10
    replay_memory_size = 1000
    mini_batch_size = 32

    loss_fn = nn.MSELoss()
    optimizer = None

    ACTIONS = ['L', 'N', 'R']

    def train(self, episodes, render=False):
        env = gym.make('MountainCar-v0', render_mode='human' if render else None, max_episode_steps=200)
        num_actions = env.action_space.n

        epsilon = 1
        memory = ReplayMemory(self.replay_memory_size)

        policy_dqn = DQN(in_states=2, h1_nodes=64, out_actions=num_actions)
        target_dqn = DQN(in_states=2, h1_nodes=64, out_actions=num_actions)

        target_dqn.load_state_dict(policy_dqn.state_dict())

        print('Policy (random, before training):')
        self.print_dqn(policy_dqn)

        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        rewards_per_episode = np.zeros(episodes)
        epsilon_history = []
        step_count = 0

        for i in range(episodes):
            state = env.reset()[0]
            terminated = False
            truncated = False

            while not terminated and not truncated:
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = policy_dqn(self.state_to_dqn_input(state)).argmax().item()

                new_state, reward, terminated, truncated, _ = env.step(action)

                # Reward based on progress toward the goal
                reward = abs(new_state[0]) * 100 if new_state[0] > -0.5 else abs(new_state[0]) * 75

                reward = -100 if -0.7 <= new_state[0] <= -0.5 else reward

                # Add a velocity-based incentive
                reward += abs(new_state[1]) * 50

                # Add a large reward for reaching the goal
                if terminated and new_state[0] >= 0.5:  # Goal position
                    reward += 2000

                memory.append((state, action, new_state, reward, terminated))
                state = new_state
                step_count += 1

            rewards_per_episode[i] = reward

            if len(memory) > self.mini_batch_size:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)

                epsilon = max(epsilon * 0.995, 0.01)  # Epsilon decay
                epsilon_history.append(epsilon)

                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0

        env.close()
        torch.save(policy_dqn.state_dict(), "zad1.pt")

        plt.figure(1)
        sum_rewards = np.zeros(episodes)
        for x in range(episodes):
            sum_rewards[x] = np.sum(rewards_per_episode[max(0, x - 100):(x + 1)])
        plt.subplot(121)
        plt.plot(sum_rewards)
        plt.subplot(122)
        plt.plot(epsilon_history)
        plt.savefig('zad1.png')

    def state_to_dqn_input(self, state):
        # Normalize the state values to [-1, 1]
        position, velocity = state
        normalized_position = (position - (-1.2)) / (0.6 - (-1.2)) * 2 - 1
        normalized_velocity = (velocity - (-0.07)) / (0.07 - (-0.07)) * 2 - 1
        return torch.FloatTensor([normalized_position, normalized_velocity])

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in mini_batch:
            if terminated:
                target = torch.FloatTensor([reward])
            else:
                with torch.no_grad():
                    target = torch.FloatTensor(
                        reward + self.discount_factor_g * target_dqn(
                            self.state_to_dqn_input(new_state)).max()
                    )

            current_q = policy_dqn(self.state_to_dqn_input(state))
            current_q_list.append(current_q)

            target_q = target_dqn(self.state_to_dqn_input(state))
            target_q[action] = target
            target_q_list.append(target_q)

        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def test(self, episodes, render=True):
        env = gym.make('MountainCar-v0', render_mode='human' if render else None, max_episode_steps=300)
        num_actions = env.action_space.n

        policy_dqn = DQN(in_states=2, h1_nodes=64, out_actions=num_actions)
        policy_dqn.load_state_dict(torch.load("zad1.pt"))
        policy_dqn.eval()

        print('Policy (trained):')
        # self.print_dqn(policy_dqn)

        total_reward = 0

        for i in range(episodes):
            state = env.reset()[0]
            terminated = False
            truncated = False
            episode_reward = 0

            while not terminated and not truncated:
                with torch.no_grad():
                    action = policy_dqn(self.state_to_dqn_input(state)).argmax().item()
                state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward

            total_reward += episode_reward

        env.close()

        print(f"Average reward over {episodes} episodes: {total_reward / episodes:.2f}")

    def print_dqn(self, dqn):
        pos_space = np.linspace(-1.2, 0.6, 10)  # Divide position into 10 segments
        velocity_space = np.linspace(-0.07, 0.07, 10)  # Divide velocity into 10 segments

        for position in pos_space:
            for velocity in velocity_space:
                state = torch.FloatTensor([(position - (-1.2)) / (0.6 - (-1.2)) * 2 - 1,
                                           (velocity - (-0.07)) / (0.07 - (-0.07)) * 2 - 1])
                q_values = dqn(state).tolist()
                best_action = self.ACTIONS[np.argmax(q_values)]
                q_values_str = " ".join(f"{q:+.2f}" for q in q_values)
                print(
                    f"Pos: {position:+.2f}, Vel: {velocity:+.2f}, Best Action: {best_action}, Q-Values: [{q_values_str}]")


if __name__ == '__main__':
    dqn = MountainCarDQN()
    dqn.train(episodes=250, render=False)
    dqn.test(episodes=5)
    dqn.test(episodes=100, render=False)
    dqn.test(episodes=50, render=False)
    # Average reward over 5 episodes: -500.00
    # Average reward over 100 episodes: -500.00
    # Average reward over 50 episodes: -500.00
