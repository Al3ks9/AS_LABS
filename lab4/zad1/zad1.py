import torch
import torch.optim as optim
import gymnasium as gym
from test_lander import test_agent
from models import *


def train_actor_critic(episodes=500, gamma=0.99):
    env = gym.make("LunarLander-v3", continuous=True)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor = Actor(state_dim, action_dim)
    critic = Critic(state_dim)

    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-3)
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)

    action_low = env.action_space.low
    action_high = env.action_space.high

    noise = OUNoise(action_dim)

    all_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        noise.reset()

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = actor(state_tensor).detach().numpy().squeeze()
            noisy_action = np.clip(action + noise.sample(), action_low, action_high)

            next_state, reward, terminated, truncated, _ = env.step(noisy_action)
            done = terminated or truncated
            total_reward += reward

            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            reward_tensor = torch.tensor([reward], dtype=torch.float32)
            done_tensor = torch.tensor([done], dtype=torch.float32)

            value = critic(state_tensor)
            next_value = critic(next_state_tensor)
            target = reward_tensor + gamma * next_value * (1 - done_tensor)
            critic_loss = nn.functional.mse_loss(value, target.detach())

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            actor_loss = -critic(torch.cat([state_tensor], dim=1)).mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            state = next_state

        all_rewards.append(total_reward)
        if (episode + 1) % 10 == 0:
            avg = np.mean(all_rewards[-10:])
            print(f"Episode {episode + 1}, Avg Reward: {avg:.2f}")

    torch.save(actor.state_dict(), "continuous/actor.pt")
    torch.save(critic.state_dict(), "continuous/critic.pt")

    env.close()


if __name__ == '__main__':
    # train_actor_critic(episodes=1000)
    test_agent('continuous/actor.pt', episodes=5, render_mode=True)
    test_agent('continuous/actor.pt', 100)
