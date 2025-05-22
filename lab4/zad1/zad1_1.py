import torch
import torch.optim as optim
import gymnasium as gym
import numpy as np
from test_lander import test_agent
from models import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_actor_critic(env_name="LunarLander-v3", episodes=500, gamma=0.99, lr=1e-3):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    actor = Actor(state_dim, action_dim).to(device)
    critic = Critic(state_dim).to(device)

    actor_optimizer = optim.Adam(actor.parameters(), lr=lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=lr)

    all_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            probs = actor(state_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            total_reward += reward

            next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
            reward_tensor = torch.tensor([reward], dtype=torch.float32, device=device)
            done_tensor = torch.tensor([done], dtype=torch.float32, device=device)

            # Critic update
            value = critic(state_tensor)
            next_value = critic(next_state_tensor)
            target = reward_tensor + gamma * next_value * (1 - done_tensor)
            critic_loss = nn.functional.mse_loss(value, target.detach())

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # Actor update
            advantage = (target - value).detach()
            log_prob = torch.log(probs.squeeze(0)[action])
            actor_loss = -log_prob * advantage

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            state = next_state

        all_rewards.append(total_reward)
        if (episode + 1) % 10 == 0:
            avg = np.mean(all_rewards[-10:])
            print(f"Episode {episode + 1}, Avg Reward: {avg:.2f}")

    torch.save(actor.state_dict(), "actor.pt")
    torch.save(critic.state_dict(), 'critic.pt')

    env.close()


if __name__ == '__main__':
    # train_actor_critic(episodes=1000)
    test_agent('new_version/actor.pt', episodes=5, render_mode=True)
    test_agent('new_version/actor.pt', 100)
    # test_agent(100, False)
