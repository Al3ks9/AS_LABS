import gymnasium as gym
from q_learning import *

if __name__ == '__main__':
    env = gym.make("FrozenLake-v1")

    num_states = env.observation_space.n
    num_actions = env.action_space.n

    q_table = random_q_table(-1, 0, (num_states, num_actions))

    learning_rate = 0.01
    discount_factor = 0.9
    epsilon = 0.8
    num_episodes = 100

    avg_iter = 0
    avg_reward = 0
    for episode in range(num_episodes):
        state, _ = env.reset()
        num_iter = 0
        while True:
            num_iter += 1
            action = get_action(env, q_table, state, epsilon)
            new_state, reward, terminated, _, _ = env.step(action)

            new_q = calculate_new_q_value(q_table,
                                          state, new_state,
                                          action, reward,
                                          learning_rate, discount_factor)

            q_table[state, action] = new_q

            state = new_state
            if terminated:
                break
        avg_iter += num_iter
        avg_reward += reward

    print(avg_iter / 100, avg_reward / 100)

    avg_iter = 0
    avg_reward = 0
    for episode in range(num_episodes):
        state, _ = env.reset()
        num_iter = 0
        while True:
            num_iter += 1
            action = get_best_action(q_table, state)

            new_state, reward, terminated, _, _ = env.step(action)

            state = new_state
            if terminated:
                break
        avg_iter += num_iter
        avg_reward += reward

    print(avg_iter / 100, avg_reward / 100)


    # Epsilon dobiva najgolem reward na kraj.
    # avg_ter = 7.9, avg_reward = 0.0
    # Epsilon: avg_ter = 6.99, avg_reward = 0.03
