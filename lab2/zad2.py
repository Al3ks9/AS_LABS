import gymnasium as gym
from q_learning import *

if __name__ == '__main__':
    env = gym.make("Taxi-v3")

    num_states = env.observation_space.n
    num_actions = env.action_space.n

    q_table = random_q_table(-1, 0, (num_states, num_actions))

    learning_rate = 0.1
    discount_factor = 0.9
    epsilon = 0.5
    num_episodes = 100

    avg_iter = 0
    avg_reward = 0
    for episode in range(num_episodes):
        state, _ = env.reset()
        num_iter = 0
        while True:
            num_iter += 1
            action = get_best_action(q_table, state)

            new_state, reward, terminated, _, _ = env.step(action)

            new_q = calculate_new_q_value(q_table,
                                          state, new_state,
                                          action, reward,
                                          learning_rate, discount_factor)

            q_table[state, action] = new_q

            state = new_state
            if terminated:
                break
        # print(f'End of {episode}')
        avg_iter += num_iter
        avg_reward += reward

    print(avg_iter / 100, avg_reward / 100)

    avg_iter = 0
    avg_reward = 0
    fails = 0
    for episode in range(num_episodes):
        state, _ = env.reset()
        num_iter = 0
        while True:
            num_iter += 1
            # action = get_random_action(env)  # 1
            action = get_best_action(q_table, state)  # 2
            # action = get_action(env, q_table, state, epsilon)

            new_state, reward, terminated, _, _ = env.step(action)

            state = new_state
            if terminated:
                break
            if num_iter > 400:
                fails += 1
                break
        # print(f'End of {episode}')
        avg_iter += num_iter
        avg_reward += reward

    print('fails: ', fails)
    print(avg_iter / 100, avg_reward / 100)


    # Q-learning ne raboti voopshto dobro, otkako se izvrshi treniranjeto, ne mozhe da dojde do terminalna sostojba posle 400
    # iteracii, t.e. ne mozhe da se zavrsi vozenjeto. Se dobivaat mnogu losi rezultati. Epsilon ne pravi razlika.
