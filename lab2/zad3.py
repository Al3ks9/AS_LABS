import gymnasium as gym
from q_learning import *


def get_best_action(q_table, state_p, state_v):
    return np.argmax(q_table[state_p, state_v])


def get_action(env, q_table, state_p, state_v, epsilon):
    num_actions = env.action_space.n
    probability = np.random.random() + epsilon / num_actions
    if probability < epsilon:
        return get_random_action(env)
    else:
        return get_best_action(q_table, state_p, state_v)


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')

    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)
    velocity_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)

    q_table = random_q_table(-1, 0, (len(pos_space), len(velocity_space), 3))

    learning_rate = 0.1
    discount_factor = 0.9
    epsilon = .8
    num_episodes = 100

    avg_iter = 0
    avg_reward = 0
    for episode in range(num_episodes):
        state = env.reset()[0]
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], velocity_space)
        num_iter = 0
        while True:
            num_iter += 1
            # action = get_random_action(env)  # 1
            # action = get_best_action(q_table, state)  # 2
            action = get_action(env, q_table, state_p, state_v, epsilon)

            new_state, reward, terminated, _, _ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], velocity_space)

            q_table[state_p, state_v, action] = q_table[state_p, state_v, action] + learning_rate * (
                        reward + discount_factor * np.max(q_table[new_state_p, new_state_v]) - q_table[
                    state_p, state_v, action])

            state_p = new_state_p
            state_v = new_state_v
            if terminated or num_iter > 1000:
                break
        # print(f'End of {episode}')
        avg_iter += num_iter
        avg_reward += reward

    print(avg_iter / 100, avg_reward / 100)

    avg_iter = 0
    avg_reward = 0
    for episode in range(num_episodes):
        state = env.reset()[0]
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], velocity_space)
        num_iter = 0
        while True:
            num_iter += 1
            action = get_best_action(q_table, state_p, state_v)  # 2

            new_state, reward, terminated, _, _ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], velocity_space)

            state_p = new_state_p
            state_v = new_state_v
            if terminated or num_iter > 1000:
                break
        avg_iter += num_iter
        avg_reward += reward

    print(avg_iter / 100, avg_reward / 100)

    # Q-learning raboti mnogu dobro ovde, no samo ako epsilon e visoka vrednost.
    # Ako se stavi epsilon na niska vrednost, algoritmot ne uspeva da nauci kako da go resi problemot.
    # Samo beshe potrebno editiranje na  funkciite za naogjanje na nova vrednost za q-tabelata.
    # Ovde goleminata na q tabelata e 20x20x3, bidejki ima 20 razlicni vrednosti za pozicijata i brzina na kolata i 3 akcii.
    # 20 e izbrana vrednost, mozhe da bide ovekje ili pomalku, zavisno na kolku delovi sakamne da go podelime prostorot.