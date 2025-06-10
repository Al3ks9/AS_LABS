import random
import pickle
from pettingzoo.classic import tictactoe_v3

Q1 = {}
Q2 = {}


def get_state(obs):
    return ''.join(map(str, obs['observation'].flatten()))


def choose_action(state, action_mask, epsilon, Q_table):
    valid_actions = [i for i, valid in enumerate(action_mask) if valid]
    if random.random() < epsilon:
        return random.choice(valid_actions)
    q_vals = [Q_table.get((state, a), 0.0) for a in valid_actions]
    max_q = max(q_vals)
    best_actions = [a for a, q in zip(valid_actions, q_vals) if q == max_q]
    return random.choice(best_actions)


def update_q_table(Q_table, state, action, reward, next_state, next_action_mask, done, alpha, gamma):
    if done:
        target = reward
    else:
        next_q_vals = [Q_table.get((next_state, a), 0.0) for a in range(9) if next_action_mask[a]]
        target = reward + gamma * max(next_q_vals, default=0.0)
    Q_table[(state, action)] = Q_table.get((state, action), 0.0) + alpha * (target - Q_table.get((state, action), 0.0))


def save_qtables():
    with open("qtable_player1.pkl", "wb") as f:
        pickle.dump(Q1, f)
    with open("qtable_player2.pkl", "wb") as f:
        pickle.dump(Q2, f)


def load_qtables():
    global Q1, Q2
    try:
        with open("qtable_player1.pkl", "rb") as f:
            Q1 = pickle.load(f)
        with open("qtable_player2.pkl", "rb") as f:
            Q2 = pickle.load(f)
        print("Q-tables loaded.")
    except FileNotFoundError:
        print("No saved Q-tables found, starting fresh.")


def train(episodes=1000):
    alpha = 0.1
    gamma = 0.9
    epsilon = 0.1
    results = {"player_1": 0, "player_2": 0, "draw": 0}

    for ep in range(episodes):
        env = tictactoe_v3.env(render_mode=None)
        env.reset(seed=random.randint(0, 10000))

        prev_states = {"player_1": None, "player_2": None}
        prev_actions = {"player_1": None, "player_2": None}
        final_rewards = {}

        for agent in env.agent_iter():
            obs, reward, terminated, truncated, _ = env.last()
            done = terminated or truncated
            state = get_state(obs)
            Q_table = Q1 if agent == "player_1" else Q2

            if done:
                if prev_states[agent] is not None:
                    update_q_table(Q_table, prev_states[agent], prev_actions[agent],
                                   reward, state, obs["action_mask"], True, alpha, gamma)
                action = None
            else:
                action = choose_action(state, obs["action_mask"], epsilon, Q_table)
                if prev_states[agent] is not None:
                    update_q_table(Q_table, prev_states[agent], prev_actions[agent],
                                   0, state, obs["action_mask"], False, alpha, gamma)
                prev_states[agent] = state
                prev_actions[agent] = action

            env.step(action)

            if done:
                final_rewards[agent] = reward

        if final_rewards.get("player_1", 0) == 1:
            results["player_1"] += 1
        elif final_rewards.get("player_2", 0) == 1:
            results["player_2"] += 1
        else:
            results["draw"] += 1

        if (ep + 1) % 100 == 0:
            print(
                f"Episode {ep + 1}: Wins - P1 = {results['player_1']}, P2 = {results['player_2']}, Draws = {results['draw']}")

    print("\nTraining complete.")
    print(f"Final results:\n{results}")
    save_qtables()


def test(render_episodes=5, render=False):
    load_qtables()
    print(f"\nShowing {render_episodes} test games:")

    results = {"player_1": 0, "player_2": 0, "draw": 0}

    for ep in range(render_episodes):
        env = tictactoe_v3.env(render_mode="human" if render else None)
        env.reset(seed=random.randint(0, 10000))
        final_rewards = {}

        for agent in env.agent_iter():
            obs, reward, terminated, truncated, _ = env.last()
            done = terminated or truncated
            state = get_state(obs)
            Q_table = Q1 if agent == "player_1" else Q2

            if done:
                action = None
            else:
                action = choose_action(state, obs["action_mask"], 0.0, Q_table)

            env.step(action)

            if done:
                final_rewards[agent] = reward

        env.close()

        if final_rewards.get("player_1", 0) == 1:
            results["player_1"] += 1
        elif final_rewards.get("player_2", 0) == 1:
            results["player_2"] += 1
        else:
            results["draw"] += 1

    print("\nTesting complete.")
    print(f"Test Results over {render_episodes} games:")
    print(f"Player 1 (Q1) wins: {results['player_1']}")
    print(f"Player 2 (Q2) wins: {results['player_2']}")
    print(f"Draws: {results['draw']}")


if __name__ == '__main__':
    train(episodes=1000)
    test(100)

# Testing complete.
# Test Results over 100 games:
# Player 1 (Q1) wins: 0
# Player 2 (Q2) wins: 0
# Draws: 100
