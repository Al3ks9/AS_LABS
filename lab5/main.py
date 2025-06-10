import random
from pettingzoo.classic import tictactoe_v3
import pickle

Q = {}


def get_state(obs):
    return ''.join(map(str, obs['observation'].flatten()))


def choose_action(state, action_mask, epsilon):
    valid_actions = [i for i, valid in enumerate(action_mask) if valid]
    if random.random() < epsilon:
        return random.choice(valid_actions)
    q_vals = [Q.get((state, a), 0.0) for a in valid_actions]
    max_q = max(q_vals)
    best_actions = [a for a, q in zip(valid_actions, q_vals) if q == max_q]
    return random.choice(best_actions)


def update_q_table(state, action, reward, next_state, next_action_mask, done, alpha, gamma):
    if done:
        target = reward
    else:
        next_q_vals = [Q.get((next_state, a), 0.0) for a in range(9) if next_action_mask[a]]
        target = reward + gamma * max(next_q_vals, default=0.0)
    Q[(state, action)] = Q.get((state, action), 0.0) + alpha * (target - Q.get((state, action), 0.0))


def save_q_table(filename='q_table.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(Q, f)
    print(f"Q-table saved to {filename}")


def load_q_table(filename='q_table.pkl'):
    global Q
    try:
        with open(filename, 'rb') as f:
            Q = pickle.load(f)
        print(f"Q-table loaded from {filename}")
    except FileNotFoundError:
        print(f"Q-table file '{filename}' not found. Starting with empty table.")


def train(episodes=100):
    alpha = 0.1
    gamma = 0.9
    epsilon = 0.1
    results = {"player_1": 0, "player_2": 0, "draw": 0}

    for ep in range(episodes):
        env = tictactoe_v3.env(render_mode=None)
        env.reset(seed=random.randint(0, 10000))

        prev_state = None
        prev_action = None
        final_rewards = {"player_1": 0, "player_2": 0}

        for agent in env.agent_iter():
            obs, reward, terminated, truncated, info = env.last()
            done = terminated or truncated
            state = get_state(obs)

            if agent == "player_1":
                if done:
                    if prev_state is not None:
                        update_q_table(prev_state, prev_action, reward, state, obs["action_mask"], True, alpha, gamma)
                    action = None
                else:
                    action = choose_action(state, obs["action_mask"], epsilon)
                    if prev_state is not None:
                        update_q_table(prev_state, prev_action, 0, state, obs["action_mask"], False, alpha, gamma)

                    prev_state = state
                    prev_action = action
            else:
                action = None if done else env.action_space(agent).sample(obs["action_mask"])

            env.step(action)

            if done:
                final_rewards[agent] = reward

        if final_rewards["player_1"] == 1:
            results["player_1"] += 1
        elif final_rewards["player_2"] == 1:
            results["player_2"] += 1
        else:
            results["draw"] += 1

        if (ep + 1) % 10 == 0:
            print(
                f"Episode {ep + 1}: Wins - Player 1 (Q) = {results['player_1']}, Player 2 (Random) = {results['player_2']}, Draws = {results['draw']}")

    print("\nTraining complete after 100 episodes.")
    print(f"Final results:\n{results}")
    save_q_table()


def test(episodes=5, render=None):
    print("\n🎮 Running test episodes\n")
    results = {"player_1": 0, "player_2": 0, "draw": 0}

    for ep in range(episodes):
        env = tictactoe_v3.env(render_mode="human" if render else None)
        env.reset(seed=random.randint(0, 10000))
        print(f"\n🔹 Episode {ep + 1}")

        final_rewards = {"player_1": 0, "player_2": 0}

        for agent in env.agent_iter():
            obs, reward, terminated, truncated, info = env.last()
            done = terminated or truncated
            state = get_state(obs)

            if agent == "player_1":
                action = None if done else choose_action(state, obs["action_mask"], epsilon=0.0)
            else:
                action = None if done else env.action_space(agent).sample(obs["action_mask"])

            env.step(action)

            if done:
                final_rewards[agent] = reward

        if final_rewards["player_1"] == 1:
            results["player_1"] += 1
            print("Player 1 (Q Agent) wins")
        elif final_rewards["player_2"] == 1:
            results["player_2"] += 1
            print("Player 2 (Random) wins")
        else:
            results["draw"] += 1
            print("It's a draw")

        env.close()

    print("\nSummary after test episodes:")
    print(f"Player 1 (Q Agent) wins: {results['player_1']}")
    print(f"Player 2 (Random) wins: {results['player_2']}")
    print(f"Draws: {results['draw']}")


if __name__ == '__main__':
    # train()
    load_q_table()
    # test()
    test(100, False)
