alpha = 0.5
gamma = 0.5
convergence_threshold = 1e-5
max_iterations = 100000

rows, cols = 3, 3

terminal_states = {
    (0, 1): -80,
    (0, 2): 100,
    (2, 0): 25,
    (2, 1): -100,
    (2, 2): 80
}

non_terminal_states = [(0, 0), (1, 0), (1, 1), (1, 2)]

actions = ['up', 'down', 'left', 'right']
action_map = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1)
}

Q = {
    state: {a: 0.0 for a in actions}
    for state in non_terminal_states
}


def move(state, action):
    dx, dy = action_map[action]
    nx, ny = state[0] + dx, state[1] + dy
    if 0 <= nx < rows and 0 <= ny < cols:
        return (nx, ny)
    return state


for iteration in range(max_iterations):
    delta = 0
    for state in non_terminal_states:
        for action in actions:
            next_state = move(state, action)
            old_q = Q[state][action]

            if next_state in terminal_states:
                reward = terminal_states[next_state]
                target = reward
            else:
                reward = 0
                target = reward + gamma * max(Q[next_state].values())

            Q[state][action] += alpha * (target - old_q)
            delta = max(delta, abs(Q[state][action] - old_q))

    if delta < convergence_threshold:
        print(f"Converged after {iteration + 1} iterations.\n")
        break
else:
    print("Did not converge within max iterations.\n")

for state in non_terminal_states:
    print(f"State {state}:")
    for action in actions:
        print(f"  {action:5}: {Q[state][action]:.4f}")
    print()
