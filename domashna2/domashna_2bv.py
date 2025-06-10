def compute_q_values():
    Q = {
        ("S", "R"): 0.0,
        ("A", "E"): 0.0
    }

    brojach = 0
    while True:
        old_Q = Q.copy()
        # 1
        Q[('S', 'R')] = 0.5 * old_Q[('S', 'R')] + 0.5 * old_Q[('A', 'E')]
        Q[("A", "E")] = 0.5 * old_Q[('A', 'E')] + 0.5

        print(brojach, '1')
        for key in Q:
            print(f"Q{key} = {Q[key]:.4f}")
        # 2
        Q[('S', 'R')] = 0.5 * Q[('S', 'R')] + 0.5 * Q[('A', 'E')]
        Q[("A", "E")] = 0.5 * Q[('A', 'E')] + 5

        print(brojach, '2')
        for key in Q:
            print(f"Q{key} = {Q[key]:.4f}")
        # 3
        # comment next two lines to solve for 2b
        Q[('S', 'R')] = 0.5 * Q[('S', 'R')] + 0.5 * Q[('A', 'E')]
        Q[("A", "E")] = 0.5 * Q[('A', 'E')] + 5

        print(brojach, '3')
        for key in Q:
            print(f"Q{key} = {Q[key]:.4f}")

        if Q[('A', 'E')] == old_Q[('A', 'E')] and Q[('S', 'R')] == old_Q[('S', 'R')]:
            break

        brojach += 1

    print('iteracii ' + str(brojach))
    # Print final Q-values
    for key in Q:
        print(f"Q{key} = {Q[key]:.4f}")


compute_q_values()
