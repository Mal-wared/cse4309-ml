import numpy as np




def value_iteration(data_file, ntr, gamma, K):
    data = np.loadtxt(data_file, dtype=str, delimiter=',')
    
    rewards = np.zeros(data.shape)
    utils = np.zeros(data.shape)
    utils_new = np.zeros(data.shape)

    states = []
    terminal_states = set()
    for row, col in np.ndindex(data.shape):
        cell = data[row, col]

        if cell == 'X':
            continue

        states.append((row, col))
            
        if cell == '.':
            rewards[row, col] = ntr
        elif cell == '1.0' or cell == '-1.0':
            rewards[row, col] = float(cell)
            terminal_states.add((row, col))

    ACTIONS = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1)
    }

    SLIP_PROBS = [0.8, 0.1, 0.1] 
    SLIPS = {
        'up': ['left', 'right'],
        'down': ['left', 'right'],
        'left': ['up', 'down'],
        'right': ['up', 'down']
    }

    # Collision helper function
    def get_next_state(state, action):
        row, col = state
        dr, dc = ACTIONS[action] #delta_row, delta_column
        next_r, next_c = row + dr, col + dc

        if not (0 <= next_r < data.shape[0] and 0 <= next_c < data.shape[1]):
            return state # hit a boundary
        if data[next_r, next_c] == 'X':
            return state # hit an obstacle
        
        # if no obstacle or boundary hit, return next state
        return (next_r, next_c)
    
    def get_action_utility(state, action, current_utils):
        # intended next step
        s_prime_intended = get_next_state(state, action)
        u_intended = current_utils[s_prime_intended]

        # slip1 next step
        slip1_action = SLIPS[action][0]
        s_prime_slip1 = get_next_state(state, slip1_action)
        u_slip1 = current_utils[s_prime_slip1]

        # slip2 next step
        slip2_action = SLIPS[action][1]
        s_prime_slip2 = get_next_state(state, slip2_action)
        u_slip2 = current_utils[s_prime_slip2]

        return (SLIP_PROBS[0] * u_intended) + (SLIP_PROBS[1] * u_slip1) + (SLIP_PROBS[2] * u_slip2)

    # Main Iteration Loop (runs K times)
    for i in range(K):
        utils = utils_new.copy()

        for state in states:
            row, col = state

            # For terminal states, utility == reward
            if state in terminal_states:
                utils_new[row, col] = rewards[row, col]
            else:
                # Bellman Update application
                action_utilities = {}

                for action in ACTIONS:
                    expected_util = get_action_utility(state, action, utils)
                    action_utilities[action] = expected_util

                max_action_utility = max(action_utilities.values())

                utils_new[row, col] = rewards[row, col] + gamma * max_action_utility
        
    # Finding Optimal Policy
    policy = np.full(data.shape, 'x', dtype='<U6')

    ACTION_SYMBOLS = {
        'up': '^', 
        'down': 'v', 
        'left': '<', 
        'right': '>'
    }

    for state in states:
        row, col = state

        if state in terminal_states:
            policy[row, col] = 'o'
        else:
            action_utilities = {}
            for action in ACTIONS:
                expected_util = get_action_utility(state, action, utils_new) 
                action_utilities[action] = expected_util
            best_action = max(action_utilities, key=action_utilities.get)
            policy[row, col] = ACTION_SYMBOLS[best_action]

    print("utilities:")
    for row in utils_new:
        print(" ".join([f"{val:6.3f}" for val in row]))
        
    print("\npolicy:")
    for row in policy:
        print(" ".join([f"{val:6s}" for val in row]))

    # The function doesn't need to return anything, just print.
    return