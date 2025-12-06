import sys
import csv
import numpy as np

def read_env(file_name):
    # parses environment file into variables 
    data = np.loadtxt(file_name, delimiter=',', dtype=str)

    rows = len(data)
    cols = len(data[0])

    rewards = {}
    walls = set()
    start_state = None

    for r in range(rows):
        for c in range(cols):
            cell = data[r][c].strip()

            if cell == 'X':
                walls.add((r, c))
            elif cell == "I":
                start_state = (r, c)
            elif cell != '.':
                try:
                    rewards[(r, c)] = float(cell)
                except ValueError:
                    pass
    return rows, cols, walls, rewards, start_state

def explore(u, n, Ne):
    # either returns an optimistic value if the pair has been visited less than ne times
    # else return the actual utility
    if n < Ne:
        return 1
    else:
        return u
    

def AgentModel_Q_Learning(environment_file, ntr, gamma, number_of_moves, Ne):
    rows, cols, walls, rewards, start_state = read_env(environment_file)

    # utility & frequency tables
    Q = {}
    N = {}

    ACTIONS = ['^', '>', 'v', '<']
    DIRECTIONS = {
        '^': (-1, 0),
        '>': (0, 1),
        'v': (1, 0),
        '<': (0, -1)
    }

    s = start_state

    # actual main learning loop
    for move in range(number_of_moves):
        # if terminal state, reset to start
        if s in rewards:
            s = start_state
            continue
        else:
            best_action = None
            max_val = -float('inf')

            # calculations for actions to decide on where to land
            for action in ACTIONS:
                u = Q.get((s, action), 0.0)
                n = N.get((s, action), 0)
                val = explore(u, n, Ne)

                if val > max_val:
                    max_val = val 
                    best_action = action
                    tied_actions = [action]
                elif val == max_val:
                    tied_actions.append(action)

            # tiebreaker
            if len(tied_actions) > 0:
                best_action = np.random.choice(tied_actions)

            # Increment frequency count
            current_n = N.get((s, best_action), 0)
            N[(s, best_action)] = current_n + 1

            # action transitions based on chance
            r = np.random.rand()
            intended_index = ACTIONS.index(best_action)

            if r < 0.8:
                actual_action = best_action
            elif r < 0.9:
                # subtract 1 from the index to rotate 90 degrees
                left_index = (intended_index - 1) % 4
                actual_action = ACTIONS[left_index]
            else:
                right_index = (intended_index + 1) % 4
                actual_action = ACTIONS[right_index]

            # actually moving the agent here, calculating next state based on movement
            dr, dc = DIRECTIONS[actual_action]
            next_r = s[0] + dr
            next_c = s[1] + dc

            # boundary/wall check
            if (0 <= next_r < rows) and (0 <= next_c < cols) and ((next_r, next_c) not in walls):
                next_state = (next_r, next_c)
            else:
                next_state = s

            if next_state in rewards:
                r_prime = ntr
                max_q_prime = rewards[next_state]
            else:
                r_prime = ntr

                q_vals_next = []
                for action in ACTIONS:
                    q_vals_next.append(Q.get((next_state, action), 0.0))
                max_q_prime = max(q_vals_next)

            # learning rate: alpha decays as visits increment
            alpha = 20.0 / (19.0 + (current_n + 1))
            old_q = Q.get((s, best_action), 0.0)

            # bellman equation
            new_q = old_q + alpha * (r_prime + (gamma * max_q_prime) - old_q)
            Q[(s, best_action)] = new_q
            s = next_state

    print("utilities:")
    for r in range(rows):
        row_vals = []
        for c in range(cols):
            check_s = (r, c)
            if check_s in walls:
                u = 0.0
            elif check_s in rewards:
                u = rewards[check_s]
            else: 
                q_values = [Q.get(((r, c), action), 0.0) for action in ACTIONS]
                u = max(q_values)
            row_vals.append(u)
        print(" ".join(["%6.3f" % val for val in row_vals]))
    
    print("\npolicy:")
    for r in range(rows):
        row_chars = []
        for c in range(cols):
            check_s = (r, c)
            if check_s in walls:
                char = 'x'
            elif check_s in rewards:
                char = 'o'
            else:
                # finding action w/ the highest Q-val
                best_action = None
                max_q = -float('inf')
                
                for action in ACTIONS:
                    val = Q.get((check_s, action), 0.0)
                    if val > max_q:
                        max_q = val
                        best_action = action
                char = best_action
            row_chars.append(char)
        
        # Print the row formatted as strings
        print(" ".join(["%6s" % val for val in row_chars]))
    return