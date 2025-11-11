from value_iteration import value_iteration

# When you test your code, you can select the function arguments you 
# want to use by modifying the next lines

data_file = r"C:\Users\malwa\Documents\GitHub Repos\Fall 2025 (Current)\cse4309\7assignment\environment2.txt"
#data_file = r"C:\Users\malwa\Documents\GitHub Repos\Fall 2025 (Current)\cse4309\7assignment\environment2.txt"
ntr = -0.04 # non_terminal_reward
gamma = 0.9
K = 20


value_iteration(data_file, ntr, gamma, K)
