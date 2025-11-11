from k_means import *


data_file = r"C:\Users\malwa\Documents\GitHub Repos\Fall 2025 (Current)\cse4309\Assignment6\set2a.txt"
#data_file = "toy_data/set2a.txt"
#data_file = "toy_data/set2_1.txt"

K = 2
#initialization = "random"
initialization = "round_robin"


k_means(data_file, K, initialization)