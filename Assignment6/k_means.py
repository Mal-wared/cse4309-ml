# Author:   Nicholas Nhat Tran
# ID:       1002027150

import numpy as np
from drawing import draw_assignments
import matplotlib.pyplot as plt

def k_means(data_file, K, initialization):
    # Step 1: Reading and Parsing Data
    data = np.loadtxt(data_file, dtype=float)
    
    N = data.shape[0]
    D = data.shape[1] if data.ndim > 1 else 1 
    
    if D == 1:
        data = data.reshape((N, 1))
        D = 1 
    
    # Step 2: Initializing Cluster Assignments
    assignments = np.zeros(N, dtype=int)

    if initialization == "round_robin":
        for i in range(N):
            assignments[i] = (i % K) + 1
    elif initialization == "random":
        assignments = np.random.randint(1, K + 1, size=N)
    else:
        print("Error: Invalid initialization method")
        return
    
    # Step 3: Main K-Means Loop
    while True:
        old_assignments = np.copy(assignments)
        centroids = np.zeros((K, D))

        for k in range(1, K + 1):
            pts_in_cluster = data[assignments == k]

            if pts_in_cluster.shape[0] > 0:
                centroids[k - 1] = np.mean(pts_in_cluster, axis=0)

        for i in range(N):
            pt = data[i]
            distances = []

            for k in range(K):
                dist = np.sum((pt - centroids[k])**2)
                distances.append(dist)
            
            closest_cluster_idx = np.argmin(distances)
            assignments[i] = closest_cluster_idx + 1

        if np.array_equal(assignments, old_assignments):
            break
            
    # Step 4: Output

    
    if D == 1: # Print for 1D data
        for i in range(N):
            print('%10.4f --> cluster %d' % (data[i, 0], assignments[i]))
    elif D == 2: # Print for 2D data
        for i in range(N):
            print('(%10.4f, %10.4f) --> cluster %d' % (data[i, 0], data[i, 1], assignments[i]))

    data_for_plotting = data.T
    draw_assignments(data_for_plotting, assignments)
    plt.show()

    return