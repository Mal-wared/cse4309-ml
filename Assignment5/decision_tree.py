import numpy as np
import math
import random
from collections import deque

class Node:
    def __init__(self, node_id = 1):
        # General tree structure
        self.node_id = node_id
        self.left_child = None
        self.right_child = None

        # Splitting criteria
        self.feature_id = -1
        self.threshold = -1.0
        self.gain = 0.0

        # Leaf node value
        self.predicted_class = None

def calculate_entropy(labels):
    # Getting all unique classes + their counts
    _, counts = np.unique(labels, return_counts=True)

    # Calculating probabilities
    probabilities = counts / len(labels)

    # H(S) = -sum(p(count) * log2(p(count))
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def calculate_info_gain(parent_labels, left_labels, right_labels):
    parent_entropy = calculate_entropy(parent_labels)

    weight_left = len(left_labels) / len(parent_labels)
    weight_right = len(right_labels) / len(parent_labels)

    entropy_left = calculate_entropy(left_labels) if len(left_labels) > 0 else 0
    entropy_right = calculate_entropy(right_labels) if len(right_labels) > 0 else 0

    weighted_child_entropy = (weight_left * entropy_left) + (weight_right * entropy_right)

    # Gain = H(parent) - Weighted_H(children)
    info_gain = parent_entropy - weighted_child_entropy
    return info_gain

def find_best_split(data, labels, option):
    '''
    Finds best feature and threshold to split onto
    'option' determines search strategy
    - "optimized": checks all features
    - integer: randomly selects a feature
    '''
    best_gain = -1.0
    best_feature = -1
    best_threshold = -1.0

    num_features = data.shape[1]

    if option == "optimized":
        features_to_check = range(num_features)
    else:
        features_to_check = [random.randint(0, num_features - 1)]
    
    for feature_index in features_to_check:
        thresholds = np.unique(data[:, feature_index])

        for threshold in thresholds:
            left_mask = data[:, feature_index] < threshold
            right_mask = data[:, feature_index] >= threshold

            left_labels = labels[left_mask]
            right_labels = labels[right_mask]

            if len(left_labels) == 0 or len(right_labels) == 0:
                continue

            gain = calculate_info_gain(labels, left_labels, right_labels)

            if gain > best_gain:
                best_gain = gain
                best_feature = feature_index
                best_threshold = threshold
    return best_feature + 1, best_threshold, best_gain

def build_tree(data, labels, node_id, pruning_thr, option):
    node = Node(node_id)

    # Step 1: Base cases

    # Case 1: All labels are the same (pure node)
    unique_labels = np.unique(labels)
    if len(unique_labels) == 1:
        node.predicted_class = unique_labels[0]
        return node
    
    # Case 2: # of samples < pruning threshold
    if len(labels) < pruning_thr:
        classes, counts = np.unique(labels, return_counts = True)
        node.predicted_class = classes[np.argmax(counts)]
    
    # Step 2: Find best split
    feature_id_1based, threshold, gain = find_best_split(data, labels, option)

    # Case 3: No good split found (gain is an unacceptable value)
    if gain <= 0:
        classes, counts = np.unique(labels, return_counts=True)
        node.predicted_class = classes[np.argmax(counts)]
        return node
    
    # Step 3: Recursive step 
    node.feature_id = feature_id_1based
    node.threshold = threshold
    node.gain = gain

    feature_id_0based = feature_id_1based - 1

    # Split data and labels for children
    left_mask = data[:, feature_id_0based] < threshold
    right_mask = data[:, feature_id_0based] >= threshold
    
    left_data = data[left_mask]
    left_labels = labels[left_mask]

    right_data = data[right_mask]
    right_labels = labels[right_mask]

    node.left_child = build_tree(left_data, left_labels, 2 * node_id, pruning_thr, option)
    node.right_child = build_tree(right_data, right_labels, (2 * node_id) + 1, pruning_thr, option)

    return node


def decision_tree(training_file, test_file, option, pruning_thr):
    # Step 1: Loading the data
    # - load training and test data into variables via np.loadtxt()
    # - separating features and labels via substring methods
    raw_training_data = np.loadtxt(training_file, dtype = str)
    raw_test_data = np.loadtxt(test_file, dtype = str)

    X_train_raw = raw_training_data[:, :-1].astype(float)
    Y_train_raw = raw_training_data[:, -1]

    X_test_raw = raw_test_data[:, :-1].astype(float)
    Y_test_raw = raw_test_data[:, -1]

    # Step 2: Training Stage
    # - Building tree 
    forest = []

    if option == "optimized":
        num_trees = 1
    else:
        num_trees = int(option)
    
    for i in range(1, num_trees + 1):
        root_node = build_tree(X_train_raw, Y_train_raw, 1, pruning_thr, option)
        forest.append((root_node, i))

    # Step 3: Training Output (BFS)
    for root_node, tree_id in forest:
        if root_node is None:
            continue
            
        # use queue for BFS
        queue = deque([root_node])
        while queue:
            current_node = queue.popleft()
            print('tree=%2d, node=%3d, feature=%2d, thr=%6.2f, gain=%f' % 
                  (tree_id, 
                   current_node.node_id, 
                   current_node.feature_id, 
                   current_node.threshold, 
                   current_node.gain))
            
            if current_node.left_child:
                queue.append(current_node.left_child)
            
            if current_node.right_child:
                queue.append(current_node.right_child)
                
    return

