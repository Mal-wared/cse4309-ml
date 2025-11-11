import numpy as np
import random
from collections import Counter

def euclidean_distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2)**2))

def knn_classify(training_file, test_file, k):
    # Step 1: Data loading
    raw_training_data = np.loadtxt(training_file, dtype = str)
    raw_test_data = np.loadtxt(test_file, dtype = str)

    X_train_raw = raw_training_data[:, :-1].astype(float)
    Y_train_raw = raw_training_data[:, -1]

    X_test_raw = raw_test_data[:, :-1].astype(float)
    Y_test_raw = raw_test_data[:, -1]

    # Step 2: Feature normalization
    mean = np.mean(X_train_raw, axis=0)
    std = np.std(X_train_raw, axis=0, ddof=1)
    std[std == 0] = 1.0

    X_train_norm = (X_train_raw - mean) / std
    X_test_norm = (X_test_raw - mean) / std

    # Step 3: Classification Stage
    total_acc = 0.0
    object_id = 1

    for i in range(len(X_test_norm)):
        test_object = X_test_norm[i]
        true_class = Y_test_raw[i]

        distances = []
        for j in range(len(X_train_norm)):
            train_object = X_train_norm[j]
            train_label = Y_train_raw[j]
            dist = euclidean_distance(test_object, train_object)
            distances.append((dist, train_label))

        # Find nearest k neighbor
        distances.sort(key=lambda x : x[0])
        neighbors = distances[0:k]
        neighbor_labels = [label for dist, label in neighbors]

        # Vote for prediction
        votes = Counter(neighbor_labels)
        max_votes = max(votes.values())
        tied_classes = [cls for cls, count in votes.items() if count == max_votes]
        
        # Final prediction by random ties
        predicted_class = random.choice(tied_classes)

        accuracy = 0.0
        if true_class in tied_classes:
            accuracy = 1.0 / len(tied_classes)

        total_acc += accuracy

        print('ID=%5d, predicted=%10s, true=%10s, accuracy=%4.2f' % 
              (object_id, predicted_class, true_class, accuracy))
        
        object_id += 1

    classification_accuracy = total_acc / len(Y_test_raw)
    print('classification accuracy=%6.4f' % (classification_accuracy))

    return
