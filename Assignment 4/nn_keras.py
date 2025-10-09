import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

def nn_keras(directory, dataset, layers, units_per_layer, epochs):
    # Step 1: Loading the data
    # Need to:
    # - load training data into variables via np.loadtxt()
    training_data_path = os.path.join(directory, f"{dataset}_training.txt")
    test_data_path = os.path.join(directory, f"{dataset}_test.txt")

    raw_training_data = np.loadtxt(training_data_path, dtype = str)
    raw_test_data = np.loadtxt(test_data_path, dtype = str)

    X_train_raw = raw_training_data[:, :-1].astype(float)
    Y_train_raw = raw_training_data[:, -1]

    X_test_raw = raw_test_data[:, :-1].astype(float)
    Y_test_raw = raw_test_data[:, -1]

    # Step 2: Data preprocessing
    # Need to:
    # - be able to handle the string labels, mapping them from string to int
    # - normalize features from 0 to 1 based on max absolute value in training data

    #print("Training data shape:", X_train_raw.shape)
    #print("First training label:", Y_train_raw[0])

    unique_labels = np.unique(Y_train_raw)
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    int_to_label = {i: label for i, label in enumerate(unique_labels)}

    Y_train = np.array([label_to_int[label] for label in Y_train_raw])
    Y_test = np.array([label_to_int[label] for label in Y_test_raw])

    #print("Label mapping:", label_to_int)
    #print("First mapped training label:", Y_train[0])

    max_value = np.max(np.abs(X_train_raw))
    #print("Max absolute value in training data:", max_value)

    X_train_normalized = X_train_raw / max_value
    X_test_normalized = X_test_raw / max_value

    #print(f"First training feature vector (normalized): {X_train_normalized[0, 0]}")
    #print(f"First test feature vector (normalized): {X_test_normalized[0, 0]}")

    # Step 3: Building the model
    # Need to:
    # - create a Keras Sequential model
    # - add input and hidden layers in a loop
    # - add output layer

    model = keras.Sequential()
    num_of_features = X_train_normalized.shape[1]
    num_of_classes = len(unique_labels)

    if layers < 2:
        raise ValueError("Must have at least 2 layers (input layer + output layer).")
    
    if layers > 2:
        model.add(keras.layers.Dense(units_per_layer, input_dim=num_of_features, activation="sigmoid"))
    else:
        model.add(keras.layers.Dense(num_of_classes, input_dim=num_of_features, activation="sigmoid"))

    for _ in range(layers - 3):
        model.add(keras.layers.Dense(units_per_layer, activation="sigmoid"))

    if layers > 2:
        model.add(keras.layers.Dense(num_of_classes, activation="sigmoid"))

    # Step 4: Compile and train the model
    # Need to:
    # - compile model with 'adam' optimizer and with a specified loss
    # - train model with .fit()

    model.compile(optimizer="adam",
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])
    
    model.fit(X_train_normalized, Y_train, epochs=epochs, verbose=0)

    # Step 5: Evaluate and print results
    # Need to:
    # - evaluate model with .predict()
    # - implement the accuracy calculation as specified in the prompt
    # - print classification accuracy and results of each object

    predictions = model.predict(X_test_normalized, verbose=0)
    total_acc = 0.0

    for i in range(len(X_test_normalized)):
        object_id = i + 1
        true_class_int = Y_test[i]
        true_class = int_to_label[true_class_int]

        pred_probs = predictions[i]
        max_prob = np.max(pred_probs)

        tied_indices = np.where(pred_probs == max_prob)[0]
        predicted_class_int = np.random.choice(tied_indices)
        predicted_class = int_to_label[predicted_class_int]

        accuracy = 0.0
        if true_class_int in tied_indices:
            accuracy = 1.0 / len(tied_indices)
        
        total_acc += accuracy

        print('ID=%5d, predicted=%10s, true=%10s, accuracy=%4.2f' % (object_id, predicted_class, true_class, accuracy))

    classification_acc = total_acc / len(X_test_normalized)
    print('classification accuracy=%6.4f\n' % (classification_acc))
    return classification_acc

def nn_keras_loop(directory, dataset, layers, units_per_layer, epochs, repeats):
    accuracies = []
    for _ in range(repeats):
        classification_acc = nn_keras(directory, dataset, layers, units_per_layer, epochs)
        accuracies.append(classification_acc)  
    if accuracies:
        avg_accuracy = sum(accuracies) / len(accuracies)
        print('Average classification accuracy over %d runs: %6.4f\n' % (repeats, avg_accuracy))
    return