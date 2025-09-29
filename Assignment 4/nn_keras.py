import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

def nn_keras(directory, dataset, layers, units_per_layer, epochs):
    # --- 1. DATA LOADING ---
    # Construct file paths for training and test data
    # Load training data (features and labels)
    # Load test data (features and labels)
    training_data_path = os.path.join(directory, f"{dataset}_training.txt")
    test_data_path = os.path.join(directory, f"{dataset}_test.txt")

    raw_training_data = np.loadtxt(training_data_path, dtype = str)
    raw_test_data = np.loadtxt(test_data_path, dtype = str)

    X_train_raw = raw_training_data[:, :-1].astype(float)
    Y_train_raw = raw_training_data[:, -1]

    X_test_raw = raw_test_data[:, :-1].astype(float)
    Y_test_raw = raw_test_data[:, -1]

    print("Training data shape:", X_train_raw.shape)
    print("First training label:", Y_train_raw[0])

    # --- 2. DATA PREPROCESSING ---
    # Handle string labels: create a mapping from string to integer
    # Apply the mapping to training and test labels
    # Normalize features: find the max absolute value in training data
    # Divide all training and test features by that max value

    unique_labels = np.unique(Y_train_raw)
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    int_to_label = {i: label for i, label in enumerate(unique_labels)}

    Y_train = np.array([label_to_int[label] for label in Y_train_raw])
    Y_test = np.array([label_to_int[label] for label in Y_test_raw])

    print("Label mapping:", label_to_int)
    print("First mapped training label:", Y_train[0])

    max_value = np.max(np.abs(X_train_raw))
    print("Max absolute value in training data:", max_value)

    X_train_normalized = X_train_raw / max_value
    X_test_normalized = X_test_raw / max_value

    print(f"First training feature vector (normalized): {X_train_normalized[0, 0]}")
    print(f"First test feature vector (normalized): {X_test_normalized[0, 0]}")

    # --- 3. MODEL BUILDING ---
    # Create a Keras Sequential model
    # Add input/hidden layers in a loop
    # Add the output layer

    # --- 4. COMPILE AND TRAIN ---
    # Compile the model with 'adam' optimizer and the specified loss
    # Train the model using .fit()

    # --- 5. EVALUATE AND REPORT ---
    # Make predictions on the test set
    # Loop through predictions to calculate accuracy based on the prompt's rules
    # Print the result for each test object
    # Print the final classification accuracy

    # For now, just print the arguments to make sure it's working
    print(f"Starting with dataset: {dataset}")
    return

# load the data

# preprocess it
# build the model