import numpy as np
import matplotlib.pyplot as plt


def linear_regression(training_file, test_file, degree, lambda1):
    # Loading training data
    training_data = np.loadtxt(training_file)
    X_training = training_data[:, :-1]
    t_training = training_data[:, -1]

    # Building design matrix and computing weights
    phi_training = generate_design_matrix(X_training, degree)
    identity_matrix = np.identity(phi_training.shape[1])
    w = np.linalg.pinv(lambda1 * identity_matrix + phi_training.T @ phi_training) @ phi_training.T @ t_training

    # Loading test data
    test_data = np.loadtxt(test_file)
    X_test = test_data[:, :-1]
    t_test = test_data[:, -1]
    phi_test = generate_design_matrix(X_test, degree)
    predictions = phi_test @ w
    RMSE = np.sqrt(np.mean((predictions - t_test))**2)

    for i in range(len(w)):
        print(f"w{i}={w[i]:.4f}")

    for i in range(len(predictions)):
        print(f"ID={i:5d}, output={predictions[i]:14.4f}, target value = {t_test[i]:10.4f}, squared error = {np.square(predictions[i] - t_test[i]):.4f}")

    #plot_results(X_training, t_training, w, degree)

def generate_design_matrix(X, degree):
    phi = np.ones((X.shape[0], 1))

    for j in range(X.shape[1]):
        for d in range(1, degree + 1):
            new_col = np.power(X[:, j], d).reshape(-1, 1)
            phi = np.hstack((phi, new_col))
    return phi

def plot_results(X, t, w, degree):
    # Plot training data
    plt.scatter(X, t, facecolors='none', edgecolors='b', label='Training Data')

    x_curve = np.linspace(min(X), max(X), 100).reshape(-1, 1)
    phi_curve = generate_design_matrix(x_curve, degree)
    y_curve = phi_curve @ w

    plt.plot(x_curve, y_curve, 'r', label=f'Degree {degree} Fit')
    plt.xlabel('X')
    plt.ylabel('t')
    plt.title('Polynomial Regression Results')
    plt.legend()
    plt.show()