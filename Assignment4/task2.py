import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.utils import plot_model
import visualkeras
from PIL import ImageFont

# --- 1. Define the Model ---
# The perceptron is a single neuron, which is a Dense layer with 1 unit.
# We use a custom lambda function for the step activation rule (output=1 if S>=0, else 0).
model = keras.Sequential([
    keras.layers.Dense(
        units=1, 
        input_shape=(3,), 
        activation=lambda x: tf.cast(x >= 0, dtype='float32')
    )
])

# --- 2. Set the Specific Weights and Bias ---
# Define the weights and bias from your Task 2 solution.
# The kernel (w1, w2, w3) must have shape (num_inputs, num_units).
weights = np.array([[1], [1], [1]])  # w1, w2, w3
bias = np.array([-2])              # w0

# Set these weights on the layer.
model.layers[0].set_weights([weights, bias])


# --- 3. Test the Perceptron ---
# Create some test data to verify the logic.
# Input 1: Two inputs are true (1+1-2=0 >= 0), should output 1.
test_input_1 = np.array([[0, 0, 0]]) 

# Input 2: One input is true (1+0-2=-1 < 0), should output 0.
test_input_2 = np.array([[0, 0, 1]])
test_input_3 = np.array([[0, 1, 0]])
test_input_4 = np.array([[0, 1, 1]])
test_input_5 = np.array([[1, 0, 0]]) 
test_input_6 = np.array([[1, 0, 1]]) 
test_input_7 = np.array([[1, 1, 0]]) 
test_input_8 = np.array([[1, 1, 1]]) 

# Get predictions
prediction_1 = model.predict(test_input_1)
prediction_2 = model.predict(test_input_2)
prediction_3 = model.predict(test_input_3)
prediction_4 = model.predict(test_input_4)
prediction_5 = model.predict(test_input_5)
prediction_6 = model.predict(test_input_6)
prediction_7 = model.predict(test_input_7)
prediction_8 = model.predict(test_input_8)

print(f"Input: {test_input_1} -> Prediction: {prediction_1[0][0]}")
print(f"Input: {test_input_2} -> Prediction: {prediction_2[0][0]}")
print(f"Input: {test_input_3} -> Prediction: {prediction_3[0][0]}")
print(f"Input: {test_input_4} -> Prediction: {prediction_4[0][0]}")
print(f"Input: {test_input_5} -> Prediction: {prediction_5[0][0]}")
print(f"Input: {test_input_6} -> Prediction: {prediction_6[0][0]}")
print(f"Input: {test_input_7} -> Prediction: {prediction_7[0][0]}")
print(f"Input: {test_input_8} -> Prediction: {prediction_8[0][0]}")

#plot_model(model, to_file='perceptron_diagram.png', show_shapes=True, show_layer_names=True)
font = ImageFont.truetype("arial.ttf", 12) # Or another font you have
visualkeras.layered_view(model, legend=True, font=font, to_file='perceptron_detailed.png') 

print("\nDiagram saved as 'perceptron_diagram.png'")

# You can also inspect the model's weights to confirm they are set correctly.
# print("\nModel Weights:\n", model.layers[0].get_weights())