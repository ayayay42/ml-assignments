import numpy as np

print("########## question 1 ##########")

# Given parameters
W_x_to_h = np.array([[0.78, 1.27], [0.88, 1.15]])
b_h = np.array([0.9, -1.85])
w_h_to_y = np.array([0.75, -2.15])
b_y = -0.35
# True labels
Y = np.array([0, 1, 1, 0])

# Activation functions
def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Input data points
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Forward propagation for each input
outputs = []
for x in X:
    # Compute hidden layer activations: h = ReLU(W_x_to_h * x + b_h)
    h = relu(np.dot(W_x_to_h.T, x) + b_h)
    
    # Compute output: y = sigmoid(w_h_to_y * h + b_y)
    y = sigmoid(np.dot(w_h_to_y, h) + b_y)
    
    outputs.append(y)

# Format results
print(np.array(outputs))

print("########## question 3 ##########")

actual = np.array([0, 1, 1, 0])  # True labels
predicted = np.array([0.581, 0.728, 0.713, 0.585])  # Predicted probabilities

cross_entropy = -(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted))
total_loss = cross_entropy.sum()
print(f"Total Binary Cross-Entropy: {total_loss:.4f}")

print("########## question 4 ##########")

# Compute output layer gradient for x1, y1
grad_yhat_J = outputs[0] - Y[0]

print(grad_yhat_J)

print("########## question 5 ##########")

h1 = np.array([0.9, 0])  # Hidden layer activation for x1
y1 = 0
y_hat1 = grad_yhat_J

grad_yhat_J2 = y_hat1 - y1

# Compute weight gradients
grad_w = grad_yhat_J2 * h1

# Compute bias gradient
grad_b = grad_yhat_J2

# Print results
print(f"{grad_w[0]:.3f}, {grad_w[1]:.3f}, {grad_b:.3f}")

print("########## question 6 ##########")
grad_yhat_J = 0.581
w_h_to_y = np.array([0.75, -2.15])

# Compute gradient w.r.t. hidden activations
grad_h = grad_yhat_J * w_h_to_y

# Print formatted result
print(f"{grad_h[0]:.3f}, {grad_h[1]:.3f}")

print("########## question 7 ##########")

import numpy as np

# Inputs and weights
x1 = np.array([0, 0])
w_h_to_y = np.array([0.75, -2.15])
b_h = np.array([0.9, -1.85])

# Gradient from output layer
grad_yhat_J = 0.581

# Hidden layer pre-activation z = W*x + b (x=0 → z = b_h)
z = b_h
relu_derivative = (z > 0).astype(float)

# Gradient w.r.t. h
grad_h = grad_yhat_J * w_h_to_y

# Backprop through ReLU
delta = grad_h * relu_derivative

# Gradients for W (outer product of delta and x1) and b
grad_W = np.outer(x1, delta).T  # Shape (2,2)
grad_b = delta

# Print in required format
print(f"{grad_W[0][0]:.3f}, {grad_W[0][1]:.3f}, {grad_W[1][0]:.3f}, {grad_W[1][1]:.3f}, {grad_b[0]:.3f}, {grad_b[1]:.3f}")

print("########## question 8 ##########")
# Initial parameters
w = np.array([0.75, -2.15])
b = -0.35

# Gradients from previous steps
grad_w = np.array([0.581 * 0.9, 0])  # [0.5229, 0]
grad_b = 0.581

# Learning rate
eta = 0.1

# Parameter update
w_new = w - eta * grad_w
b_new = b - eta * grad_b

# Print new weights and bias
print(f"{w_new[0]:.3f}, {w_new[1]:.3f}, {b_new:.3f}")

print("########## question 9 ##########")

import numpy as np

# Initial parameters
W = np.array([[0.4, 0.2], [-0.3, 0.1]])
b_h = np.array([0.9, -1.85])

# Gradients
grad_W = np.zeros_like(W)  # Since x = [0, 0]
grad_b = np.array([0.43575, 0])  # From previous step

# Learning rate
eta = 0.1

# Updated parameters
W_new = W - eta * grad_W  # No change
b_new = b_h - eta * grad_b

# Print results in required format
print(f"{W_new[0,0]:.3f}, {W_new[0,1]:.3f}, {W_new[1,0]:.3f}, {W_new[1,1]:.3f}, {b_new[0]:.3f}, {b_new[1]:.3f}")
