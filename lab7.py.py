import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. Input and target
# -------------------------------
x = np.array([0.05, 0.10])

target = np.array([0.01, 0.99])

# -------------------------------
# 2. Initial weights
# -------------------------------
# Input -> Hidden
w1 = 0.15
w2 = 0.20
w3 = 0.25
w4 = 0.30

# Hidden -> Output
w5 = 0.40
w6 = 0.45
w7 = 0.50
w8 = 0.55

# Bias
b1 = 0.35
b2 = 0.60

# Learning rate
lr = 0.5

# Epochs
epochs = 20

# Store error
error_list = []


# -------------------------------
# 3. ReLU function
# -------------------------------
def relu(net):
    return max(0, net)


# -------------------------------
# 4. ReLU derivative
# -------------------------------
def relu_derivative(net):
    if net > 0:
        return 1
    else:
        return 0


# -------------------------------
# 5. Training using Backpropagation
# -------------------------------
for epoch in range(epochs):

    # ===== Forward Pass =====

    # Hidden layer net inputs
    net_h1 = x[0]*w1 + x[1]*w2 + b1
    net_h2 = x[0]*w3 + x[1]*w4 + b1

    # Hidden layer outputs
    h1 = relu(net_h1)
    h2 = relu(net_h2)

    # Output layer net inputs
    net_y1 = h1*w5 + h2*w6 + b2
    net_y2 = h1*w7 + h2*w8 + b2

    # Final outputs
    y1 = relu(net_y1)
    y2 = relu(net_y2)

    output = np.array([y1, y2])

    # ===== Error Calculation =====
    error = 0.5 * np.sum((target - output) ** 2)
    error_list.append(error)

    # ===== Output Layer Error =====
    delta_y1 = (y1 - target[0]) * relu_derivative(net_y1)
    delta_y2 = (y2 - target[1]) * relu_derivative(net_y2)

    # ===== Hidden Layer Error =====
    delta_h1 = (delta_y1*w5 + delta_y2*w7) * relu_derivative(net_h1)
    delta_h2 = (delta_y1*w6 + delta_y2*w8) * relu_derivative(net_h2)

    # ===== Update Hidden -> Output Weights =====
    w5 = w5 - lr * delta_y1 * h1
    w6 = w6 - lr * delta_y1 * h2
    w7 = w7 - lr * delta_y2 * h1
    w8 = w8 - lr * delta_y2 * h2

    # ===== Update Input -> Hidden Weights =====
    w1 = w1 - lr * delta_h1 * x[0]
    w2 = w2 - lr * delta_h1 * x[1]
    w3 = w3 - lr * delta_h2 * x[0]
    w4 = w4 - lr * delta_h2 * x[1]

    # Show some epochs
    if epoch == 0 or epoch == 1 or epoch == epochs - 1:
        print("Epoch:", epoch + 1)
        print("Output y1 =", round(y1, 6), " Output y2 =", round(y2, 6))
        print("Error =", round(error, 6))
        print("-----------------------------")


# -------------------------------
# 6. Final output
# -------------------------------
print("\nFinal Result:")
print("y1 =", round(y1, 6))
print("y2 =", round(y2, 6))
print("Target =", target)


# -------------------------------
# 7. Convergence Curve
# -------------------------------
plt.figure(figsize=(7, 4))
plt.plot(range(1, epochs + 1), error_list, 'g-o')
plt.xlabel("Epoch")
plt.ylabel("Total Error")
plt.title("Backpropagation with ReLU")
plt.grid(True)
plt.show()