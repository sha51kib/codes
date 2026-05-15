import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. 5x5 digit images
# Gray pixel = 1, White pixel = 0
# --------------------------------------------------

digit_1 = np.array([
    [0, 1, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 1, 1, 1, 0]
])

digit_2 = np.array([
    [1, 1, 1, 1, 1],
    [0, 0, 0, 0, 1],
    [0, 1, 1, 1, 1],
    [1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1]
])

digit_3 = np.array([
    [1, 1, 1, 1, 1],
    [0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1],
    [0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1]
])
digit_4 = np.array([
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1],
    [0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1]
])


digit_5 = np.array([
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1],
    [0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1]
])

digits = [digit_1, digit_2, digit_3, digit_4, digit_5]


# --------------------------------------------------
# 2. Show input digit images
# --------------------------------------------------

plt.figure(figsize=(12, 3))

for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(digits[i], cmap="gray_r", interpolation="nearest")
    plt.title("Digit " + str(i + 1))
    
    plt.xticks(np.arange(-0.5, 5, 1), [])
    plt.yticks(np.arange(-0.5, 5, 1), [])
    plt.grid(color="black", linewidth=1)
    
    plt.xlim(-0.5, 4.5)
    plt.ylim(4.5, -0.5)

plt.tight_layout()
plt.show()


# --------------------------------------------------
# 3. Convert every 5x5 image into 25 input values
# --------------------------------------------------

X = np.array([
    digit_1.flatten(),
    digit_2.flatten(),
    digit_3.flatten(),
    digit_4.flatten(),
    digit_5.flatten()
])


# --------------------------------------------------
# 4. Target outputs using one-hot encoding
# --------------------------------------------------

D = np.array([
    [1, 0, 0, 0, 0],   # Digit 1
    [0, 1, 0, 0, 0],   # Digit 2
    [0, 0, 1, 0, 0],   # Digit 3
    [0, 0, 0, 1, 0],   # Digit 4
    [0, 0, 0, 0, 1]    # Digit 5
])


# --------------------------------------------------
# 5. Add bias input
# --------------------------------------------------

bias = np.ones((5, 1))
X = np.hstack((X, bias))

# Now input size = 26
# 25 pixel values + 1 bias


# --------------------------------------------------
# 6. Initial weights
# 26 inputs -> 5 output neurons
# --------------------------------------------------

W = np.zeros((26, 5))

lr = 0.1
epochs = 20

error_list = []


# --------------------------------------------------
# 7. Training using Delta Learning Rule
# --------------------------------------------------

for epoch in range(epochs):
    total_error = 0

    for i in range(len(X)):
        x = X[i]
        target = D[i]

        # Network output
        y = np.dot(x, W)

        # Error
        error = target - y

        # Delta rule weight update
        W = W + lr * np.outer(x, error)

        # Total squared error
        total_error += np.sum(error ** 2)

    error_list.append(total_error)


# --------------------------------------------------
# 8. Testing / Recognition Result
# --------------------------------------------------

print("Digit Recognition Results:\n")

for i in range(len(X)):
    y = np.dot(X[i], W)

    predicted_digit = np.argmax(y) + 1

    print("Actual Digit     :", i + 1)
    print("Network Output   :", np.round(y, 3))
    print("Recognized Digit :", predicted_digit)
    print("-----------------------------------")


# --------------------------------------------------
# 9. Convergence Curve
# --------------------------------------------------

plt.figure(figsize=(7, 4))
plt.plot(range(1, epochs + 1), error_list, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Total Squared Error")
plt.title("Convergence Curve for Digit Recognition")
plt.grid(True)
plt.show()