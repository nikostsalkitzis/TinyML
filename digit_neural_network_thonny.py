import machine
import utime
import random
import math

# Random seed for reproducibility
random.seed(42)

# LED setup
led = machine.Pin(25, machine.Pin.OUT)

# Generate a larger dataset (simulate 8x8 digit images)
def generate_data(num_samples=100000):
    X = [[random.uniform(0, 1) for _ in range(64)] for _ in range(num_samples)]  # 8x8 flattened images
    y = [random.randint(0, 9) for _ in range(num_samples)]  # Labels (0-9)
    return X, y

X_train, y_train = generate_data(100)  # Increase the dataset size

# Neural Network Parameters
input_size = 64  # 8x8 image
hidden_size = 16  # Hidden neurons
output_size = 10  # Digits 0-9
learning_rate = 0.01
epochs = 10  # Number of training iterations

# Initialize Weights and Biases
W1 = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(input_size)]
b1 = [random.uniform(-1, 1) for _ in range(hidden_size)]
W2 = [[random.uniform(-1, 1) for _ in range(output_size)] for _ in range(hidden_size)]
b2 = [random.uniform(-1, 1) for _ in range(output_size)]

# Activation Functions
def relu(x):
    return max(0, x)

def softmax(X):
    exp_values = [math.exp(x) for x in X]
    sum_values = sum(exp_values)
    return [x / sum_values for x in exp_values]

# Forward Propagation
def forward(x):
    # Input to Hidden Layer
    hidden = [relu(sum(x[i] * W1[i][j] for i in range(input_size)) + b1[j]) for j in range(hidden_size)]
    # Hidden to Output Layer
    output = [sum(hidden[j] * W2[j][k] for j in range(hidden_size)) + b2[k] for k in range(output_size)]
    return softmax(output)

# Training (Simple SGD)
for epoch in range(epochs):
    for i in range(len(X_train)):
        x = X_train[i]
        y = y_train[i]

        # Forward pass
        output = forward(x)

        # Compute error (simplified)
        target = [1 if j == y else 0 for j in range(output_size)]
        error = [(output[j] - target[j]) for j in range(output_size)]

        # Backpropagation (Manual Gradient Descent Update)
        for j in range(hidden_size):
            for k in range(output_size):
                W2[j][k] -= learning_rate * error[k] * W2[j][k]  # Update hidden-output weights
            b2[k] -= learning_rate * error[k]  # Update biases

        for i in range(input_size):
            for j in range(hidden_size):
                W1[i][j] -= learning_rate * sum(error) * W1[i][j]  # Update input-hidden weights
            b1[j] -= learning_rate * sum(error)  # Update biases
        
    print(f"Epoch {epoch} completed")

# Test with a new example
example_digit = [random.uniform(0, 1) for _ in range(64)]  # Random digit input
predicted_probs = forward(example_digit)
predicted_digit = predicted_probs.index(max(predicted_probs))
print("Predicted Digit:", predicted_digit)

# Blink LED as many times as the predicted digit
for _ in range(predicted_digit):
    led.on()
    utime.sleep(0.5)
    led.off()
    utime.sleep(0.5)

