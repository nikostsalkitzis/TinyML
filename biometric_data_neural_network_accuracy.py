import machine
import utime
import random
import math

# Set seed for reproducibility
random.seed(42)

# LED setup for Raspberry Pi Pico
led = machine.Pin(25, machine.Pin.OUT)

# Normalize input data function
def normalize_data(x, min_val, max_val):
    return [(val - min_val) / (max_val - min_val) for val in x]

# Generate synthetic patient data
def generate_patient_data(num_samples=1000):
    X = [[random.uniform(36, 40),   # Body temperature (°C)
          random.uniform(80, 180),  # Systolic Blood Pressure (mmHg)
          random.uniform(60, 120),  # Diastolic Blood Pressure (mmHg)
          random.uniform(60, 150)]  # Heart Rate (bpm)
         for _ in range(num_samples)]
    X = [normalize_data(x, 36, 180) for x in X]  # Normalize inputs
    y = [classify_health_status(x) for x in X]
    return X, y

# Classify health status based on synthetic thresholds
def classify_health_status(patient):
    temp, sys_bp, dia_bp, heart_rate = patient
    temp = temp * 144 + 36  # Revert normalization for classification
    sys_bp = sys_bp * 144 + 36
    dia_bp = dia_bp * 144 + 36
    heart_rate = heart_rate * 144 + 36
    
    if temp < 37.5 and 90 <= sys_bp <= 140 and 60 <= dia_bp <= 90 and 60 <= heart_rate <= 100:
        return 0  # Healthy
    elif 37.5 <= temp < 39 or sys_bp < 90 or sys_bp > 160 or dia_bp < 60 or dia_bp > 100 or heart_rate < 50 or heart_rate > 120:
        return 1  # At Risk
    else:
        return 2  # Critical

# Generate dataset
X_train, y_train = generate_patient_data(1000)
X_test, y_test = generate_patient_data(200)

# Neural Network Parameters
input_size = 4   # Temp, BP (Sys/Dia), Heart Rate
hidden_size = 16  # Increased hidden layer neurons
hidden_layer_2 = 16  # Added another hidden layer
output_size = 3  # Health Status (Healthy, At Risk, Critical)
learning_rate = 0.003  # Adjusted learning rate
epochs = 20  # Increased epochs for better training

# Initialize Weights and Biases
W1 = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(input_size)]
b1 = [random.uniform(-1, 1) for _ in range(hidden_size)]
W2 = [[random.uniform(-1, 1) for _ in range(hidden_layer_2)] for _ in range(hidden_size)]
b2 = [random.uniform(-1, 1) for _ in range(hidden_layer_2)]
W3 = [[random.uniform(-1, 1) for _ in range(output_size)] for _ in range(hidden_layer_2)]
b3 = [random.uniform(-1, 1) for _ in range(output_size)]

# Activation Functions
def relu(x):
    return max(0, x)

def softmax(X):
    max_x = max(X) if X else 0  # Prevent empty list issues
    exp_values = [math.exp(x - max_x) for x in X]
    sum_values = sum(exp_values) if sum(exp_values) != 0 else 1  # Prevent division by zero
    return [x / sum_values for x in exp_values]

# Forward Propagation
def forward(x):
    hidden1 = [relu(sum(x[i] * W1[i][j] for i in range(input_size)) + b1[j]) for j in range(hidden_size)]
    hidden2 = [relu(sum(hidden1[j] * W2[j][k] for j in range(hidden_size)) + b2[k]) for k in range(hidden_layer_2)]
    output = [sum(hidden2[k] * W3[k][m] for k in range(hidden_layer_2)) + b3[m] for m in range(output_size)]
    return softmax(output)

# Training Loop
for epoch in range(epochs):
    for i in range(len(X_train)):
        x = X_train[i]
        y = y_train[i]
        output = forward(x)
        target = [1 if j == y else 0 for j in range(output_size)]
        error = [(output[j] - target[j]) for j in range(output_size)]
        for j in range(hidden_size):
            for k in range(hidden_layer_2):
                if k < len(error):
                    W2[j][k] -= learning_rate * error[k]
            if j < len(error):
                b2[j] -= learning_rate * error[j]
        for k in range(hidden_layer_2):
            for m in range(output_size):
                if m < len(error):
                    W3[k][m] -= learning_rate * error[m]
            if k < len(error):
                b3[k] -= learning_rate * error[k]
    print(f"Epoch {epoch+1} completed")

# Test with a new patient sample
new_patient = normalize_data([random.uniform(36, 40), random.uniform(80, 180), random.uniform(60, 120), random.uniform(60, 150)], 36, 180)
real_status = classify_health_status(new_patient)
predicted_probs = forward(new_patient)
if predicted_probs:
    predicted_status = predicted_probs.index(max(predicted_probs))
else:
    predicted_status = 0  # Default to 'Healthy' in case of an error

status_labels = ["Healthy", "At Risk", "Critical"]
print(f"Real Health Status: {status_labels[real_status]}")
print(f"Predicted Health Status: {status_labels[predicted_status]}")

# Evaluate Accuracy
correct_predictions = 0
for i in range(len(X_test)):
    real_status = y_test[i]
    predicted_probs = forward(X_test[i])
    if predicted_probs:
        predicted_status = predicted_probs.index(max(predicted_probs))
    else:
        predicted_status = 0
    if predicted_status == real_status:
        correct_predictions += 1

accuracy = correct_predictions / len(X_test) * 100
print(f"Model Accuracy: {accuracy:.2f}%")


