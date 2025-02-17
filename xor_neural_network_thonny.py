import random
import math
import machine

# Initialize the onboard LED
led = machine.Pin(25, machine.Pin.OUT)

# Sigmoid Activation Function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Derivative of Sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Neural Network Class
class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Random initialization of weights and biases
        self.weights_input_hidden = [[random.uniform(-1, 1) for _ in range(self.hidden_size)] for _ in range(self.input_size)]
        self.weights_hidden_output = [random.uniform(-1, 1) for _ in range(self.hidden_size)]
        
        self.bias_hidden = [random.uniform(-1, 1) for _ in range(self.hidden_size)]
        self.bias_output = random.uniform(-1, 1)

    def forward(self, inputs):
        # Feedforward pass
        self.hidden_layer_input = [sum(inputs[i] * self.weights_input_hidden[i][j] for i in range(self.input_size)) + self.bias_hidden[j] for j in range(self.hidden_size)]
        self.hidden_layer_output = [sigmoid(i) for i in self.hidden_layer_input]
        
        self.output_layer_input = sum(self.hidden_layer_output[i] * self.weights_hidden_output[i] for i in range(self.hidden_size)) + self.bias_output
        self.output_layer_output = sigmoid(self.output_layer_input)
        
        return self.output_layer_output

    def train(self, inputs, targets, epochs=10000, learning_rate=0.1):
        for epoch in range(epochs):
            for i in range(len(inputs)):
                input_data = inputs[i]
                target = targets[i]
                predicted_output = self.forward(input_data)
                
                # Backpropagation
                output_error = target - predicted_output
                output_delta = output_error * sigmoid_derivative(predicted_output)
                
                hidden_errors = [output_delta * self.weights_hidden_output[j] for j in range(self.hidden_size)]
                hidden_deltas = [hidden_errors[j] * sigmoid_derivative(self.hidden_layer_output[j]) for j in range(self.hidden_size)]
                
                # Update weights and biases
                for j in range(self.hidden_size):
                    self.weights_hidden_output[j] += learning_rate * output_delta * self.hidden_layer_output[j]
                    self.bias_output += learning_rate * output_delta
                
                for j in range(self.hidden_size):
                    for k in range(self.input_size):
                        self.weights_input_hidden[k][j] += learning_rate * hidden_deltas[j] * input_data[k]
                    self.bias_hidden[j] += learning_rate * hidden_deltas[j]
                
            if epoch % 1000 == 0:
                total_error = sum((targets[i] - self.forward(inputs[i]))**2 for i in range(len(inputs))) / len(inputs)
                print(f"Epoch {epoch}/{epochs}, Error: {total_error}")

# Train the neural network
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
targets = [0, 1, 1, 0]

nn = SimpleNN(input_size=2, hidden_size=2, output_size=1)
nn.train(inputs, targets)

# Take user input and control LED
while True:
    try:
        x1 = int(input("Enter first input (0 or 1): "))
        x2 = int(input("Enter second input (0 or 1): "))
        if x1 not in [0, 1] or x2 not in [0, 1]:
            print("Invalid input! Enter only 0 or 1.")
            continue
        
        output = nn.forward([x1, x2])
        print(f"Predicted Output: {output:.4f}")
        
        if output > 0.5:
            led.value(1)  # Turn on LED
            print("LED ON")
        else:
            led.value(0)  # Turn off LED
            print("LED OFF")
    except ValueError:
        print("Invalid input! Enter numerical values 0 or 1.")

