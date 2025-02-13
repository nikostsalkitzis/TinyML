import serial
import numpy as np
import tensorflow.lite as tflite
import time  # Import time module for delay

#Code for sending data to the pico





# Open serial connection to Raspberry Pi Pico
ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)  # Adjust port if needed

# Load TFLite model
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Loop to process 100 images
for i in range(100):
    # Generate a new random input (simulated CIFAR-10 image)
    input_data = np.random.rand(1, 32, 32, 3).astype(np.float32)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get the prediction
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data)  # Get the class index

    # Send the prediction to Raspberry Pi Pico
    ser.write(f"{predicted_class}\n".encode())

    print(f"Sent prediction {i+1}: {predicted_class}")

    # Wait for 5 seconds before sending the next prediction
    time.sleep(5)

ser.close()

