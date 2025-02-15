import serial
import numpy as np
import tensorflow.lite as tflite
import time
from ultralytics import YOLO



# Open serial connection to Raspberry Pi Pico
ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)  # Adjust port if needed

# Load YOLO model
yolo_model = YOLO("best.pt")
input_size = (32, 32) 


# Load both models (original and quantized)
interpreter_original = tflite.Interpreter(model_path="model.tflite")
interpreter_quantized = tflite.Interpreter(model_path="model_quantized.tflite")

# Allocate tensors
interpreter_original.allocate_tensors()
interpreter_quantized.allocate_tensors()

# Get input and output details
input_details_original = interpreter_original.get_input_details()
output_details_original = interpreter_original.get_output_details()

input_details_quantized = interpreter_quantized.get_input_details()
output_details_quantized = interpreter_quantized.get_output_details()

# Extract scale and zero point for quantized model
input_scale, input_zero_point = input_details_quantized[0]['quantization']

# Loop to process 100 images
for i in range(100):
    # Generate a new random input (simulated CIFAR-10 image)
    input_data_float32 = np.random.rand(1, 32, 32, 3).astype(np.float32)

    # Run inference on the original model (keeps float32 input)
    interpreter_original.set_tensor(input_details_original[0]['index'], input_data_float32)
    interpreter_original.invoke()
    output_original = interpreter_original.get_tensor(output_details_original[0]['index'])
    predicted_class_original = np.argmax(output_original)

    # Convert input to UINT8 for quantized model
    input_data_uint8 = np.uint8(input_data_float32 / input_scale + input_zero_point)

    # Run inference on the quantized model
    interpreter_quantized.set_tensor(input_details_quantized[0]['index'], input_data_uint8)
    interpreter_quantized.invoke()
    output_quantized = interpreter_quantized.get_tensor(output_details_quantized[0]['index'])
    predicted_class_quantized = np.argmax(output_quantized)

    # Run YOLO inference
    
    # Convert input image for YOLO (scale & type conversion)
    input_image = (input_data_float32 * 255).astype(np.uint8)[0]  # Scale and convert to uint8 # Convert float32 to uint8
    yolo_results = yolo_model.predict(input_image, save=False)

    # Extract YOLO top1 class index
    yolo_top1_class = None
    for result in yolo_results:
        if result.probs is not None:  # Check if probabilities exist
            yolo_top1_class = result.probs.top1  # Get the top1 class index

    # If no objects were detected, assign a default value (e.g., -1)
    yolo_top1_class = yolo_top1_class if yolo_top1_class is not None else -1

    # Format the message for serial transmission
    message = f"Original: {predicted_class_original}, Quantized: {predicted_class_quantized}, YOLO: {yolo_top1_class}\n"

    #Test locally
   #print(message)
    
    # Send predictions to Raspberry Pi Pico
    ser.write(message.encode())

    print(f"Sent prediction {i+1}: {message.strip()}")

    # Wait for 5 seconds before sending the next prediction
    time.sleep(5)

# Close the serial connection
ser.close()
