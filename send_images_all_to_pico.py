import serial
import numpy as np
import tensorflow.lite as tflite
import time  
from ultralytics import YOLO
import cv2
import os

# Open serial connection to Raspberry Pi Pico
ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)  # Adjust port if needed

# Load TensorFlow Lite models
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

# Load YOLO model
yolo_model = YOLO("best.pt")

# Define the test folder path
test_folder = "test1/"

# Get a list of all image files in the test folder (limit to 100 images)
image_files = [f for f in os.listdir(test_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:50]

if not image_files:
    print("No images found in test folder.")
    exit()

print(f"Found {len(image_files)} images in {test_folder} (Processing up to 100)")

# Loop through the first 100 images in the test folder
for image_name in image_files:
    image_path = os.path.join(test_folder, image_name)
    
    print(f"🔄 Processing {image_name}...")
    
    # Load and preprocess the image
    input_image = cv2.imread(image_path)

    if input_image is None:
        print(f"Error: Could not load {image_name}. Skipping.")
        continue

    # Resize the image to match TensorFlow Lite expected input size (assuming 32x32)
    input_image_resized = cv2.resize(input_image, (32, 32))  # Resize for TFLite
    input_image_float32 = input_image_resized.astype(np.float32) / 255.0  # Normalize to [0,1]
    
    # Add batch dimension for TensorFlow Lite (convert to (1, 32, 32, 3))
    input_data_float32 = np.expand_dims(input_image_float32, axis=0)

    # Run inference on the original TFLite model
    interpreter_original.set_tensor(input_details_original[0]['index'], input_data_float32)
    interpreter_original.invoke()
    output_original = interpreter_original.get_tensor(output_details_original[0]['index'])
    predicted_class_original = np.argmax(output_original)

    # Convert input for quantized model
    input_data_uint8 = np.uint8(input_data_float32 / input_scale + input_zero_point)

    # Run inference on the quantized TFLite model
    interpreter_quantized.set_tensor(input_details_quantized[0]['index'], input_data_uint8)
    interpreter_quantized.invoke()
    output_quantized = interpreter_quantized.get_tensor(output_details_quantized[0]['index'])
    predicted_class_quantized = np.argmax(output_quantized)

    # Prepare image for YOLO (resize to 640x640 if needed)
    yolo_input_image = cv2.resize(input_image, (640, 640))
    yolo_input_image = cv2.cvtColor(yolo_input_image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Run YOLO inference
    yolo_results = yolo_model.predict(yolo_input_image, save=False)

    # Extract YOLO top1 class index
    yolo_top1_class = None
    for result in yolo_results:
        if result.probs is not None:  # Check if probabilities exist
            yolo_top1_class = result.probs.top1  # Get the top1 class index

    # If no objects were detected, assign a default value (e.g., -1)
    yolo_top1_class = yolo_top1_class if yolo_top1_class is not None else -1

    # Format and send the results
    message = f"Original: {predicted_class_original}, Quantized: {predicted_class_quantized}, YOLO: {yolo_top1_class}\n"
    
    # Send predictions to Raspberry Pi Pico
    ser.write(message.encode())

    print(f"Sent predictions for {image_name}: {message.strip()}")

    # Wait for 2 seconds before processing the next image
    time.sleep(5)

# Close the serial connection
ser.close()
print("Finished processing up to 100 images.")
