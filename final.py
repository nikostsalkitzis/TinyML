import serial
import numpy as np
import tensorflow.lite as tflite
import time  

# Open serial connection to Raspberry Pi Pico
ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)  # Adjust port if needed

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

    # Compare predictions and decide LED state
    if predicted_class_original == predicted_class_quantized:
        ser.write(b'1')  # Send "1" to turn on LED
        led_state = "ON"
    else:
        ser.write(b'0')  # Send "0" to turn off LED
        led_state = "OFF"

    # Print for debugging
    print(f"Sent prediction {i+1}: Original={predicted_class_original}, Quantized={predicted_class_quantized}, LED={led_state}")

    # Wait before sending the next prediction
    time.sleep(1)

ser.close()

