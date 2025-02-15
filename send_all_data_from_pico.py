import serial

# Open serial connection (adjust port if needed)
ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)  # Change to /dev/ttyUSB0 or COMx if needed

print("Waiting for data from Pico...")

while True:
    if ser.in_waiting > 0:  # Check if data is available
        data = ser.readline().decode().strip()  # Read a line and decode it
        
        # Extract predictions from message format: "Original: X, Quantized: Y, YOLO: Z"
        if data:
            try:
                parts = data.replace("Original:", "").replace("Quantized:", "").replace("YOLO:", "").split(",")
                original_pred = parts[0].strip()
                quantized_pred = parts[1].strip()
                yolo_pred = parts[2].strip()

                print(f"🔵 Original Model Prediction: {original_pred}")
                print(f"🟢 Quantized Model Prediction: {quantized_pred}")
                print(f"🟠 YOLO Model Prediction: {yolo_pred}\n")

            except Exception as e:
                print(f"⚠️ Error parsing received data: {data} | {e}")
        else:
            print(f"Received from Pico: {data}")  # Print raw data if format is different
