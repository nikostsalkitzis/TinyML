import serial

#See what is received by the raspberry py pico

# Open serial connection (check the correct port, e.g., /dev/ttyACM0 or COMx)
ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)  # Change to /dev/ttyUSB0 if needed

print("Waiting for data from Pico...")

while True:
    if ser.in_waiting > 0:  # Check if data is available
        data = ser.readline().decode().strip()  # Read a line and decode it
        print(f"Received from Pico: {data}")

