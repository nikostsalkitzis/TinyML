import machine
import sys

# Define onboard LED pin (GPIO 25 for built-in LED on Raspberry Pi Pico)
led = machine.Pin(25, machine.Pin.OUT)

while True:
    data = sys.stdin.read(1)  # Read one byte from serial (non-blocking)
    if data:
        if data == '1':
            led.value(1)  # Turn ON LED
        elif data == '0':
            led.value(0)  # Turn OFF LED
