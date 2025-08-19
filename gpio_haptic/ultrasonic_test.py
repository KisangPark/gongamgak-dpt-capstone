# ultrasonic sensor on jetson pin
import Jetson.GPIO as GPIO
import time

# Pin definitions (BOARD numbering = physical pins on 40-pin header)
TRIG = 16   # Physical pin 16
ECHO = 18   # Physical pin 18

GPIO.setmode(GPIO.BOARD)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

def get_distance():
    # Ensure trigger is LOW
    GPIO.output(TRIG, False)
    time.sleep(0.05)

    # Send 10µs pulse to TRIG
    GPIO.output(TRIG, True)
    time.sleep(0.00001)   # 10 µs
    GPIO.output(TRIG, False)

    # Wait for ECHO HIGH
    while GPIO.input(ECHO) == 0:
        pulse_start = time.time()

    # Wait for ECHO LOW
    while GPIO.input(ECHO) == 1:
        pulse_end = time.time()

    # Pulse duration
    pulse_duration = pulse_end - pulse_start

    # Speed of sound = 34300 cm/s → divide by 2 (go + return)
    distance = (pulse_duration * 34300) / 2
    return distance

try:
    while True:
        dist = get_distance()
        print(f"Distance: {dist:.2f} cm")
        time.sleep(0.5)

except KeyboardInterrupt:
    print("Measurement stopped by User")
    GPIO.cleanup()
