import Jetson.GPIO as GPIO
import time

board_pin = 32 # for mode GPIO.BOARD

GPIO.setmode(GPIO.BOARD)
GPIO.setup(board_pin, GPIO.OUT)

# Create PWM at 200 Hz
pwm = GPIO.PWM(board_pin, 200)
pwm.start(0)  # 0% duty cycle (off)

try:
    while True:
        print("Value up")
        for duty in range(0, 101, 5):   # ramp up
            pwm.ChangeDutyCycle(duty)
            time.sleep(0.05)
        print("Value down")
        for duty in range(100, -1, -5): # ramp down
            pwm.ChangeDutyCycle(duty)
            time.sleep(0.05)

except KeyboardInterrupt:
    pass

finally:
    pwm.stop()
    GPIO.cleanup()
