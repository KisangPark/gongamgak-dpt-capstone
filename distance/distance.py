import Jetson.GPIO as GPIO
import time

TRIG = 29   # BOARD 핀 번호
ECHO = 31   # BOARD 핀 번호

GPIO.setmode(GPIO.BOARD)
GPIO.setup(TRIG, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(ECHO, GPIO.IN)

SPEED_OF_SOUND = 34300  # cm/s

def measure_distance():
    # 1) TRIG 핀 LOW 안정화
    GPIO.output(TRIG, GPIO.LOW)
    time.sleep(0.002)   # 2ms

    # 2) TRIG 핀 HIGH (10 µs 펄스)
    GPIO.output(TRIG, GPIO.HIGH)
    time.sleep(0.00001)   # 10 µs
    GPIO.output(TRIG, GPIO.LOW)

    # 3) ECHO가 HIGH로 바뀔 때까지 대기
    timeout = time.time() + 0.05  # 50ms 타임아웃
    while GPIO.input(ECHO) == 0:
        if time.time() > timeout:
            return None
    pulse_start = time.time()

    # 4) ECHO가 LOW로 떨어질 때까지 대기
    timeout = time.time() + 0.05
    while GPIO.input(ECHO) == 1:
        if time.time() > timeout:
            return None
    pulse_end = time.time()

    # 5) ECHO HIGH 시간 계산
    pulse_duration = pulse_end - pulse_start   # 초 단위
    distance = (pulse_duration * SPEED_OF_SOUND) / 2
    return distance, pulse_duration

try:
    while True:
        result = measure_distance()
        if result is None:
            print("No echo (timeout)")
        else:
            dist_cm, pulse_s = result
            print(f"ECHO HIGH = {pulse_s*1e6:.0f} µs, Distance = {dist_cm:.2f} cm")
        time.sleep(1)

except KeyboardInterrupt:
    print("Measurement stopped by User")
finally:
    GPIO.cleanup()
