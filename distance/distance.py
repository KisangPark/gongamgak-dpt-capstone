import Jetson.GPIO as GPIO
import time
from statistics import median

# BOARD 번호(물리 핀 번호)로 사용
TRIG = 16   # BOARD #16 (출력)
ECHO = 18   # BOARD #18 (입력, 5V->3.3V 변환 후 연결)

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(TRIG, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(ECHO, GPIO.IN)

SPEED_OF_SOUND = 343.0  # m/s (20°C 기준)
MAX_RANGE_M = 4.0       # HC-SR04 권장 범위(대략)
# 최대 왕복 시간(여유 포함) 계산
MAX_PULSE_SEC = (2 * MAX_RANGE_M) / SPEED_OF_SOUND  # 왕복 시간
TIMEOUT_SEC = MAX_PULSE_SEC * 1.5  # 여유

def pulse_in(pin, level, timeout_sec):
    """Arduino pulseIn 유사: 원하는 level의 펄스 폭(s)을 측정, timeout 시 None"""
    start_time = time.monotonic()
    end_time = start_time

    # 1) 원하는 level이 시작될 때까지 대기
    while GPIO.input(pin) != level:
        if (time.monotonic() - start_time) > timeout_sec:
            return None
    t_rise = time.monotonic_ns()

    # 2) level이 끝날 때까지 대기
    while GPIO.input(pin) == level:
        if (time.monotonic() - start_time) > timeout_sec:
            return None
    t_fall = time.monotonic_ns()

    return (t_fall - t_rise) / 1e9  # seconds

def single_measure():
    # 트리거 펄스: 10µs HIGH
    GPIO.output(TRIG, GPIO.LOW)
    time.sleep(0.000002)  # 2µs 안정화
    GPIO.output(TRIG, GPIO.HIGH)
    time.sleep(0.000010)  # 10µs
    GPIO.output(TRIG, GPIO.LOW)

    # ECHO High 펄스 폭 측정
    high_time = pulse_in(ECHO, GPIO.HIGH, TIMEOUT_SEC)
    if high_time is None:
        return None  # 타임아웃

    # 거리 = (시간 * 음속) / 2
    dist_m = (high_time * SPEED_OF_SOUND) / 2.0
    return dist_m

def get_distance(num_samples=5):
    # 첫 값은 워밍업으로 버림
    _ = single_measure()
    samples = []
    for _ in range(num_samples):
        d = single_measure()
        if d is not None:
            samples.append(d)
        time.sleep(0.05)  # 50ms 간격

    if not samples:
        return None
    # 메디안이 튼튼
    return median(samples)

try:
    while True:
        d_m = get_distance(num_samples=5)
        if d_m is None:
            print("No echo (timeout). Check wiring/target range.")
        else:
            print(f"Distance: {d_m*100:.1f} cm")
        time.sleep(0.3)

except KeyboardInterrupt:
    pass
finally:
    GPIO.cleanup()
