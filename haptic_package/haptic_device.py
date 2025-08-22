import serial
import time
import numpy as np

class HAPTIC_DEVICE():
    def __init__(self, port_name, baudrate):
        # asdf

        # establish serial connection
        self.ser = serial.Serial(port_name, baudrate, timeout=1)
        time.sleep(2)

    def write_data(self, string_data=np.zeros(6)):
        # write bytes to serial
        # Convert to string (space-separated)
        arr_str = ' '.join(map(str, string_data))
        print("String representation:", arr_str)

        byte = arr_str.encode('utf-8')
        self.ser.write(byte)

    def end_connection(self):
        self.ser.close()