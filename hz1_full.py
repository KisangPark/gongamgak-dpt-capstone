# full pipeline code with 1 Hz real-time 

from haptic_package.haptic_device import HAPTIC_DEVICE
from dpt_package.asdf import asdf


BAUDRATE = 15200
PORT_NAME = "/dev/ttyUSB0"

device = HAPTIC_DEVICE(port_name=PORT_NAME, baudrate=BAUDRATE)


data_to_write = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
device.write_data(data_to_write)

