# full pipeline code with 1 Hz real-time 

from haptic_package.haptic_device import HAPTIC_DEVICE
from model_package.dpt_models import DPT
from model_package.detection_models import *
import cv2

# BAUDRATE = 15200
# PORT_NAME = "/dev/ttyUSB0"
# haptic_device = HAPTIC_DEVICE(port_name=PORT_NAME, baudrate=BAUDRATE)


dpt_model = DPT()


detection_model = YOLODetector()
detection_model.detect()
bbox_list = detection_model.get_bounding_boxes()
frame = detection_model.get_annotated_frame()


cropped_image = crop_to_bounding_box(frame, bbox_list[0])
dpt_model.receive_image(frame)

visualized_depth = dpt_model.get_depth()
cv2.imshow('example', visualized_depth)
cv2.waitKey(0)
cv2.destroyAllWindows()
# print(visualized_depth.shape) # (256,256)


# print("bbox list:", bbox_list)
# cv2.imshow('example', frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
