import cv2
from ultralytics import YOLO

class YOLODetector:
    def __init__(self):
        # Load YOLO model
        self.model = YOLO("yolo_models/yolov8n.pt")
        self.model.to("cuda")
        self.detection=None

    def inference(self, frame):
        """Capture a frame, run detection, and return results and annotated frame"""
        results = self.model(frame)
        self.detection = results[0]
        self.annotated_frame = results[0].plot()

    def get_bounding_boxes(self):
        """Return list of bounding boxes [(x1, y1, x2, y2), ...] from a detection result"""
        if self.detection is None:
            return []
        boxes = self.detection.boxes.xyxy.cpu().numpy()  # Convert to numpy array
        return boxes.tolist()

    def get_annotated_frame(self):
        return self.annotated_frame
    


def crop_to_bounding_box(image, box):
    """Return the cropped image inside the bounding box (x1, y1, x2, y2)"""
    x1, y1, x2, y2 = map(int, box)
    cropped = image[y1:y2, x1:x2]
    return cropped
