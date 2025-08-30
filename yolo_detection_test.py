from ultralytics import YOLO
import cv2

# Load YOLOv8 nano model (pretrained on COCO dataset)
# You can also use "yolov8s.pt" or "yolov8n.pt"
model = YOLO("yolov8n.pt")  
model.to("cuda")

# Open camera (0 = default webcam, replace with your CSI camera if needed)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame)

    # Draw bounding boxes
    annotated_frame = results[0].plot()

    # Show result
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



