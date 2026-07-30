import cv2
import numpy as np
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")  # auto-downloads on first run
        self.classes = self.model.names  # dict: {0: 'person', 1: 'bicycle', ...}

    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]

        boxes, confidences, class_ids = [], [], []

        for box in results.boxes:
            confidence = float(box.conf[0])
            if confidence < 0.5:
                continue

            class_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Convert to (x, y, w, h) for DeepSort
            x, y, w, h = x1, y1, x2 - x1, y2 - y1

            # Clamp to frame
            height, width = frame.shape[:2]
            x = max(0, x)
            y = max(0, y)
            w = min(w, width - x)
            h = min(h, height - y)

            boxes.append([x, y, w, h])
            confidences.append(confidence)
            class_ids.append(class_id)

        return boxes, confidences, class_ids, self.classes