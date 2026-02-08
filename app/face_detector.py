import cv2
import numpy as np
from utils.paths import PROTOTXT, MODEL

class FaceDetector:
    def __init__(self):
        self.net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)

    def detect(self, frame, conf_threshold=0.5):
        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0),
        )

        self.net.setInput(blob)
        detections = self.net.forward()

        boxes = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                boxes.append(box.astype(int))

        return boxes
