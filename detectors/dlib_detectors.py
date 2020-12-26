import dlib
import cv2
import numpy as np


class DlibHogDetector:
    def __init__(self):
        self.__detector = dlib.get_frontal_face_detector()

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = self.__detector(rgb, 1)
        boxes = [(det.left(), det.top(), det.right(), det.bottom()) for det in detections]
        return np.asarray(boxes, dtype=np.uint16)
