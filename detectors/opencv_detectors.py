import cv2
import numpy as np


class OpenCVHaarFaceDetector:
    def __init__(self, model_path, scaleFactor=1.3, minNeighbors=5):
        self.face_cascade = cv2.CascadeClassifier(model_path)
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors

    def detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray, self.scaleFactor,
                                                   self.minNeighbors)
        faces = [(x, y, x+w, y+h) for x, y, w, h in faces]
        return np.asarray(faces, dtype=np.uint16)
