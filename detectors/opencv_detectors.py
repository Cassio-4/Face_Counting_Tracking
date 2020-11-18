import cv2


class OpenCVHaarFaceDetector:
    def __init__(self, scaleFactor=1.3, minNeighbors=5):
        model_path = "models/haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(model_path)
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors

    def detect_face(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray, self.scaleFactor,
                                                   self.minNeighbors)
        faces = [[x, y, w, h, 1] for x, y, w, h in faces]
        return faces
