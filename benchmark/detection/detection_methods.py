import cv2


class DetectorHolder:
    def __init__(self, detector_id, protoxt=None, model=None):
        # Initialize the various holders for all detection methods
        # which will be instantiated later when necessary
        self.__face_cascade = None
        # Save the ID of the desired detector
        self.detector_id = detector_id
        # Load the desired detector through its id
        self.__load_desired_detector()

    def detect(self, frame):
        if self.detector_id == 1:
            return self.__haar_cascade(frame)

    def __load_desired_detector(self):
        if self.__detector_id == 1:
            self.__face_cascade = cv2.CascadeClassifier()
            path_to_cascade = 'haarcascades/haarcascade_frontalface_default.xml'
            if not self.__face_cascade.load(cv2.samples.findFile(path_to_cascade)):
                print('--(!)Error loading face cascade')
                exit(0)
            pass

    def __haar_cascade(self, frame):
        bounding_boxes = []
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)
        # Detect faces
        faces = self.__face_cascade.detectMultiScale(frame_gray)
        # TODO find a way to return detection score
        for (x, y, w, h) in faces:
            bounding_boxes.append((int(x), int(y), int(w), int(h)))
        return bounding_boxes
