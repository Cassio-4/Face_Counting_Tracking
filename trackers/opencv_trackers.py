from trackers.tracker_base import TrackerBase
import cv2
import numpy as np


# https://www.learnopencv.com/multitracker-multiple-object-tracking-using-opencv-c-python/
class OpenCVMultitrackerBase(TrackerBase):
    def __init__(self, tracker_name="MEDIANFLOW"):
        # Start a holder for the trackers
        self.__multi_tracker = None

    def _get_tracker(self):
        return None

    def create_trackers(self, boxes, frame):
        self.__multi_tracker = cv2.MultiTracker_create()
        for box in boxes:
            self.__multi_tracker.add(self._get_tracker(), frame, tuple(box))

    def update_trackers(self, frame):
        success, boxes = self.__multi_tracker.update(frame)
        if type(boxes) == tuple:
            return boxes
        else:
            return boxes.astype(np.uint16)


class OpenCVMultitrackerBoosting(OpenCVMultitrackerBase):
    def _get_tracker(self):
        return cv2.TrackerBoosting_create()


class OpenCVMultitrackerMil(OpenCVMultitrackerBase):
    def _get_tracker(self):
        return cv2.TrackerMIL_create()


class OpenCVMultitrackerKcf(OpenCVMultitrackerBase):
    def _get_tracker(self):
        return cv2.TrackerKCF_create()


class OpenCVMultitrackerTld(OpenCVMultitrackerBase):
    def _get_tracker(self):
        return cv2.TrackerTLD_create()


class OpenCVMultitrackerGoturn(OpenCVMultitrackerBase):
    def _get_tracker(self):
        return cv2.TrackerGOTURN_create()


class OpenCVMultitrackerMosse(OpenCVMultitrackerBase):
    def _get_tracker(self):
        return cv2.TrackerMOSSE_create()


class OpenCVMultitrackerCsrt(OpenCVMultitrackerBase):
    def _get_tracker(self):
        return cv2.TrackerCSRT_create()


class OpenCVMultitrackerMedianFlow(OpenCVMultitrackerBase):
    def _get_tracker(self):
        return cv2.TrackerMedianFlow_create()
