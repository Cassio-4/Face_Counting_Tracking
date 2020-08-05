import cv2


#https://www.learnopencv.com/multitracker-multiple-object-tracking-using-opencv-c-python/
class OpenCVMultitracker:
    def __init__(self, tracker_name="MEDIANFLOW"):
        # Save the tracker name
        self.tracker_name = tracker_name
        # Start a holder for the trackers
        self.__multi_tracker = cv2.MultiTracker_create()

    def create_tracker(self):
        if self.tracker_name == "BOOSTING":
            tracker = cv2.TrackerBoosting_create()
        elif self.tracker_name == "MIL":
            tracker = cv2.TrackerMIL_create()
        elif self.tracker_name == "KCF":
            tracker = cv2.TrackerKCF_create()
        elif self.tracker_name == "TLD":
            tracker = cv2.TrackerTLD_create()
        elif self.tracker_name == "GOTURN":
            tracker = cv2.TrackerGOTURN_create()
        elif self.tracker_name == "MOSSE":
            tracker = cv2.TrackerMOSSE_create()
        elif self.tracker_name == "CSRT":
            tracker = cv2.TrackerCSRT_create()
        else:
            tracker = cv2.TrackerMedianFlow_create()
        return tracker

    def track(self, frame, new_bboxes):
        for new_bbox in new_bboxes:
            self.__multi_tracker.add(self.create_tracker(), frame, new_bbox)
        success, boxes = self.__multi_tracker.update(frame)
        return boxes
