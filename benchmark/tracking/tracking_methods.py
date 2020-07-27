import cv2

trackers_available = ((1, "Boosting"), (2, "MIL"), (3, "KCF"), (4, "TLD"),
                      (5, "MedianFlow"),
                      (6, "GOTURN (requires caffe model)"),
                      (7, "MOSSE"), (8, "CSRT"))


class TrackingHolder:

    def __init__(self, tracker_id):
        # Save the tracker id
        self.__tracker_id = tracker_id
        # Start a holder for the trackers
        self.__tracker = None
        self.__multi_tracker = None
        # Instantiate desired tracker

    def ___create_tracker(self):
        # Boosting
        if self.__tracker_id == 1:
            return cv2.TrackerBoosting_create()
        # MIL
        elif self.__tracker_id == 2:
            return cv2.TrackerMIL_create()
        # KCF
        elif self.__tracker_id == 3:
            return cv2.TrackerKCF_create()
        # TLD
        elif self.__tracker_id == 4:
            return cv2.TrackerTLD_create()
        # MedianFlow
        elif self.__tracker_id == 5:
            return cv2.TrackerMedianFlow_create()
        # GOTURN
        elif self.__tracker_id == 6:
            return cv2.TrackerGOTURN_create()
        # MOSSE
        elif self.__tracker_id == 7:
            return cv2.TrackerMOSSE_create()
        # CSRT
        elif self.__tracker_id == 8:
            return cv2.TrackerCSRT_create()
        else:
            print("ERROR: unavailable tracker id")
            print("(!) available trackers are:")
            for t in trackers_available:
                print(t)

    def track(self, frame, new_bboxes):
        if self.__tracker_id <= 8:
            for new_bbox in new_bboxes:
                self.__multi_tracker.add(self.___create_tracker(), frame,
                                         new_bbox)
            success, boxes = self.__multi_tracker.update(frame)
            return boxes
