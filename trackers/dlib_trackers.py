from trackers.tracker_base import TrackerBase
import dlib
import cv2


class DlibCorrelationTrackers(TrackerBase):
    def __init__(self):
        self.trackers = []

    def create_trackers(self, boxes, frame):
        # Throw the last trackers away, its a very dummy implementation
        # but we can work on it later and probably use the best of both worlds,
        self.trackers.clear()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for box in boxes:
            (left_x, top_y, right_x, bottom_y) = box.astype("int")
            # Construct a dlib rectangle object from the bounding
            # box coordinates and then start the dlib correlation
            # tracker
            rect = dlib.rectangle(left_x, top_y, right_x, bottom_y)
            # Create a correlation tracker and start tracking
            tracker = dlib.correlation_tracker()
            tracker.start_track(rgb, rect)
            self.trackers.append(tracker)

    def update_trackers(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = []
        for tracker in self.trackers:
            # Update the tracker and grab the updated position
            tracker.update(rgb)
            pos = tracker.get_position()
            # Unpack the position object
            left_x = int(pos.left())
            top_y = int(pos.top())
            right_x = int(pos.right())
            bottom_y = int(pos.bottom())
            # Add the bounding box coordinates to the rectangles list
            boxes.append((left_x, top_y, right_x, bottom_y))

        return boxes
