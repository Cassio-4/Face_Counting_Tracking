from abc import ABC, abstractmethod


class TrackerBase(ABC):
    @abstractmethod
    def create_trackers(self, boxes, frame):
        pass

    @abstractmethod
    def update_trackers(self, frame):
        pass
