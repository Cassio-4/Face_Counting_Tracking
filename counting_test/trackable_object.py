class TrackableObject:
    def __init__(self, object_id, centroid):
        # Store the object ID, and centroid
        self.objectID = object_id
        self.centroid = centroid
        # Flag used to indicate if the object has already been counted
        # or not
        self.counted = False
