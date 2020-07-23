from counting_test.trackable_object import TrackableObject
from counting_test.centroidtracker import CentroidTracker
from imutils.video import FPS
import numpy as np
import argparse
import dlib
import cv2
"""
This code is a prototype to run a people counter through a video file
or images. This code is heavily based on the code provided by Adrian 
at: https://www.pyimagesearch.com/2018/08/13/opencv-people-counter/
It uses a mobile detector network and counts people's positions based
on a grid that is superimposed on the video.
"""
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
                help="Path to Caffe 'deploy prototxt file")
ap.add_argument("-m", "--model", required=True,
                help="Path to Caffe pre-trained model")
ap.add_argument("-i", "--input", type=str, required=True,
                help="Path to chokepoint dataset's video folder")
ap.add_argument("-o", "--output", type=str,
                help="Path to optional input video file")
ap.add_argument("-c", "--confidence", type=float, default=0.7,
                help="Minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=30,
                help="# of skipped frames between detections")
ap.add_argument("-gh", "--gridHeight", type=int, default=8,
                help="Grid height")
ap.add_argument("-gw", "--gridWidth", type=int, default=8,
                help="Grid width")
args = vars(ap.parse_args())

# Initialize the list of class labels MobileNet SSD was trained to detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle",
           "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
           "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
           "tvmonitor"]

# Load the serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# Opening txt file with frame names
txt_file = open(args["input"]+"all_file.txt", "r")
# Getting its contents as a list of frame names
frame_names = txt_file.readlines()
# Grabbing the first frame of the sequence to get it's dimensions
frame = cv2.imread(args["input"] + frame_names[0].strip("\n"))

# Initializing video dimensions
H, W = frame.shape[:2]
print('Frame dimensions: H={}, W={}'.format(H, W))
# If we're to write a video on disk, instantiate the writer
writer = None
if args["output"] is not None:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(args["output"], fourcc, 30, (W, H), True)

# Instantiate centroid tracker which contains the Grid
ct = CentroidTracker(maxDisappeared=40, maxDistance=50, video_dim=(H, W),
                     height=args["gridHeight"], width=args["gridWidth"])

# Initialize a list to store each dlib's correlation tracker
trackers = []
# Initialize a dictionary to map each unique object ID to a
# TrackableObject
trackableObjects = {}

# Initialize the total number of frames processed thus far
totalFrames = 0
# Start the frames per second throughput estimator
fps = FPS().start()

# Loop over frames from the chokepoint dataset
for line in frame_names:
    # Read the frame using its name from file
    frame = cv2.imread(args["input"]+line.strip("\n"))

    # Resize the frame to have maximum width of 500 pixels (the less
    # data we have, the faster we can process it), then convert the
    # frame from BGR to RGB for dlib
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Initialize the current status along with our list of bounding
    # box rectangles returned by either (1) the object detector, or
    # (2) the correlation trackers
    status = "Waiting"
    rects = []

    # Check to see if we should run a more computationally expensive
    # object detection method to aid the tracker
    if totalFrames % args["skip_frames"] == 0:
        # Set status and initialize a new trackers list
        status = "Detecting"
        trackers = []

        # Convert the frame to a blob and pass it through the network
        # and obtain the detections
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        net.setInput(blob)
        detections = net.forward()

        # Loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # Extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]
            # Filter out weak detections
            if confidence > args["confidence"]:
                # Extract the index of the class label from the
                # detections list
                idx = int(detections[0, 0, i, 1])
                # If the class label is not a person, ignore it
                if CLASSES[idx] != "person":
                    continue

                # Compute the (x, y) coordinates of the bounding box
                # for the object
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")
                # Construct a dlib rectangle object from the bounding
                # box coordinates and then start the dlib correlation
                # tracker
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track(rgb, rect)

                # Add the tracker to our list of trackers so we can
                # utilize it during skip frames
                trackers.append(tracker)
    # Otherwise, we should utilize our object trackers rather than
    # object detectors to obtain a higher fps
    else:
        # Loop over each tracker
        for tracker in trackers:
            # Set status to tracking
            status = "Tracking"
            # Update the tracker and grab the updated position
            tracker.update(rgb)
            pos = tracker.get_position()
            # Unpack the position object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())
            # Add the bounding box coordinates to the rectangles list
            rects.append((startX, startY, endX, endY))

    # Use the centroid tracker to associate the (1) old object
    # centroids with (2) the newly computed object centroids
    objects = ct.update(rects)

    # Loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # Check to see if a trackable object exists for the current
        # object ID
        to = trackableObjects.get(objectID, None)

        # If there is no existing trackable object, create one
        if to is None:
            to = TrackableObject(objectID, centroid)

        # Store the trackable object in our dictionary
        trackableObjects[objectID] = to

        # Draw the ID of the object and the centroid of the object on
        # the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 2, (0, 255, 0), -1)

    # Draw status on output frame
    text = "Status: {}".format(status)
    cv2.putText(frame, text, (10, H-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255), 2)

    # Check to see if we should write the frame to disk
    if writer is not None:
        writer.write(frame)

    # Show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # If the 'q' key was pressed, break from the loop
    if key == ord("q"):
        break

    # Increment the total number of frames processed thus far and then
    # update the FPS counter
    totalFrames += 1
    fps.update()

# Stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
print("[INFO] total frames: {:.2f}".format(totalFrames))
ct.grid.print_grid_as_array()

# Check to see if we need to release the video writer pointer
if writer is not None:
    writer.release()

# Close any open windows
cv2.destroyAllWindows()
