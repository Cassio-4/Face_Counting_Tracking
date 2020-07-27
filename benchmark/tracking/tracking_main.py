from benchmark.tracking.tracking_methods import TrackingHolder
from imutils.video import FPS
import argparse
import cv2

"""
This is a script for running various face tracking methods on the
#TODO dataset 
It also saves the hypothesis result from a tracker in a XML file.
Programmer: Cassio B. Nascimento
"""

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--tracker", type=int, default=1,
                help="The tracker to be used")
ap.add_argument("-i", "--input", type=str, required=True,
                help="Path to dataset's folder")
ap.add_argument("-o", "--output", type=str,
                help="Path to output hypothesis file")

args = vars(ap.parse_args())

# Instantiate the tracker holder class with all the trackers in it
trackers = TrackingHolder(args["tracker"])

# Files with image names to be tested
text_files = ("", "")

# Start the frames per second throughput estimator
fps = FPS().start()

# Open output file to write results on
output_file = open(args["output"] + str(args["tracker"]) + ".xml", "w")

# Loop through all 10 files with image names
for file in text_files:
    # Open the file
    txt_file = open(args["input"] + "/FDDB-folds/" + file, "r")
    # Loop through names inside file
    for image_name in txt_file.readlines():
        # Get full frame path
        path_to_frame = args["input"] + "/originalPics/" + \
            image_name.strip("\n") + ".jpg"
        # load frame
        frame = cv2.imread(path_to_frame)
        # Send frame to desired tracker
        tracked_objects = trackers.track(frame)
        # Step with the FPS counter
        fps.update()
        # Output the tracking results to the output_file
        # TODO
    txt_file.close()
# Before closing the output file write FPS info
fps.stop()
output_file.write("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
output_file.write("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# Finally close output file
output_file.close()
