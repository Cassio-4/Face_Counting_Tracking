from benchmark.tracking.tracking_methods import *
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
ap.add_argument("-d", "--tracker", type=str, default='opencv_medianflow',
                required=True, help="The tracker to be used")

args = vars(ap.parse_args())
if args["tracker"] == "opencv_boosting":
    print("loading opencv_boosting")
    tracker = OpenCVMultitracker("BOOSTING")
elif args["tracker"] == "opencv_mil":
    print("loading opencv_mil")
    tracker = OpenCVMultitracker("MIL")
elif args["tracker"] == "opencv_kcf":
    print("loading opencv_kcf")
    tracker = OpenCVMultitracker("KCF")
elif args["tracker"] == "opencv_tld":
    print("loading opencv_tld")
    tracker = OpenCVMultitracker("TLD")
elif args["tracker"] == "opencv_csrt":
    print("loading opencv_csrt")
    tracker = OpenCVMultitracker("CSRT")
elif args["tracker"] == "opencv_goturn":
    print("loading opencv_goturn")
    tracker = OpenCVMultitracker("GOTURN")
elif args["tracker"] == "opencv_mosse":
    print("loading opencv_mosse")
    tracker = OpenCVMultitracker("MOSSE")
else:
    print("loading opencv_medianflow")
    tracker = OpenCVMultitracker("MEDIANFLOW")

# Start the frames per second throughput estimator
fps = FPS().start()

# Open output file to write results on
output_file = open("outputs/" + args["tracker"] + ".xml", "w")

# Open video
# cap = cv2.VideoCapture(videoPath)

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
