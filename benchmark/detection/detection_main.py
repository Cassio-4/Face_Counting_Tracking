from benchmark.detection.detection_methods import DetectorHolder
from imutils.video import FPS
import argparse
import json
import cv2

"""
This is a script for running various face detection methods on the 
FDDB: Face Detection Data Set and Benchmark 
http://vis-www.cs.umass.edu/fddb/
It also saves the hypothesis result from a detector in a JSON file.
Programmer: Cassio B. Nascimento
Last change: 23/07/2020
"""

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", type=int, default=1,
                help="The detector to be used")
ap.add_argument("-i", "--input", type=str, required=True,
                help="Path to dataset's folder")
ap.add_argument("-o", "--output", type=str,
                help="Path to output hypothesis file")
ap.add_argument("-p", "--prototxt", default=None,
                help="Path to prototxt file")
ap.add_argument("-m", "--model", default=None,
                help="Path to model's weights")
ap.add_argument("-c", "--confidence", type=float, default=0.7,
                help="Minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Instantiate the detector holder class with all the detectors in it
if args.get("protoxt") is not None:
    detectors = DetectorHolder(args["detector"], args["protoxt"], args["model"])
else:
    detectors = DetectorHolder(args["detector"])

# Files with image names to be tested
text_files = ("FDDB-fold-01.txt", "FDDB-fold-02.txt", "FDDB-fold-03.txt",
              "FDDB-fold-04.txt", "FDDB-fold-05.txt", "FDDB-fold-06.txt",
              "FDDB-fold-07.txt", "FDDB-fold-08.txt", "FDDB-fold-09.txt",
              "FDDB-fold-10.txt",)

# Start the frames per second throughput estimator
fps = FPS().start()
# Start the dictionary that will hold the info to be made into a JSON
info_dict = {}
# Open output file to write results on
output_file = open(args["output"] + str(args["detector"]) + ".json", "w")
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
        # Send frame to desired detector
        detections = detectors.detect(frame)

        # Save detections and other info to info_dict
        info_dict["frame"] = image_name
        detection_list = []
        for coord in detections:
            detection_list.append({"x": coord[0], "y": coord[1], "w": coord[2],
                                   "h": coord[3]})
        info_dict["detections"] = detection_list

        # Write the JSON to the output file
        output_file.write(json.dumps(info_dict) + "\n")
        # Clear the dictionary for the next frame
        info_dict.clear()
    txt_file.close()
output_file.close()
