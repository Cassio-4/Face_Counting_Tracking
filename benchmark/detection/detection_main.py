from benchmark.detection.detection_methods import *
from imutils.video import FPS
import argparse
import cv2
"""
This is a script for running various face detection methods on the 
FDDB: Face Detection Data Set and Benchmark 
http://vis-www.cs.umass.edu/fddb/
It outputs the hypothesis as a .txt file, according to the evaluation 
rules required by the dataset on their site.
Programmer: Cassio B. Nascimento
"""


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", required=True, type=str,
                default='opencv_haar', help="The detector to be used")

args = vars(ap.parse_args())

if args["detector"] == "opencv_haar":
    detector = OpenCVHaarFaceDetector()
else:
    detector = OpenCVHaarFaceDetector()

# Files with image names to be tested
text_files = ("FDDB-fold-01.txt", "FDDB-fold-02.txt", "FDDB-fold-03.txt",
              "FDDB-fold-04.txt", "FDDB-fold-05.txt", "FDDB-fold-06.txt",
              "FDDB-fold-07.txt", "FDDB-fold-08.txt", "FDDB-fold-09.txt",
              "FDDB-fold-10.txt")

# Start the frames per second throughput estimator
fps = FPS().start()
# Open output file to write results on
output_file = open("outputs/" + args["detector"] + ".txt", "w")

# Loop through all 10 files with image names
for file in text_files:
    # Open the file
    txt_file = open("datasets/" + file, "r")
    # Loop through names inside file
    for image_name in txt_file.readlines():
        # Get full frame path
        image_file = image_name.strip("\n") + ".jpg"
        path_to_frame = "datasets/FDDB_dataset/originalPics/" + image_file
        # load image
        image = cv2.imread(path_to_frame)
        # Send frame to desired detector
        detections = detector.detect_face(image)
        # Step with the FPS counter
        fps.update()
        # Write image name to output file
        output_file.write(image_name)
        # Write number of faces detected in this image
        output_file.write("{}\n".format(str(len(detections))))
        for result in detections:
            output_file.write("{} {} {} {}\n".format(result[0], result[1],
                                                     result[2], result[3]))
    txt_file.close()
# Before closing the output file write FPS info
fps.stop()
output_file.write("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
output_file.write("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# Finally close output file
output_file.close()
