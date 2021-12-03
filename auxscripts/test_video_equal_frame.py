import cv2
import glob
import os
from numpy import array_equal

# Grab images
img_names = []
for filename in glob.glob("/home/cassio/CrowdedDataset/Choke1/*.jpg"):
    img_names.append(filename)
img_names = sorted(img_names, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))

# Initiating video writer object
out = cv2.VideoWriter("/home/cassio/CrowdedDataset/choke1_video.avi", cv2.VideoWriter_fourcc(*'DIVX'), 30, (800, 600), True)

for i in range(len(img_names)):
    # Read image from disk
    im = cv2.imread(img_names[i])

    # Write frame to video
    out.write(im)
    print("wrote frame {} of video {}".format(i, "choke1_video.avi"))
# Release the VideoWriter object
out.release()

vc = cv2.VideoCapture("/home/cassio/CrowdedDataset/street30fps.mp4")
success, image = vc.read()
count = 0

im = cv2.imread(img_names[100])

while success:
    if array_equal(image, im):
        print(count)
    success, image = vc.read()
    count += 1


