import glob
import cv2
import os


MOT_frames_path = {"01": "/home/cassio/dataset/Images/MOT17-01/", "04": "/home/cassio/dataset/Images/MOT17-04/",
                   "09": "/home/cassio/dataset/Images/MOT17-09/"}
MOT_video_names = {"01": "MOT17-01_video.avi", "04": "MOT17-04_video.avi", "09": "MOT17-09_video.avi"}
# For each sequence of frames
for key in MOT_frames_path:
    img_paths = []
    # Grab all sequence related frame paths
    sequence_path = MOT_frames_path[key] + "*.jpg"
    for filename in glob.glob(sequence_path):
        img_paths.append(filename)
    # Sort all paths
    img_paths = sorted(img_paths, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))

    # Initiating video writer object
    out = cv2.VideoWriter("/home/cassio/dataset/" + MOT_video_names[key], cv2.VideoWriter_fourcc(*'DIVX'), 30, (1920, 1080),
                          True)
    # Read each frame from disk and turn it into a vid
    for i in range(len(img_paths)):
        # Read image from disk
        im = cv2.imread(img_paths[i])
        # Write frame to video
        out.write(im)
        print("wrote frame {} of video {}".format(i, MOT_video_names[key]))
    # Release the VideoWriter object
    out.release()
