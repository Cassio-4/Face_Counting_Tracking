import cv2
import glob
import os
from os import mkdir
from shutil import copy


# path to folder containing all videos and chokepoint folders
videos_folder_path = "/home/cassio/CrowdedDataset/"
# Frames path
choke1_path = "Choke1"
choke2_path = "Choke2"
# video names
street_path = "street30fps.mp4"
bengal_path = "bengal.mp4"
sidewalk_path = "sidewalk.mp4"
shibuya_path = "Shibuya.webm"
terminal_path = "Terminal30fps.mp4"


def create_choke_videos():
    # Get the path for all of Choke1 frames
    choke1_images = []
    for filename in glob.glob(videos_folder_path+choke1_path + "/*.jpg"):
        choke1_images.append(filename)
        # Sort all paths
    choke1_images = sorted(choke1_images, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
    # Create VideoWriter for Choke1
    out1 = cv2.VideoWriter("/home/cassio/CrowdedDataset/Choke1_CutVideo.avi", cv2.VideoWriter_fourcc(*'DIVX'), 30,
                           (800, 600), True)
    # Write Choke1 video
    for i in range(len(choke1_images)):
        im = cv2.imread(choke1_images[i])
        # Write frame to video
        out1.write(im)
        print("wrote frame {} of video {}".format(i, choke1_path))
    out1.release()

    # Get the path to all of Choke2 frames
    choke2_images = []
    for filename in glob.glob(videos_folder_path+choke2_path + "/*.jpg"):
        choke2_images.append(filename)
        # Sort all paths
    choke2_images = sorted(choke2_images, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
    # Create VideoWriter for Choke2
    out2 = cv2.VideoWriter("/home/cassio/CrowdedDataset/Choke2_CutVideo.avi", cv2.VideoWriter_fourcc(*'DIVX'), 30,
                           (800, 600), True)
    # Write Choke2 video
    for i in range(len(choke2_images)):
        im = cv2.imread(choke2_images[i])
        # Write frame to video
        out2.write(im)
        print("wrote frame {} of video {}".format(i, choke2_path))
    out2.release()


def separate_street_video():
    out = cv2.VideoWriter("/home/cassio/CrowdedDataset/Street_CutVideo.avi", cv2.VideoWriter_fourcc(*'DIVX'), 30,
                          (1920, 1080), True)
    vc = cv2.VideoCapture(videos_folder_path + street_path)
    count = 0
    success, image = vc.read()
    print('Read frame: %d ' % count, success)
    # Then, cut it from the beginning (frame 0) to frame 2041.
    while success and count < 2042:
        out.write(image)
        print("wrote frame {} of video {}".format(count, "Street_CutVideo.avi"))
        success, image = vc.read()
        count += 1
        print('Read frame: %d ' % count, success)
    # Release the VideoWriter object
    out.release()


def separate_sidewalk_video():
    out = cv2.VideoWriter("/home/cassio/CrowdedDataset/Sidewalk_CutVideo.avi", cv2.VideoWriter_fourcc(*'DIVX'), 30,
                          (1920, 1080), True)
    vc = cv2.VideoCapture(videos_folder_path + sidewalk_path)
    count = 0
    success, image = vc.read()
    print('Read frame: %d ' % count, success)
    # Cut it from frame 140 to frame 1436.
    while success and count < 1437:
        if count >= 140 and (count % 2 == 0):
            out.write(image)
            print("wrote frame {} of video {}".format(count, "Sidewalk_CutVideo.avi"))
        success, image = vc.read()
        count += 1
        print('Read frame: %d ' % count, success)
    out.release()


def separate_bengal_video():
    out = cv2.VideoWriter("/home/cassio/CrowdedDataset/Bengal_CutVideo.avi", cv2.VideoWriter_fourcc(*'DIVX'), 30,
                          (1920, 1080), True)
    vc = cv2.VideoCapture(videos_folder_path + bengal_path)
    count = 0
    success, image = vc.read()
    print('Read frame: %d ' % count, success)
    # Then, cut it from frame 8475 to frame 9474.
    while success and count < 9475:
        if count >= 8475:
            out.write(image)
            print("wrote frame {} of video {}".format(count, "Bengal_CutVideo.avi"))
        success, image = vc.read()
        count += 1
        print('Read frame: %d ' % count, success)
    out.release()


def separate_terminal_video():
    vc = cv2.VideoCapture(videos_folder_path + terminal_path)
    success, image = vc.read()
    count = 0
    print('Read frame: %d ' % count, success)

    out1 = cv2.VideoWriter("/home/cassio/CrowdedDataset/Terminal1_CutVideo.avi", cv2.VideoWriter_fourcc(*'DIVX'), 30,
                           (1920, 1080), True)
    print('beginning Terminal1')
    # Terminal 1: From frame 2400 to frame 4740, both included.
    # 4741 + 1 = 4742
    while success and count < 4742:
        # 2400 + 1 = 2401
        if count >= 2401:
            out1.write(image)
            print("wrote frame {} of video {}".format(count, "Terminal1_CutVideo.avi"))
        success, image = vc.read()
        count += 1
        print('Read frame: %d ' % count, success)
    out1.release()
    print("end of terminal1 frames")

    # Terminal 4: From 00:06:45 to 00:07:21. to a total of 1070 frames
    print('beginning Terminal4')
    out4 = cv2.VideoWriter("/home/cassio/CrowdedDataset/Terminal4_CutVideo.avi", cv2.VideoWriter_fourcc(*'DIVX'), 30,
                           (1920, 1080), True)
    # 13220 + 11 = 13231
    while success and count < 13231:
        # 12150 + 11 = 12161
        if count >= 12161:
            out4.write(image)
            print("wrote frame {} of video {}".format(count, "Terminal4_CutVideo.avi"))
        success, image = vc.read()
        count += 1
        print('Read frame: %d ' % count, success)
    out4.release()
    print("End of Terminal4 frames")

    # Terminal 3: From 00:19:49 to 00:20:15. to a total of 771 frames
    print('beginning Terminal3')
    out3 = cv2.VideoWriter("/home/cassio/CrowdedDataset/Terminal3_CutVideo.avi", cv2.VideoWriter_fourcc(*'DIVX'), 30,
                           (1920, 1080), True)
    # 36441 + 11 = 36452
    while success and count < 36452:
        # 35670 + 11 = 35681
        if count >= 35681:
            out3.write(image)
            print("wrote frame {} of video {}".format(count, "Terminal3_CutVideo.avi"))
        success, image = vc.read()
        count += 1
        print('Read frame: %d ' % count, success)
    out3.release()
    print("end of terminal3 frames")

    # Terminal 2: From 00:23:37 to 00:24:52. to a total of 2240 frames
    print('beginning Terminal2')
    out2 = cv2.VideoWriter("/home/cassio/CrowdedDataset/Terminal2_CutVideo.avi", cv2.VideoWriter_fourcc(*'DIVX'), 30,
                           (1920, 1080), True)
    # 44750 + 11 = 44761
    while success and count < 44761:
        #42510 + 11 = 42521
        if count >= 42521:
            out2.write(image)
            print("wrote frame {} of video {}".format(count, "Terminal2_CutVideo.avi"))
        success, image = vc.read()
        count += 1
        print('Read frame: %d ' % count, success)
    out2.release()
    print("End of Terminal2 frames")


def separate_shibuya_video():
    out = cv2.VideoWriter("/home/cassio/CrowdedDataset/Shibuya_CutVideo.avi", cv2.VideoWriter_fourcc(*'DIVX'), 30,
                          (3840, 2160), True)
    vc = cv2.VideoCapture(videos_folder_path + shibuya_path)
    success, image = vc.read()
    count = 0
    # Then, cut it from 00:04:50 to 00:05:20.
    # 8142 - 7 = 8135
    # 30 fps -> 9602
    while success and count < 9592:
        # 7250 - 7 = 7243
        # 29 fps -> 8710
        if count >= 8700:
            out.write(image)
            print("wrote frame {} of video {}".format(count, "Shibuya_CutVideo.avi"))
        success, image = vc.read()
        count += 1
        print('Read frame: %d ' % count, success)
    out.release()


create_choke_videos()
"""
separate_street_video()
separate_sidewalk_video()
separate_bengal_video()
separate_terminal_video()
separate_shibuya_video()
"""