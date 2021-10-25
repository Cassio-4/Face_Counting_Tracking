import cv2
import glob
import os
from os import mkdir
from shutil import copy


# path to folder containing all videos and chokepoint folders
videos_folder_path = "/home/cassio/CrowdedDataset/"
# "concatenate frames from P2E_S5_C1.2, P2E_S5_C1.1 and P2E_S5_C1.3, in this specific order"
choke1_path = ["P2E_S5/P2E_S5_C2.1/", "P2E_S5/P2E_S5_C1.1/", "P2E_S5/P2E_S5_C3.1/"]
choke2_path = ["P2L_S5/P2L_S5_C2.1/",  "P2L_S5/P2L_S5_C3.1/", "P2L_S5/P2L_S5_C1.1/"]
# video names
street_path = "street30fps.mp4"
bengal_path = "bengal.mp4"
sidewalk_path = "sidewalk.mp4"
shibuya_path = "Shibuya.webm"
terminal_path = "Terminal30fps.mp4"
"""
# The expected directory structure is:
  CrowdedDataset/
    |- P2E_S5/
    |  |- P2E_S5_C1.1/
    |  |  |-...
    |  |- P2E_S5_C2.1/
    |  |  |-...
    |  |- P2E_S5_C3.1/
    |  |  |-...
    |- P2L_S5/
    |  |- P2L_S5_C1.1/
    |  |  |-...
    |  |- P2L_S5_C2.1/
    |  |  |-...
    |  |- P2L_S5_C3.1/
    |  |  |-...
    |- bengal.mp4
    |- sidewalk.mp4
    |- shibuya.mp4
    |- street.mp4
    |- terminal.mp4
"""


def concatenate_choke(choke_path, choke_name):
    # from frames 99 to 136 there are no frames, but there are annotations, therefore we must fill them
    fill_images_choke1 = {"P2E_S5/P2E_S5_C2.1/": videos_folder_path+"P2E_S5/P2E_S5_C2.1/00000099.jpg",
                          "P2E_S5/P2E_S5_C1.1/": videos_folder_path+"P2E_S5/P2E_S5_C1.1/00000099.jpg",
                          "P2E_S5/P2E_S5_C3.1/": videos_folder_path+"P2E_S5/P2E_S5_C3.1/00000099.jpg"}

    all_image_paths = {}
    for sequence in choke_path:
        img_names = []
        for filename in glob.glob(videos_folder_path + sequence + "*.jpg"):
            img_names.append(filename)

        # Choke 1 has to fill frames from 99 to 136
        if choke_name == "Choke1":
            fill_list = [fill_images_choke1[sequence]] * 36
            img_names.extend(fill_list)

        # Sort all sequence frames
        img_names = sorted(img_names, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))

        # Choke 2 has to remove a few frames from the very beginning, thus why we have to sort them first
        if choke_name == "Choke2":
            #if sequence == choke2_path[0]:
            del img_names[0: 42]

        all_image_paths[sequence] = img_names
    # got all paths, time to copy them to a new folder with new names
    concatenated_images = [*all_image_paths[choke_path[0]], *all_image_paths[choke_path[1]],
                           *all_image_paths[choke_path[2]]]
    count = 0
    destination_path = videos_folder_path + choke_name + "/"
    mkdir(videos_folder_path + choke_name)
    for image in concatenated_images:
        copy(image, destination_path+"{}.jpg".format(count))
        count += 1


def separate_street_video():
    destination_folder = videos_folder_path + "Street"
    mkdir(destination_folder)
    vc = cv2.VideoCapture(videos_folder_path + street_path)
    success, image = vc.read()
    count = 0
    # Then, cut it from the beginning (frame 0) to frame 2041.
    while success and count < 2042:
        cv2.imwrite(destination_folder + "/" + "%d.jpg" % count, image)
        success, image = vc.read()
        print('Read frame: %d ' % count, success)
        count += 1


def separate_sidewalk_video():
    destination_folder = videos_folder_path + "Sidewalk"
    mkdir(destination_folder)
    vc = cv2.VideoCapture(videos_folder_path + sidewalk_path)
    success, image = vc.read()
    count = 0
    # Cut it from frame 140 to frame 1436.
    while success and count < 1437:
        if count >= 140 and (count % 2 == 0):
            cv2.imwrite(destination_folder + "/" + "%d.jpg" % (count-140), image)
        success, image = vc.read()
        print('Read frame: %d ' % count, success)
        count += 1


def separate_bengal_video():
    destination_folder = videos_folder_path + "Bengal"
    mkdir(destination_folder)
    vc = cv2.VideoCapture(videos_folder_path + bengal_path)
    success, image = vc.read()
    count = 0
    # Then, cut it from frame 8475 to frame 9474.
    while success and count < 9475:
        if count >= 8475:
            cv2.imwrite(destination_folder + "/" + "%d.jpg" % (count - 8475), image)
        success, image = vc.read()
        print('Read frame: %d ' % count, success)
        count += 1


def separate_terminal_video():
    vc = cv2.VideoCapture(videos_folder_path + terminal_path)
    success, image = vc.read()
    count = 0

    destination_folder = videos_folder_path + "Terminal1"
    mkdir(destination_folder)
    print('beginning Terminal1')
    # Terminal 1: From frame 2400 to frame 4740, both included.
    # 4741 + 1 = 4742
    while success and count < 4742:
        # 2400 + 1 = 2401
        if count >= 2401:
            cv2.imwrite(destination_folder + "/" + "%d.jpg" % (count - 2401), image)
        success, image = vc.read()
        print('Read frame: %d ' % count, success)
        count += 1
    print("end of terminal1 frames")

    # Terminal 4: From 00:06:45 to 00:07:21. to a total of 1070 frames
    print('beginning Terminal4')
    destination_folder = videos_folder_path + "Terminal4"
    mkdir(destination_folder)
    # 13220 + 11 = 13231
    while success and count < 13231:
        # 12150 + 11 = 12161
        if count >= 12161:
            cv2.imwrite(destination_folder + "/" + "%d.jpg" % (count - 12161), image)
        success, image = vc.read()
        print('Read frame: %d ' % count, success)
        count += 1
    print("end of terminal4 frames")

    # Terminal 3: From 00:19:49 to 00:20:15. to a total of 771 frames
    print('beginning Terminal3')
    destination_folder = videos_folder_path + "Terminal3"
    mkdir(destination_folder)
    # 36441 + 11 = 36452
    while success and count < 36452:
        # 35670 + 11 = 35681
        if count >= 35681:
            cv2.imwrite(destination_folder + "/" + "%d.jpg" % (count - 35681), image)
        success, image = vc.read()
        print('Read frame: %d ' % count, success)
        count += 1
    print("end of terminal3 frames")

    # Terminal 2: From 00:23:37 to 00:24:52. to a total of 2240 frames
    print('beginning Terminal2')
    destination_folder = videos_folder_path + "Terminal2"
    mkdir(destination_folder)
    # 44750 + 11 = 44761
    while success and count < 44761:
        #42510 + 11 = 42521
        if count >= 42521:
            cv2.imwrite(destination_folder + "/" + "%d.jpg" % (count - 42521), image)
        success, image = vc.read()
        print('Read frame: %d ' % count, success)
        count += 1
    print("end of terminal2 frames")


def separate_shibuya_video():
    destination_folder = videos_folder_path + "Shibuya"
    mkdir(destination_folder)
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
            cv2.imwrite(destination_folder + "/" + "%d.jpg" % (count - 8700), image)
        success, image = vc.read()
        print('Read frame: %d ' % count, success)
        count += 1


#concatenate_choke(choke1_path, "Choke1")
concatenate_choke(choke2_path, "Choke2")
#separate_street_video()
#separate_sidewalk_video()
#separate_bengal_video()
#separate_terminal_video()
#separate_shibuya_video()
