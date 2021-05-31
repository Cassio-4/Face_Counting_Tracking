from bounding_box import bounding_box as bb
from collections import Counter
import cv2
import csv
import os


def read_gt_as_csv(gt_file):
    gt = []
    with open(gt_file, mode='r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            type_converted_row = (int(row[0]), row[1], float(row[2]), float(row[3]), float(row[4]), float(row[5]),
                                  float(row[6]))
            gt.append(type_converted_row)
    return gt


def load_all_seq_frames(sequence_path):
    images = []
    for filename in os.listdir(sequence_path):
        img = cv2.imread(os.path.join(sequence_path, filename))
        if img is not None:
            images.append((img, filename))
    images.sort(key=lambda x: x[1])
    return images


def print_sequence_info(gt):
    ids = [int(gt[i][1]) for i in range(len(gt))]
    aux = Counter(ids).keys()
    print(aux)
    print("Total IDs: {}".format(len(aux)))
    print("total number of bboxes is {}.".format(len(gt)))
    # Smaller and larger Bbox
    bbox_areas = []
    for line in gt:
        bbox_areas.append(float(line[4]) * float(line[5]))
    bbox_areas.sort()
    print("smaller bbox is {} pxs; larger bbox is {} pxs.".format(bbox_areas[0], bbox_areas[-1]))


dataset_path_prefix = "/home/cassio/dataset/"
frames_path_prefix = "Images/"
gt_path_prefix = "CVAT-dataset-anotado/"
sequences = ["MOT17-01/", "MOT17-09/"]
#"bar001/", "market001/", "street001/", "P2E_S5_C2/", "P2L_S5_C3.1/", "MOT17-04/",
for sequence in sequences:
    gt_file = dataset_path_prefix + gt_path_prefix + sequence + "gt.txt"
    frames_folder = dataset_path_prefix + frames_path_prefix + sequence
    print("Now analysing sequence name {}.".format(sequence))
    # Reading ground truth file
    gt = read_gt_as_csv(gt_file)
    # Reading all frames as list of tuples (frame, frame_name)
    all_frames = load_all_seq_frames(frames_folder)

    # Printing sequence info
    print_sequence_info(gt)

    # Make directory to save annotated images
    if not os.path.exists("results/" + sequence):
        os.makedirs(os.path.dirname("results/" + sequence), exist_ok=True)

    # Write bounding boxes in each frame and save each frame
    num_frames = len(all_frames)
    for i in range(num_frames):
        frame = all_frames[i][0].copy()
        for line in gt:
            if line[0] == i+1:
                bb.add(frame, line[2], line[3], line[2]+line[4], line[3]+line[5], line[1], 'green')
        cv2.imwrite("results/"+sequence+"{}.jpg".format(i), frame)
    #    0       1      2           3         4            5          6     7    8    9
    # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>

"""
vc = cv2.VideoCapture(path)
W = vc.get(cv2.CAP_PROP_FRAME_WIDTH)
H = vc.get(cv2.CAP_PROP_FRAME_HEIGHT)
size = (int(W), int(H))

out = cv2.VideoWriter('{}.mp4'.format(video_sequence_name), cv2.VideoWriter_fourcc(*'DIVX'), 20, size)

while True:
    frame = vc.read()
    frame = frame[1]
    if frame is None:
        break
    out.write(frame)

out.release()
vc.release()
"""
