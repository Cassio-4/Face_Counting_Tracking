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


dataset_path_prefix = "/home/cassio/dataset/"
frames_path_prefix = "Images/"
gt_path_prefix = "CVAT-dataset-anotado/"
sequences = ["bar001/", "market001/", "street001/", "P2E_S5_C2/", "P2L_S5_C3.1/", "MOT17-01/", "MOT17-04/", "MOT17-09/"]
no_need_images = ["bar001/", "market001/", "street001/"]
six_digit_sequences = ["MOT17-01/", "MOT17-04/", "MOT17-09/"]
"""
# Copying and renaming images, except bar, market and street
for sequence in sequences:
    if sequence in no_need_images:
        continue
    frames_folder = dataset_path_prefix + frames_path_prefix + sequence
    # Reading all frames as list of tuples (frame, frame_name)
    all_frames = load_all_seq_frames(frames_folder)
    for frame in all_frames:
        new_name = "detection_dataset/images/" + sequence[:-1] + "_" + frame[1]
        cv2.imwrite(new_name, frame[0])
        pass
    del all_frames
"""
# Write a .txt file for each image
for sequence in sequences:
    gt_file = dataset_path_prefix + gt_path_prefix + sequence + "gt.txt"
    # Reading ground truth file
    gt = read_gt_as_csv(gt_file)
    for line in gt:
        frame_num = str(line[0])
        if sequence in six_digit_sequences:
            frame_num = frame_num.zfill(6)
        else:
            frame_num = frame_num.zfill(5)
        # if sequence is bar, street or market then separator is - instead of _
        if sequence in no_need_images:
            file_name = "detection_dataset/gt/" + sequence[:-1] + "-" + frame_num + ".txt"
        else:
            file_name = "detection_dataset/gt/" + sequence[:-1] + "_" + frame_num + ".txt"
        file_empty = False
        try:
            if os.stat(file_name).st_size == 0:
                file_empty = True
        except OSError:
            file_empty = True

        with open(file_name, 'a+') as f:
            # <class_name> <left> <top> <right> <bottom> [<difficult>]
            if file_empty:
                f.write("face {} {} {} {}".format(int(line[2]), int(line[3]), int(line[2] + line[4]), int(line[3] + line[5])))
                file_empty = False
            else:
                f.write("\nface {} {} {} {}".format(int(line[2]), int(line[3]), int(line[2] + line[4]), int(line[3] + line[5])))
        #    0       1      2           3         4            5          6     7    8    9
        # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
