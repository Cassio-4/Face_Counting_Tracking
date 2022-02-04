import random
import cv2
import os


def get_all_filenames_in_path(path):
    files = []
    for filename in os.listdir(path):
        files.append(filename)
    return files


gt_path = "detection_dataset/gt/"
images_path = "detection_dataset/images/"

all_gt = get_all_filenames_in_path(gt_path)

# choose random file
random_gt_file = random.choice(all_gt)
# get its image
image_name = random_gt_file[:-3] + "jpg"
img = cv2.imread(images_path+image_name, cv2.IMREAD_COLOR)
# read gt and paint
with open(gt_path + random_gt_file, "r") as f:
    for line in f:
        l = line.split("\n")[0]
        l = l.split(" ")
        # <class_name> <left> <top> <right> <bottom> [<difficult>]
        left = int(l[1])
        top = int(l[2])
        right = int(l[3])
        bottom = int(l[4])
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 1)
cv2.imwrite("TESTE"+image_name, img)