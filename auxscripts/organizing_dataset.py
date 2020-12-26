import os


# Navigate to folder with all videos in it
og_path = "/home/cassio/PycharmProjects/Face_Counting_Tracking/dataset/market/event001/"
video_path = "/home/cassio/PycharmProjects/Face_Counting_Tracking/dataset/videos/market/market001/"
gt_path = "/home/cassio/PycharmProjects/Face_Counting_Tracking/dataset/groundtruth/market/market001/"
# Get all paths
all_xml = []
all_jpg = []
for r, d, f in os.walk(og_path):
    for file in f:
        if file.endswith("jpg"):
            all_jpg.append(os.path.join(r, file))
        elif file.endswith("xml"):
            all_xml.append(os.path.join(r, file))

# Rename all videos and gt files
for gt in all_xml:
    new_gt = gt.split('/')[-1].split('-')[0]
    new_gt = "market001-{}.xml".format(new_gt)
    # Move gt files
    os.rename(gt, gt_path+new_gt)

for jpg in all_jpg:
    new_jpg = jpg.split('/')[-1].split('-')[0]
    new_jpg = "market001-{}.jpg".format(new_jpg)
    os.rename(jpg, video_path+new_jpg)


