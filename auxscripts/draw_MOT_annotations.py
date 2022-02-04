import cv2
import glob
import os


def centroid_from_bbox(rect):
    """
    calculate centroid of a bounding box
    :param rect: 4 numbers corresponding to a box coordinate [xmin, ymin, xmax, ymax]
    :return: a coordinate tuple (x, y)
    """
    # use the bounding box coordinates to derive the centroid
    cx = int((rect[0] + rect[2]) / 2.0)
    cy = int((rect[1] + rect[3]) / 2.0)
    return (cx, cy)


def parse_MOTstyle_annotations(file_path):
    with open(file_path) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    frame_to_annotation_dict = {}
    for line in lines:
        l = line.split(", ")
        l = [int(i) for i in l]
        frame = l[0]
        id = l[1]
        xmin = l[2]
        ymin = l[3]
        xmax = xmin + l[4]
        ymax = ymin + l[5]
        # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        if not (frame in frame_to_annotation_dict):
            frame_to_annotation_dict[frame] = []
        ann_tuple = (id, xmin, ymin, xmax, ymax)
        frame_to_annotation_dict[frame].append(ann_tuple)
    return frame_to_annotation_dict


def get_image_names_ordered(path_to_folder):
    img_names = []
    for filename in glob.glob(path_to_folder + "*.jpg"):
        img_names.append(filename)
    img_names = sorted(img_names, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
    return img_names


def draw_tracks(path_to_images_folder, path_to_txt, fps, dimension):
    img_names = get_image_names_ordered(path_to_images_folder)
    frame_to_ann_dict = parse_MOTstyle_annotations(path_to_txt)
    # Create VideoWriter
    video_name = img_names[0].split("/")[-2]
    video_name = "{}_tracks".format(video_name)
    out1 = cv2.VideoWriter("/home/cassio/Videos/{}.avi".format(video_name), cv2.VideoWriter_fourcc(*'DIVX'), fps,
                           dimension, True)
    for img in img_names:
        frame = cv2.imread(img)
        img_num = int(img.split("/")[-1].split(".")[0])
        list_of_frame_annotations = frame_to_ann_dict[img_num]
        if not(list_of_frame_annotations is None):
            for tup in list_of_frame_annotations:
                frame = cv2.rectangle(frame, (tup[1], tup[2]), (tup[3], tup[4]), (10, 240, 10), 2)
                centroid = centroid_from_bbox((tup[1:]))
                cv2.putText(frame, str(tup[0]), (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 240, 10), 2)
        out1.write(frame)
    out1.release()


draw_tracks("/home/cassio/dataset/Images/MOT17-09/",
            "/home/cassio/PycharmProjects/LTFT-implementation/data/results/MOT17-09.txt", 30, (1920, 1080))


