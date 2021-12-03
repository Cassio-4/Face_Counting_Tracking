import os
import cv2
import glob
from draw_LTFT_annotations import parse_annotations

# path to folder containing all videos and chokepoint folders
common_folder_path = "/home/cassio/CrowdedDataset/"
frames_path = {"choke1_path": "Choke1/", "choke2_path": "Choke2/", "street_path": "Street/", "bengal_path": "Bengal/",
               "sidewalk_path": "Sidewalk/", "shibuya_path": "Shibuya/", "terminal1_path": "Terminal1/",
               "terminal2_path": "Terminal2/", "terminal3_path": "Terminal3/", "terminal4_path": "Terminal4/"}
annotations_path = {"retinaface_det": "DetectionFiles/RetinaFace/",
                    "faceboxes_afw_det": "DetectionFiles/FaceBoxesPytorch_GPU_AFWscale/FaceBoxes_Pytorch_afwscale_dets_sco_x_y_x_y/"}
framerates = {"choke_fps": 30, "street_fps": 30, "sidewalk_fps": 24, "bengal_fps": 25, "terminal_fps": 30,
              "shibuya_fps": 25}
video_dimensions = {"choke_size": (800, 600), "full_hd_size": (1920, 1080), "shibuya_size": (3840, 2160)}


def get_det_file_path_from_frame_path(path_to_frame, prefix, list_of_txts):
    frame_num = path_to_frame.split('/')[-1].split('.')[0]
    txt_name = "{}{}.txt".format(prefix, frame_num)
    for i in list_of_txts:
        if txt_name in i:
            list_of_txts.remove(i)
            return i
    return None


def draw_and_create_video(imgs_path, det_path, prefix, size, fps, video_name):
    # Grab images
    img_names = []
    for filename in glob.glob(common_folder_path + "SequenceFrames/" + imgs_path + "*.jpg"):
        img_names.append(filename)
    img_names = sorted(img_names, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
    # Grab dets
    det_names = []
    for filename in glob.glob(common_folder_path + det_path + "{}*.txt".format(prefix)):
        det_names.append(filename)
    det_names = sorted(det_names, key=lambda i: int(os.path.splitext(os.path.basename(i))[0].split('_')[1]))

    """
    with open(common_folder_path + 'OG_Annotations/terminal_2.txt') as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
        # Parse all groundtruth annotations
    parsed_annotations = parse_annotations(lines)
    """
    # Initiating video writer object
    out = cv2.VideoWriter(common_folder_path + "Tests/" + video_name, cv2.VideoWriter_fourcc(*'DIVX'), fps, size, True)
    # Loop through every frame
    count = 0
    for frame in img_names:
        # Read frame
        im = cv2.imread(frame)
        # Get path to det annotation
        det_file_path = get_det_file_path_from_frame_path(frame, prefix, det_names)
        if det_file_path is not None:
            # Get lines
            with open(det_file_path) as file:
                lines = file.readlines()
                lines = [line.rstrip() for line in lines]
            # Draw boxes over frame
            for l in lines:
                l = l.split(' ')
                score = float(l[1])
                xmin = int(float(l[2]))
                ymin = int(float(l[3]))
                xmax = int(float(l[4]))
                ymax = int(float(l[5]))
                # Draw a bbox from the annotation
                im = cv2.rectangle(im, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                # Write detection id on the bbox
                #im = cv2.putText(im, "{:.2f}".format(score), (xmin, ymin - 3), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA,
                #                 bottomLeftOrigin=False)
        """
        popped_annotation = parsed_annotations.pop(count, None)
        if popped_annotation is not None:
            for key in popped_annotation.detections_dictionary:
                pos_x = popped_annotation.detections_dictionary[key]["pos_x"]
                pos_y = popped_annotation.detections_dictionary[key]["pos_y"]
                w = popped_annotation.detections_dictionary[key]["width"]
                h = popped_annotation.detections_dictionary[key]["height"]
                face = popped_annotation.detections_dictionary[key]["face"]
                conf = popped_annotation.detections_dictionary[key]["confidence"]
                if face:
                    # Draw a bbox from the annotation
                    im = cv2.rectangle(im, (pos_x, pos_y), (pos_x + w, pos_y + h), (0, 255, 0), 2)
        """
        # Write frame on video
        out.write(im)
        print("wrote frame {}".format(frame.split('/')[-1]))
        count += 1
    # release video writer
    out.release()

"""
draw_and_create_video(imgs_path=frames_path["choke1_path"], det_path=annotations_path["retinaface_det"],
                      prefix="choke1_", size=video_dimensions["choke_size"], fps=framerates["choke_fps"],
                      video_name="choke1_DETS_vs_GT.avi")

draw_and_create_video(imgs_path=frames_path["terminal1_path"], det_path=annotations_path["retinaface_det"],
                      prefix="terminal1_", size=video_dimensions["full_hd_size"], fps=framerates["terminal_fps"],
                      video_name="terminal1_DETS.avi")
"""
draw_and_create_video(imgs_path=frames_path["terminal2_path"], det_path=annotations_path["faceboxes_afw_det"],
                      prefix="terminal2_", size=video_dimensions["full_hd_size"], fps=framerates["terminal_fps"],
                      video_name="terminal2_faceboxes_DETS.avi")

