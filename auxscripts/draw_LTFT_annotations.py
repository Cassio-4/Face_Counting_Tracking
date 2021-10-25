import os
import cv2
import glob

# path to folder containing all videos and chokepoint folders
common_folder_path = "/home/cassio/CrowdedDataset/"
frames_path = {"choke1_path": "Choke1/", "choke2_path": "Choke2/", "street_path": "Street/", "bengal_path": "Bengal/",
               "sidewalk_path": "Sidewalk/", "shibuya_path": "Shibuya/", "terminal1_path": "Terminal1/",
               "terminal2_path": "Terminal2/", "terminal3_path": "Terminal3/", "terminal4_path": "Terminal4/"}
annotations_path = {"bengal_gt": "Annotations/bengal.txt", "choke1_gt": "Annotations/choke1.txt",
                    "choke2_gt": "Annotations/choke2.txt", "street_gt": "Annotations/street.txt",
                    "shibuya_gt": "Annotations/shibuya.txt", "sidewalk_gt": "Annotations/sidewalk.txt",
                    "terminal1_gt": "Annotations/terminal_1.txt", "terminal2_gt": "Annotations/terminal_2.txt",
                    "terminal3_gt": "Annotations/terminal_3.txt", "terminal4_gt": "Annotations/terminal_4.txt"}
framerates = {"choke_fps": 30, "street_fps": 30, "sidewalk_fps": 24, "bengal_fps": 25, "terminal_fps": 30,
              "shibuya_fps": 25}
video_dimensions = {"choke_size": (800, 600), "full_hd_size": (1920, 1080), "shibuya_size": (3840, 2160)}


class FrameAnnotation:
    def __init__(self, line):
        line = line.split(' ')
        self.frame_id = int(line.pop(0))
        self.num_detections = int(line.pop(0))
        # where each detection (#det_1, #det_2, etc) corresponds to:
        self.detections_dictionary = {}
        if self.num_detections <= 0:
            return
        for i in range(self.num_detections):
            # det_id #pos_x #pos_y #width #height #face #confidence
            det_id = int(line.pop(0))
            pos_x = int(round(float(line.pop(0))))
            pos_y = int(round(float(line.pop(0))))
            width = int(round(float(line.pop(0))))
            height = int(round(float(line.pop(0))))
            face = bool(line.pop(0))
            confidence = float(line.pop(0))
            detection = {"pos_x": pos_x, "pos_y": pos_y, "width": width, "height": height, "face": face,
                         "confidence": confidence}
            self.detections_dictionary[det_id] = detection
        if line:
            print('Line parsing went wrong, line list is not empty')
            exit(9)


def parse_annotations(lines):
    count = 0
    parsed_annotation = {}
    for line in lines:
        if count == 0:
            count += 1
            continue
        fa = FrameAnnotation(line)
        parsed_annotation[fa.frame_id] = fa
        count += 1
    """
    dictionary of frame_id: FrameAnnotation
    """
    return parsed_annotation


def draw_and_create_video(path, gt, size, fps, video_name):
    # Grab images
    img_names = []
    for filename in glob.glob(common_folder_path + path + "*.jpg"):
        img_names.append(filename)
    img_names = sorted(img_names, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
    # Get all annotations as lines
    with open(common_folder_path + gt) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    # Parse all groundtruth annotations
    parsed_annotations = parse_annotations(lines)
    # Initiating video writer object
    out = cv2.VideoWriter(common_folder_path+video_name, cv2.VideoWriter_fourcc(*'DIVX'), fps, size, True)

    for i in range(len(img_names)):
        # Read image from disk
        im = cv2.imread(img_names[i])
        # Draw Bboxes on image
        popped_annotation = parsed_annotations.pop(i, None)
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
                    # Write detection id on the bbox
                    im = cv2.putText(im, str(key), (pos_x+1, pos_y+1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA,
                                     bottomLeftOrigin=False)
        else:
            print("Desired frame key %d does not exist in annotations" % i)
            exit(2)

        # Write frame to video
        out.write(im)
        print("wrote frame {} of video {}".format(i, video_name))
    # Release the VideoWriter object
    out.release()

"""
draw_and_create_video(path=frames_path["choke1_path"], gt=annotations_path["choke1_gt"],
                      size=video_dimensions["choke_size"], fps=framerates["choke_fps"], video_name="choke1_bboxes.avi")
"""
draw_and_create_video(path=frames_path["choke2_path"], gt=annotations_path["choke2_gt"],
                      size=video_dimensions["choke_size"], fps=framerates["choke_fps"], video_name="choke2_bboxes_faceOnly.avi")
"""
draw_and_create_video(path=frames_path["street_path"], gt=annotations_path["street_gt"],
                      size=video_dimensions["full_hd_size"], fps=framerates["street_fps"], video_name="street_bboxes_faceonly.avi")
#draw_and_create_video(path=frames_path["sidewalk_path"], gt=annotations_path["sidewalk_gt"],
                      size=video_dimensions["full_hd_size"], fps=framerates["sidewalk_fps"],
                      video_name="sidewalk_bboxes_faceonly.avi")
draw_and_create_video(path=frames_path["bengal_path"], gt=annotations_path["bengal_gt"],
                      size=video_dimensions["full_hd_size"], fps=framerates["bengal_fps"],
                      video_name="bengal_bboxes_faceonly.avi")

draw_and_create_video(path=frames_path["terminal1_path"], gt=annotations_path["terminal1_gt"],
                      size=video_dimensions["full_hd_size"], fps=framerates["terminal_fps"],
                      video_name="terminal1_bboxes_faceonly.avi")

draw_and_create_video(path=frames_path["terminal2_path"], gt=annotations_path["terminal2_gt"],
                      size=video_dimensions["full_hd_size"], fps=framerates["terminal_fps"],
                      video_name="terminal2_bboxes_faceonly.avi")

draw_and_create_video(path=frames_path["terminal3_path"], gt=annotations_path["terminal3_gt"],
                      size=video_dimensions["full_hd_size"], fps=framerates["terminal_fps"],
                      video_name="terminal3_bboxes_faceonly.avi")

draw_and_create_video(path=frames_path["terminal4_path"], gt=annotations_path["terminal4_gt"],
                      size=video_dimensions["full_hd_size"], fps=framerates["terminal_fps"],
                      video_name="terminal4_bboxes_faceonly.avi")

draw_and_create_video(path=frames_path["shibuya_path"], gt=annotations_path["shibuya_gt"],
                      size=video_dimensions["shibuya_size"], fps=framerates["shibuya_fps"],
                      video_name="shibuya_bboxes_faceonly.avi")
"""

