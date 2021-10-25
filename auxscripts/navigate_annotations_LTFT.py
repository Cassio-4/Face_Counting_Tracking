import cv2

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


def draw_and_save_image(path, gt, name, img_num):

    # Get all annotations as lines
    with open(common_folder_path + gt) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    # Parse all groundtruth annotations
    parsed_annotations = parse_annotations(lines)
    # Grab image
    og_image = cv2.imread(path)
    for i in range(img_num-50, img_num-25):
        # Get original image
        im = og_image.copy()
        # Get frame relative groundtruth
        popped_annotation = parsed_annotations.pop(i, None)
        # Draw Bboxes on image
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

        # Write frame
        cv2.imwrite(name+"choke2_frame160_ann{}.jpg".format(i), im)
        print("Wrote frame choke2_frame160_ann{}.jpg".format(i))

draw_and_save_image("/home/cassio/CrowdedDataset/Choke2/160.jpg", annotations_path["choke2_gt"],
"/home/cassio/CrowdedDataset/Sinc_Choke2/", 160)