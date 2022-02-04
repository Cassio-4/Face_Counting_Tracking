from draw_LTFT_annotations import parse_annotations

common_ltft_path = "/home/cassio/CrowdedDataset/annotations_TBIOM/"
ltft_annotations = ["bengal.txt", "choke1.txt", "choke2.txt", "shibuya.txt", "sidewalk.txt", "street.txt",
                    "terminal_1.txt", "terminal_2.txt", "terminal_3.txt", "terminal_4.txt"]

for ltft_ann in ltft_annotations:
    # Get all annotations as lines
    with open(common_ltft_path + ltft_ann) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    # Parse all groundtruth annotations
    parsed_annotations = parse_annotations(lines)
    new_name = ltft_ann.split(".")[0]
    with open("{}_mot.txt".format(new_name), 'w') as f:
        frame_num = 0
        line_num = 0
        while parsed_annotations:
            popped_annotation = parsed_annotations.pop(frame_num, None)
            if popped_annotation is not None:
                for key in popped_annotation.detections_dictionary:
                    pos_x = popped_annotation.detections_dictionary[key]["pos_x"]
                    pos_y = popped_annotation.detections_dictionary[key]["pos_y"]
                    w = popped_annotation.detections_dictionary[key]["width"]
                    h = popped_annotation.detections_dictionary[key]["height"]
                    face = popped_annotation.detections_dictionary[key]["face"]
                    conf = popped_annotation.detections_dictionary[key]["confidence"]
                    if face:
                        # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
                        line = "{}, {}, {}, {}, {}, {}, -1, -1, -1, -1".format(frame_num, key,  )
                        if line_num == 0:
                            f.write(line)


