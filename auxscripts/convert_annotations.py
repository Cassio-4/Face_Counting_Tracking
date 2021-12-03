from draw_LTFT_annotations import FrameAnnotation, parse_annotations
import xml.etree.ElementTree as ET
from os import mkdir

annotations_path = {"LTFT_path": "/home/cassio/CrowdedDataset/",
                    "WildFace_path": "/home/cassio/dataset/CVAT-dataset-anotado/"}
LTFT_annotations_path = {"bengal_gt": "Annotations/bengal.txt", "choke1_gt": "Annotations/choke1.txt",
                         "choke2_gt": "Annotations/choke2.txt", "street_gt": "Annotations/street.txt",
                         "shibuya_gt": "Annotations/shibuya.txt", "sidewalk_gt": "Annotations/sidewalk.txt",
                         "terminal1_gt": "Annotations/terminal_1.txt", "terminal2_gt": "Annotations/terminal_2.txt",
                         "terminal3_gt": "Annotations/terminal_3.txt", "terminal4_gt": "Annotations/terminal_4.txt"}
MOT_style_my_annotations_path = {"MOT17-01": "MOT17-01/gt.txt", "MOT17-04": "MOT17-04/gt.txt",
                                 "MOT17-09": "MOT17-09/gt.txt"}


class FromMOTFrameAnnotation:
    def __init__(self, dic):
        self.detections_dictionary = dic

def parse_MOT_annotations(lines):
    # Dictionary where the key is the frame number and the value is another dictionary that maps detection id to values
    # {Frame_id -> {det_id -> {bb_left: x, bb_top: y...}}}
    parsed_annotation = {}
    # Each line of the MOT annotation is:
    # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
    for line in lines:
        # Split the line
        line = line.split(',')
        # Get the frame id (first value)
        frame_id = int(line[0])
        if frame_id not in parsed_annotation:
            parsed_annotation[frame_id] = {}
        # Get the detection id
        det_id = int(line[1])
        if det_id not in parsed_annotation[frame_id]:
            parsed_annotation[frame_id][det_id] = {}
        pos_x = int(round(float(line[2])))
        pos_y = int(round(float(line[3])))
        width = int(round(float(line[4])))
        height = int(round(float(line[5])))
        face = int(line[6])
        parsed_annotation[frame_id][det_id] = {"pos_x": pos_x, "pos_y": pos_y, "width": width, "height": height,
                                               "face": face, "confidence": face}
    return parsed_annotation


def read_txt_as_lines(txt_path):
    with open(txt_path) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    return lines


def write_xml(parsed_annotations, prefix, folder_name, frame_size, write_path):
    for i in range(len(parsed_annotations)):
        # Create xml root <annotation>
        data = ET.Element('annotation')
        folder = ET.SubElement(data, 'folder')
        folder.text = folder_name
        filename = ET.SubElement(data, 'filename')
        filename.text = "{}_{}.jpg".format(prefix, i)
        path = ET.SubElement(data, 'path')
        path.text = "./{}/{}_{}.jpg".format(folder_name, prefix, i)
        # source
        source = ET.SubElement(data, 'source')
        source.text = folder_name
        database = ET.SubElement(source, "database")
        database.text = folder_name
        # size
        size = ET.SubElement(data, 'size')
        width = ET.SubElement(size, 'width')
        width.text = str(frame_size[0])
        height = ET.SubElement(size, 'height')
        height.text = str(frame_size[1])
        depth = ET.SubElement(size, 'depth')
        depth = '3'
        # segmented
        segmented = ET.SubElement(data, 'segmented')
        segmented.text = '0'
        # add bounding boxes
        popped_annotation = parsed_annotations.pop(i, None)
        if popped_annotation is not None:
            for key in popped_annotation.detections_dictionary:
                # object
                obj = ET.SubElement(data, 'object')
                name = ET.SubElement(obj, 'name')
                name.text = "face"
                pose = ET.SubElement(obj, 'pose')
                pose.text = "Unspecified"
                truncated = ET.SubElement(obj, 'truncated')
                truncated.text = '0'
                difficult = ET.SubElement(obj, 'difficult')
                difficult.text = '0'
                # bounding box
                bndbox = ET.SubElement(obj, 'bndbox')
                xmin = ET.SubElement(bndbox, 'xmin')
                xmin.text = str(popped_annotation.detections_dictionary[key]["pos_x"])
                ymin = ET.SubElement(bndbox, 'ymin')
                ymin.text = str(popped_annotation.detections_dictionary[key]["pos_y"])
                xmax = ET.SubElement(bndbox, 'xmax')
                xmax.text = str(popped_annotation.detections_dictionary[key]["pos_x"] +
                                popped_annotation.detections_dictionary[key]["width"])
                ymax = ET.SubElement(bndbox, 'ymax')
                ymax.text = str(popped_annotation.detections_dictionary[key]["pos_y"] +
                                popped_annotation.detections_dictionary[key]["height"])
        else:
            print("Could not find annotation for the frame {} of sequence {}.".format(i, prefix))
            #exit(99)
        # create a new XML file with the results
        mydata = ET.tostring(data)
        myfile = open(write_path+"{}_{}.xml".format(prefix, i), "wb")
        myfile.write(mydata)


def LTFT_to_pascalVOC(ann_path, prefix, folder, size):
    # Read original annotation
    lines = read_txt_as_lines(ann_path)
    # Parse it, get a dictionary of "frame_id: frame annotation".
    parsed_annotations = parse_annotations(lines)
    # Create a folder to hold all the files (PASCAL VOC annotations are a single xml for each image)
    dir = "PascalVOC_Annotations"
    try:
        mkdir(annotations_path["LTFT_path"] + dir)
    except OSError as error:
        print("Directory already exists")
    # Write annotations as PascalVOC
    write_xml(parsed_annotations, prefix, folder, size, annotations_path["LTFT_path"] + dir + '/')


def MOT_to_pascalVOC(ann_path, prefix, folder, size):
    # Read original annotation
    lines = read_txt_as_lines(ann_path)
    # Parse it, get a dictionary of "frame_id: frame annotation".
    parsed_annotations = parse_MOT_annotations(lines)
    for key in parsed_annotations:
        parsed_annotations[key] = FromMOTFrameAnnotation(parsed_annotations[key])
    # Create a folder for the MOT to PASCAL VOC files
    dir = "MOT_to_PascalVOC_Annotations"
    try:
        mkdir(annotations_path["LTFT_path"] + dir)
    except OSError as error:
        print("Directory already exists")
    # Write annotations as PascalVOC
    write_xml(parsed_annotations, prefix, folder, size, annotations_path["LTFT_path"] + dir + '/')


def separate_into_many_files(ann_path):
    # Read original annotation
    lines = read_txt_as_lines(ann_path)
    print(lines[-1])
    name_ann_dict = {}
    for line in lines:
        # Split the line
        line = line.split(", ")
        # Separate each field
        score = float(line[1])
        xmin = int(float(line[2]))
        ymin = int(float(line[3]))
        xmax = int(float(line[4]))
        ymax = int(float(line[5]))
        # Build a tuple
        tup = (score, xmin, ymin, xmax, ymax)
        # if this frame already has an entry, just append to its list
        if line[0] in name_ann_dict:
            name_ann_dict[line[0]].append(tup)
        # else, add the new frame
        else:
            name_ann_dict[line[0]] = []
            name_ann_dict[line[0]].append(tup)
    # Navigate the dictionary of frame -> annotation
    for frame in name_ann_dict:
        with open("/home/cassio/CrowdedDataset/DetectionFiles/RetinaFace/{}.txt".format(frame), 'w') as f:
            ann_list = name_ann_dict[frame]
            i = 0
            for t in ann_list:
                if i == 0:
                    f.write('face {:.2f} {:d} {:d} {:d} {:d}'.format(t[0], t[1], t[2], t[3], t[4]))
                else:
                    f.write('\nface {:.2f} {:d} {:d} {:d} {:d}'.format(t[0], t[1], t[2], t[3], t[4]))
                i += 1


"""
LTFT_to_pascalVOC(annotations_path["LTFT_path"]+LTFT_annotations_path["choke1_gt"], "choke1", "Choke1", (800, 600))
LTFT_to_pascalVOC(annotations_path["LTFT_path"]+LTFT_annotations_path["choke2_gt"], "choke2", "Choke2", (800, 600))
LTFT_to_pascalVOC(annotations_path["LTFT_path"]+LTFT_annotations_path["street_gt"], "street", "Street", (1920, 1080))
LTFT_to_pascalVOC(annotations_path["LTFT_path"]+LTFT_annotations_path["sidewalk_gt"], "sidewalk", "Sidewalk", (1920, 1080))
LTFT_to_pascalVOC(annotations_path["LTFT_path"]+LTFT_annotations_path["bengal_gt"], "bengal", "Bengal", (1920, 1080))
LTFT_to_pascalVOC(annotations_path["LTFT_path"]+LTFT_annotations_path["terminal1_gt"], "terminal1", "Terminal1", (1920, 1080))
LTFT_to_pascalVOC(annotations_path["LTFT_path"]+LTFT_annotations_path["terminal2_gt"], "terminal2", "Terminal2", (1920, 1080))
LTFT_to_pascalVOC(annotations_path["LTFT_path"]+LTFT_annotations_path["terminal3_gt"], "terminal3", "Terminal3", (1920, 1080))
LTFT_to_pascalVOC(annotations_path["LTFT_path"]+LTFT_annotations_path["terminal4_gt"], "terminal4", "Terminal4", (1920, 1080))
LTFT_to_pascalVOC(annotations_path["LTFT_path"]+LTFT_annotations_path["shibuya_gt"], "shibuya", "Shibuya", (3840, 2160))

MOT_to_pascalVOC(annotations_path["WildFace_path"]+MOT_style_my_annotations_path["MOT17-01"], "MOT17-01", "MOT17-01",
                 (1920, 1080))
MOT_to_pascalVOC(annotations_path["WildFace_path"]+MOT_style_my_annotations_path["MOT17-04"], "MOT17-04", "MOT17-04",
                 (1920, 1080))
MOT_to_pascalVOC(annotations_path["WildFace_path"]+MOT_style_my_annotations_path["MOT17-09"], "MOT17-09", "MOT17-09",
                 (1920, 1080))"""
separate_into_many_files("/home/cassio/CrowdedDataset/retinaface_pascvoc_det_face.txt")

