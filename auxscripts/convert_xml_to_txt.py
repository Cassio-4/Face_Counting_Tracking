import os
import xml.etree.ElementTree as ET


def get_all_files(path):
    # Get all paths
    all_xml = []
    for r, d, f in os.walk(path):
        for file in f:
            if file.endswith("xml"):
                all_xml.append(os.path.join(r, file))
    return all_xml


def convert_gt(path):
    # Get all annotated xml groundtruth files
    all_gt_files = get_all_files(path)
    # For each one, create a copy in txt with the name of the frame
    # With the format: class left top right bottom
    for gt in all_gt_files:
        file_name = gt.split('/')[-1].split('.')[0]
        tree = ET.parse(gt)
        root = tree.getroot()
        faces = []
        for child in root:
            # We found an object
            if child.tag == "object":
                # For the children of object we must find "name" and "bndbox"
                is_face = False
                for another_child in child:
                    if another_child.tag == "name":
                        if another_child.text == "face":
                            is_face = True
                        elif another_child.text == "person":
                            continue
                    if another_child.tag == "bndbox":
                        coord = [None, None, None, None]
                        for coordinates in another_child:
                            if coordinates.tag == "xmin":
                                coord[0] = int(coordinates.text)
                            elif coordinates.tag == "ymin":
                                coord[1] = int(coordinates.text)
                            elif coordinates.tag == "xmax":
                                coord[2] = int(coordinates.text)
                            elif coordinates.tag == "ymax":
                                coord[3] = int(coordinates.text)
                if is_face:
                    faces.append(coord)

                with open("{}{}.txt".format(path, file_name), "w") as txt:
                    for i in range(len(faces)):
                        if i == len(faces)-1:
                            txt.write("face {} {} {} {}".format(faces[i][0], faces[i][1], faces[i][2], faces[i][3]))
                        else:
                            txt.write("face {} {} {} {}\n".format(faces[i][0], faces[i][1], faces[i][2], faces[i][3]))


def convert_hp(path):
    # Get all annotated xml groundtruth files
    all_hp_files = get_all_files(path)
    # For each one, create a copy in txt with the name of the frame
    # With the format: class left top right bottom
    for hp in all_hp_files:
        tree = ET.parse(hp)
        root = tree.getroot()
        for child in root:
            try:
                file_name = child.attrib["name"]
            except KeyError:
                continue
            detections = []
            for grandchild in child:
                detections.append((int(grandchild.attrib["x_left"]), int(grandchild.attrib["y_top"]),
                                   int(grandchild.attrib["x_right"]), int(grandchild.attrib["y_bottom"])))
            with open("{}.txt".format(file_name.split('.')[0]), 'w') as txt:
                for i in range(len(detections)):
                    if i == len(detections)-1:
                        txt.write("face 0.8 {} {} {} {}".format(detections[i][0], detections[i][1], detections[i][2],
                                                                detections[i][3]))
                    else:
                        txt.write("face 0.8 {} {} {} {}\n".format(detections[i][0], detections[i][1], detections[i][2],
                                                                  detections[i][3]))


if __name__ == '__main__':
    gt = False
    # Navigate to folder with files in it
    path_to_file = "/home/cassio/PycharmProjects/Face_Counting_Tracking/output/"
    if gt:
        convert_gt(path_to_file)
    else:
        convert_hp(path_to_file)
