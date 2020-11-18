import xml.etree.ElementTree as ET

root = ET.Element("dataset")
frame_info = ET.SubElement(root, "frame")

detection_info = ET.SubElement(frame_info, "person", id="01")
ET.SubElement(detection_info, "bbox", top_y="10", bottom_y="20", right_x="40", left_x="100")
ET.SubElement(detection_info, "bbox", top_y="10", bottom_y="20", right_x="40", left_x="100")
detection_info = ET.SubElement(frame_info, "person", id="01")
ET.SubElement(detection_info, "bbox", top_y="10", bottom_y="20", right_x="40", left_x="100")
ET.SubElement(detection_info, "bbox", top_y="10", bottom_y="20", right_x="40", left_x="100")
tree = ET.ElementTree(root)
tree.write("filename.xml")
