import xml.etree.ElementTree as ET
from imutils.video import FPS
import config
import cv2
"""
This is a script for running various face detectors and trackers methods
It output the hypothesis as a xml file.
Programmer: Cassio B. Nascimento
"""


def draw_boxes_on_image(image, detections):
    """
    Draws bounding boxes and Ids on image
    :param detections: list/array with shape (num_faces, 4)
    :param image: the image to draw on (using opencv)
    :return: the drawn on frame
    """
    image_copy = image.copy()
    for box in detections:
        cv2.rectangle(image_copy, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
    return image_copy


def get_all_image_names_as_list(path):
    path_to_txt = path + 'all_images.txt'
    image_paths = []
    image_names = []
    with open(path_to_txt, 'r') as file:
        for fileline in file:
            image_paths.append(path + '/' + fileline.strip(" \n"))
            image_names.append(fileline.strip(" \n"))
    return image_paths, image_names


if __name__ == '__main__':
    # Files with image names to be tested
    dataset = ("dataset/images/bar/bar001/", "dataset/images/market/market001/")

    # Loop through each video, outputting results in a xml file
    for video_sequence in dataset:
        # Get the name of this video sequence
        video_sequence_name = video_sequence.split('/')[-2]
        # Open the txt file with all images's names
        image_paths, image_names = get_all_image_names_as_list(video_sequence)
        # Frame counter
        total_frames = 0
        # Start the xml writer
        root = ET.Element("dataset", name=video_sequence_name)
        # Start the frames per second throughput estimator
        fps = FPS().start()

        for img_path, img_name in zip(image_paths, image_names):
            frame = cv2.imread(img_path)

            # If skip frames passed, run detector
            if total_frames % config.SKIP == 0:
                # detections should be a list/array of shape (num_faces, 4)
                detections = config.DETECTOR.detect(frame)
                if config.SKIP != 1:
                    config.TRACKER.create_trackers(detections, frame)
            else:
                detections = config.TRACKER.update_trackers(frame)

            frame_info = ET.SubElement(root, "frame", name="{}".format(img_name))
            for detection in detections:
                ET.SubElement(frame_info, "bbox", x_left="{}".format(detection[0]), y_top="{}".format(detection[1]),
                              x_right="{}".format(detection[2]), y_bottom="{}".format(detection[3]))
            # Update counters
            fps.update()
            total_frames += 1

            # If we want to see things happening, render video. This should be off during
            # actual benchmark
            if config.SHOW:
                cv2.imshow(video_sequence_name, draw_boxes_on_image(frame, detections))
                _ = cv2.waitKey(1) & 0xFF

        # Before closing the output file write FPS info
        fps.stop()
        fps_info = ET.SubElement(root, "fps_info", approx_fps='{}'.format(fps.fps()),
                                 elaps_time='{}'.format(fps.elapsed()))
        tree = ET.ElementTree(root)
        tree.write("output/"+config.TEST_NAME+video_sequence_name+".xml")
