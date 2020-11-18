import xml.etree.ElementTree as ET
from imutils.video import FPS
import config
import cv2
"""
This is a script for running various face detectors and trackers methods
It output the hypothesis as a xml file.
Programmer: Cassio B. Nascimento
"""


def get_all_image_names_as_list(common_path, video):
    path_to_txt = common_path + video + '/all_file.txt'
    image_names = []
    with open(path_to_txt, 'r') as file:
        for fileline in file:
            image_names.append(common_path+video+'/'+fileline.strip(" \n"))
    return image_names


if __name__ == '__main__':
    common_path = "/home/cassio/PycharmProjects/Face_Counting_Tracking/dataset/Chokepoint/"
    # Files with image names to be tested

    chokepoint_videos = ("P1E_S1/P1E_S1_C1", )
    # my_videos = ("")

    # Loop through each video, outputting results in a xml file
    for video in chokepoint_videos:
        # Get the name of this video sequence
        video_sequence_name = video.split('/')[1]
        # Open the txt file with all images's names
        images = get_all_image_names_as_list(common_path, video)
        # Frame counter
        total_frames = 0
        # Start the xml writer
        root = ET.Element("dataset", name=video_sequence_name)
        # Start the frames per second throughput estimator
        fps = FPS().start()

        for image in images:
            frame = cv2.imread(image)

            # If skip frames passed, run detector
            if total_frames % config.SKIP == 0:
                detections = config.DETECTOR.detect(frame)
            else:
                # RUN TRACKEr
                pass

            frame_info = ET.SubElement(root, "frame", number="{}".format(total_frames))
            for detection in detections:
                ET.SubElement(frame_info, "bbox", x_left="{}".format(), y_top="{}".format(),
                              x_right="{}".format(), y_bottom="{}".format())
            # Update counters
            fps.update()
            total_frames += 1

            # If we want to see things happening, render video. This should be off during
            # actual benchmark
            if config.SHOW:
                cv2.imshow(video_sequence_name, frame)
                cv2.waitKey(0)

        # Before closing the output file write FPS info
        fps.stop()
        fps_info = ET.SubElement(root, "fps_info", approx_fps='{}'.format(fps.fps()),
                                 elaps_time='{}'.format(fps.elapsed()))
        tree = ET.ElementTree(root)
        tree.write("output/"+video_sequence_name+".xml")
