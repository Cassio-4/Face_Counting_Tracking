from detectors.tensorflow_detectors import FaceboxesTensorflow
from imutils.video import FPS
import dlib
import cv2
import os


def draw_boxes_on_image(image, boxes, detector):
    """
    Draws bounding boxes and Ids on image
    :param image: the image to draw on (using opencv)
    :return: the drawn on frame
    """
    image_copy = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
    if detector == "faceboxes":
        color = (0, 0, 255)
        for box in boxes:
            cv2.rectangle(image_copy, (box[0], box[1]), (box[2], box[3]), color, 2)
    else:
        color = (0, 255, 0)
        for box in boxes:
            cv2.rectangle(image_copy, (box.left(), box.top()), (box.right(), box.bottom()), color, 2)
    return image_copy


def get_all_image_paths(path):
    all_image_paths = []
    for r, d, f in os.walk(path):
        for file in f:
            if file.endswith(".jpg"):
                all_image_paths.append(os.path.join(r, file))
    return all_image_paths


def run_on_images(all_images):
    fb_det = FaceboxesTensorflow("../detectors/models/faceboxesTensorflow.pb", score_threshold=0.6)
    dlib_det = dlib.get_frontal_face_detector()

    for img in all_images:
        frame = cv2.imread(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        fb_detections, _ = fb_det.detect(frame)
        dlib_detections = dlib_det(frame, 1)

        img_name = img.split("/")[-1]
        img_name = img_name.split(".")[0]

        cv2.imwrite("../output/{}-faceBoxes{}.jpg".format(img_name, len(fb_detections)),
                    draw_boxes_on_image(frame, fb_detections, "faceboxes"))
        cv2.imwrite("../output/{}-dlibFrontal{}.jpg".format(img_name, len(dlib_detections)),
                    draw_boxes_on_image(frame, dlib_detections, "dlib"))


if __name__ == '__main__':
    path = "/home/cassio/Downloads/teste_imagens/"
    all_images = get_all_image_paths(path)
    test = "images"
    algo = "faceboxes"

    if test == "images":
        run_on_images(all_images)
    else:

        if algo == "faceboxes":
            detector = FaceboxesTensorflow("../detectors/models/faceboxesTensorflow.pb", score_threshold=0.6)
        else:
            detector = dlib.get_frontal_face_detector()

        vc = cv2.VideoCapture(path)
        fps = FPS().start()

        while True:
            frame = vc.read()
            frame = frame[1]
            # Se estamos processando um video e nao recebemos um frame, chegamos ao fim
            if frame is None:
                break
            # Convert frame to RGB format required by dlib and Faceboxes
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if algo == "faceboxes":
                detections = detector.detect(rgb)
            else:
                detections = detector(rgb, 1)
            fps.update()

        # Close video capture
        vc.release()
        # whatever
        fps.stop()

        print("approx_fps {}".format(fps.fps()))
        print("elaps_time={}".format(fps.elapsed()))
