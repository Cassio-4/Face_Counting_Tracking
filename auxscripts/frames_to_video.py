import cv2

video_sequence_name = "bar001"
path = "/home/cassio/PycharmProjects/Face_Counting_Tracking/dataset/images/bar/bar001/bar001-%05d.jpg"
vc = cv2.VideoCapture(path)
W = vc.get(cv2.CAP_PROP_FRAME_WIDTH)
H = vc.get(cv2.CAP_PROP_FRAME_HEIGHT)
size = (int(W), int(H))

out = cv2.VideoWriter('{}.mp4'.format(video_sequence_name), cv2.VideoWriter_fourcc(*'DIVX'), 20, size)

while True:
    frame = vc.read()
    frame = frame[1]
    if frame is None:
        break
    out.write(frame)

out.release()
vc.release()
