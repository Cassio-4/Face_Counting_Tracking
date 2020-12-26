import cv2

cv = cv2.VideoCapture("sim01_collider.mp4")
i = 0
while cv.isOpened():
    ret, frame = cv.read()
    if ret:
        cv2.imwrite("sim01_collider-frame{}.jpg".format(i), frame)
        i += 1
    else:
        break
