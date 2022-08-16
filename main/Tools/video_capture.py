import cv2
import numpy as np


camera = cv2.VideoCapture(0)
camera.set(3,640)
camera.set(4,480)

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640,480))

while(camera.isOpened()):
    ret, frame = camera.read()
    out.write(frame)
    cv2.imshow('frame', frame)
    c = cv2.waitKey(1)
    if c & 0xFF == ord('q'):
        break

camera.release()
out.release()
cv2.destroyAllWindows()