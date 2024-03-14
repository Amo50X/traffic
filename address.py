
# import numpy as np
import cv2
import queue
cap = cv2.VideoCapture()
cap.open("http://192.168.0.154:8080/video")

while(True):
     # Capture frame-by-frame
    
    ret, frame = cap.read()

    if not ret:
        break
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
# from modules import RunningCamera

# queue_cam_dict={}
# cap_Path = "static/clips/captured"
# queue_cam_dict.update({f'{cap}': queue.Queue() })
# RunningCamera(cap=cap ,cap_Path=cap_Path, queue_cam_dict=queue_cam_dict)