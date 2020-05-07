
# coding: utf-8

# # Taking real time video using webcam

# In[1]:


import cv2
import time

def Video(Height = 600, Width = 800, mode = "GRAY"):
    cap = cv2.VideoCapture(0)
    cap.set(3,Width)                       # Set the width
    cap.set(4,Height)                       # Set the height
    time.sleep(1)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            if mode == "GRAY":
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('frame', frame)    
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
Video()

