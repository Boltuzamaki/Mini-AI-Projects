
# coding: utf-8

# In[ ]:


import cv2
import numpy as np


def Countour_detector(input_form = "video", path = ""):
    if input_form == "image":
        img = cv2.imread(path)
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(imgray, 127, 255, 0)
        contours, hierarchy = cv2.findContour(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
        cv2.imshow('Image GRAY', imgray)  
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()    
        
    if input_form == "video":
        if path == "":
            cap = cv2.VideoCapture(0)
        if path != "":
            cap = cv2.VideoCapture(path)
        while(cap.isOpened()):
            ret, img = cap.read()
            if ret == True:
                imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(imgray, 127, 255, 0)
                _,contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
                cv2.imshow('Image GRAY', imgray)  
                cv2.imshow("Image", img)
                if cv2.waitKey(1) & 0xff == ord('q'):
                    break
            else:
                break
    cap.release()
    cv2.destroyAllWindows() 


# In[ ]:


Countour_detector()

