
# coding: utf-8

# # Thresholding on photo and video in real time

# In[ ]:


import cv2
import numpy as np


def Thresholding_func(input_form = "video", path = "", operation = "adaptive_gaussian", mode = "gray"):
    if input_form == "image":
        img = cv2.imread(path)
        if mode == "gray":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if operation == "normal":
            _, img  = cv2.threshold(img, 127 , 255, cv2.THRESH_BINARY)
        if operation == "adaptive_mean":    
            img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        if operation  == "adaptive_gaussian":    
            img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
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
                if mode == "gray":
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if operation == "normal":
                    _, img  = cv2.threshold(img, 100 , 255, cv2.THRESH_BINARY)
                if operation == "adaptive_mean":    
                    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
                if operation  == "adaptive_gaussian":   
                    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                cv2.imshow('frame', img)    
                if cv2.waitKey(1) & 0xff == ord('q'):
                    break
            else:
                break
    cap.release()
    cv2.destroyAllWindows()  
    
Thresholding_func()    

