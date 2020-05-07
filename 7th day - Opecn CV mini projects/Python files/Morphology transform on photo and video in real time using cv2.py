
# coding: utf-8

# # Morphology transform on photo and video in real time

# In[ ]:


import cv2
import numpy as np
def Morphology_transform(input_form = "video", path = "", operation = "closing", mode = "gray"):
    if input_form == "image":
        img = cv2.imread(path)
        if mode == "gray":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(img, 155, 255, cv2.THRESH_BINARY_INV)
        kernal = np.ones((5, 5), np.uint8)
        if operation == "dilation":
            img =cv2.dilate(mask, kernal, iterations = 2)
        if operation == "erosion":    
            img = cv2.erode(mask, kernel, iterations = 1)
        if operation  == "opening":    
            img = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal)
        if operation == "closing":    
            img = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernal)
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
                _, mask = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
                kernal = np.ones((5, 5), np.uint8)
                if operation == "dilation":
                    img =cv2.dilate(mask, kernal, iterations = 2)
                if operation == "erosion":    
                    img = cv2.erode(mask, kernel, iterations = 1)
                if operation  == "opening":    
                    img = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal)
                if operation == "closing":    
                    img = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernal)  
                cv2.imshow('frame', img)    
                if cv2.waitKey(1) & 0xff == ord('q'):
                    break
            else:
                break
    cap.release()
    cv2.destroyAllWindows()     


# In[ ]:


Morphology_transform()   

