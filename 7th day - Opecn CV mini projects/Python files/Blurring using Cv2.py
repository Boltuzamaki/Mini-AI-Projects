
# coding: utf-8

# # Different types of blurring using cv2

# In[1]:


import cv2
import numpy as np
def Blur_fun(kernel = None,input_form = "video", path = "", operation = "meadianBlur", mode = "rbg"):
    if input_form == "image":
        img = cv2.imread(path)
        if mode == "gray":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if operation == "custom":
            img = cv2.filter2D(img, -1, kernel)
        if operation == "blur":    
            img = cv2.blur(img, (5,5))
        if operation  == "GaussianBlur":    
            img = cv2.GaussianBlur(img, (5,5), 0)
        if operation  == "meadianBlur":    
            img = cv2.meadianBlur(img, 5)
        if operation  == "bilateralFilter":    
            img = cv2.bilateralFilter(img, 9, 75, 75)   
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
                if operation == "custom":
                    img = cv2.filter2D(img, -1, kernel)
                if operation == "blur":    
                    img = cv2.blur(img, (5,5))
                if operation  == "GaussianBlur":    
                    img = cv2.GaussianBlur(img, (5,5), 0)
                if operation  == "meadianBlur":    
                    img = cv2.medianBlur(img, 5)
                if operation  == "bilateralFilter":    
                    img = cv2.bilateralFilter(img, 9, 75, 75) 
                cv2.imshow('frame', img)    
                if cv2.waitKey(1) & 0xff == ord('q'):
                    break
            else:
                break
    cap.release()
    cv2.destroyAllWindows()  


# In[2]:


kernel = np.ones((5,5), np.float32)/25
Blur_fun(kernel)

