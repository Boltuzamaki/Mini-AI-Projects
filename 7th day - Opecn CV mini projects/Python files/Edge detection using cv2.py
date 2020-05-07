
# coding: utf-8

# In[3]:


import cv2
import numpy as np

def Edge_detector(input_form = "video", path = "", operation = "canny", mode = "gray"):
    if input_form == "image":
        img = cv2.imread(path)
        if mode == "gray":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if operation == "laplacian":
            img = cv2.Laplacian(img, cv2.CV_64F, ksize  =3)
            img = np.uint8(np.absolute(img))
        if operation == "sobelx":    
            img = cv2.Sobel(img, cv2.CV_64F, 1 , 0)
        if operation  == "sobely":    
            img = cv2.Sobel(img, cv2.CV_64F, 0, 1)
        if operation  == "canny":    
            img = cv2.Canny(img, 100, 200)
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
                if operation == "laplacian":
                    img = cv2.Laplacian(img, cv2.CV_64F, ksize  =3)
                    img = np.uint8(np.absolute(img))
                if operation == "sobelx":    
                    img = cv2.Sobel(img, cv2.CV_64F, 1 , 0)
                if operation  == "sobely":    
                    img = cv2.Sobel(img, cv2.CV_64F, 0, 1)
                if operation  == "canny":    
                    img = cv2.Canny(img, 100, 200)
                cv2.imshow('frame', img)    
                if cv2.waitKey(1) & 0xff == ord('q'):
                    break
            else:
                break
    cap.release()
    cv2.destroyAllWindows() 


# In[4]:


Edge_detector()

