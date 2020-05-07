
# coding: utf-8

# In[1]:


import cv2
import time
def Background_Subtractor(mode = "GMG", path = ""):
    if path == "":
        cap = cv2.VideoCapture(0)
    if path != "":
        cap = cv2.VideoCapture(path)
    time.sleep(1)
    if mode == "MOG":
        fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=2, nmixtures=10, backgroundRatio=0.0001)
        while True:
            ret, frame = cap.read()
            fgmask = fgbg.apply(frame)
            cv2.imshow('Frame', frame)
            cv2.imshow('FG Mask Frame', fgmask)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    if mode == "MOG2":
        fgbg = cv2.createBackgroundSubtractorMOG2(history = 50,varThreshold = 16,detectShadows = False)
        while True:
            ret, frame = cap.read()
            fgmask = fgbg.apply(frame)
            cv2.imshow('Frame', frame)
            cv2.imshow('FG Mask Frame', fgmask)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    if mode == "KNN":
        fgbg = cv2.createBackgroundSubtractorKNN(history = 50,dist2Threshold = 400.0,detectShadows = False)
        while True:
            ret, frame = cap.read()
            fgmask = fgbg.apply(frame)
            cv2.imshow('Frame', frame)
            cv2.imshow('FG Mask Frame', fgmask)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    if mode == "CNT":
        fgbg = cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability = 15,useHistory = False,maxPixelStability = 15 *60,isParallel = True ) 
        while True:
            ret, frame = cap.read()
            fgmask = fgbg.apply(frame)
            cv2.imshow('Frame', frame)
            cv2.imshow('FG Mask Frame', fgmask)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    if mode == "GMG":
        cap = cv2.VideoCapture(0)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        fgbd =cv2.bgsegm.createBackgroundSubtractorGMG()
    while True:
        ret, frame = cap.read()
        if frame is None:
            break
        fgmask = fgbd.apply(frame)  
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        cv2.imshow('Frame', frame)
        cv2.imshow('FG MASK Frame', fgmask)
        if cv2.waitKey(1) & 0xff == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()


# In[2]:


Background_Subtractor()

