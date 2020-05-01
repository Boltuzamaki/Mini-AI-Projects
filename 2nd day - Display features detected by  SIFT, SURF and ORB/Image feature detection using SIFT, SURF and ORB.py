
# coding: utf-8

# # Display features detected by  SIFT, SURF and ORB

# In[11]:


import cv2
import numpy as np
import cv2
# Surf and Sift are under patient but orb is opensource

def Surf_Sift_Orb(img_path, algo=None):
    img = cv2.imread(img_path , cv2.IMREAD_GRAYSCALE)
    if algo == "SIFT":
        sift = cv2.xfeatures2d.SIFT_create()
        keypoints , descriptors = sift.detectAndCompute(img, None)
        
    if algo == "SURF":    
        surf = cv2.xfeatures2d.SURF_create()
        keypoints , descriptors = surf.detectAndCompute(img, None)

    if algo == "ORB": 
        orb = cv2.ORB_create()
        keypoints , descriptors = orb.detectAndCompute(img, None)
    
    img = cv2.drawKeypoints(img, keypoints, None)
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# In[12]:


Surf_Sift_Orb('10.jpg', "SIFT")                     # fed path and algo here


# # Brute force feature matching b/w two image using SURF/ SIFT/ORB

# In[32]:


import cv2
import numpy as np

def Brute_Matching(base_img_path, test_img_path, Number_of_matches, algo = None):
    img1 = cv2.imread(base_img_path , cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(test_img_path , cv2.IMREAD_GRAYSCALE)
    
    # ORB Detector
    if algo == "ORB":
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        
        # Brute force search
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key = lambda x:x.distance)
        for m in matches:
            m.distance
        matching_result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:Number_of_matches], None) 

    # SIFT Detector
    if algo == "SIFT":
        sift= cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        
        # Brute force search
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2,k=2)
        good = []
        good_without_list = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])
                good_without_list.append(m)
        matching_result = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good[0:Number_of_matches], None, flags=2)       

    # SURF Detector
    if algo == "SURF":
        surf = cv2.xfeatures2d.SURF_create()
        kp1, des1 = surf.detectAndCompute(img1, None)
        kp2, des2 = surf.detectAndCompute(img2, None)
        
        # Brute force search
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2,k=2)
        good = []
        good_without_list = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])
                good_without_list.append(m)
        matching_result = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good[0:Number_of_matches], None, flags=2)       
    
    matching_result = cv2.resize(matching_result, (1500, 1000),interpolation = cv2.INTER_NEAREST) 
    cv2.imshow("Img1", img1)
    cv2.imshow("Img2", img2)
    cv2.imshow("Matches Keypoints", matching_result)
   
    cv2.waitKey()
    cv2.destroyAllWindows()


# In[33]:


Brute_Matching("10.jpg", "11.jpg",20, algo = "SIFT")   # Show the feature common in two image using above techniques


# # Real time object detection using SIFT features

# In[35]:


import cv2
import numpy as np

def Real_time(image,video_sourc = 0):                # video_sourc can be 0 (web cam) or any file path location
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)  # queryiamge
    cap = cv2.VideoCapture(video_sourc)
    # Features
    sift = cv2.xfeatures2d.SIFT_create()
    kp_image, desc_image = sift.detectAndCompute(img, None)
    # Feature matching
    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    while cap.isOpened():
        _, frame = cap.read()
        grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # trainimage
        kp_grayframe, desc_grayframe = sift.detectAndCompute(grayframe, None)
        matches = flann.knnMatch(desc_image, desc_grayframe, k=2)
        good_points = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good_points.append(m)
        # img3 = cv2.drawMatches(img, kp_image, grayframe, kp_grayframe, good_points, grayframe)
        # Homography
        if len(good_points) > 10:
            query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
            train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
            matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
            matches_mask = mask.ravel().tolist()
            # Perspective transform
            h, w = img.shape
            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, matrix)
            homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)
            cv2.imshow("Homography", homography)
        else:
            cv2.imshow("Homography", grayframe)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


# In[ ]:


Real_time('10.JPG')                                         # In jupyter this mat sometime throw error so use python ide or spyder instead

