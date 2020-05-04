import cv2
import pyautogui as pg
import time


sleep_time = 5
margin_x = 300
margin_y = 300
image_loc = "5.jpg"
    
def draw_image(image):
    im = cv2.imread(image)
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.drawContours( im , contours, -1, (0,255,0), 3)
    
    time.sleep(sleep_time)
    
    for contour in contours:
        pg.click()
        for i in range(0, contour.shape[0]):
           x_cor = contour[i][0][0] + margin_x
           y_cor = contour[i][0][1] + margin_y
           if i == 0:
               pg.moveTo(x_cor, y_cor)
           pg.dragTo(x_cor, y_cor )
   
       
     
draw_image(image_loc)        