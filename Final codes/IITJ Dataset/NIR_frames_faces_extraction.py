#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 01:08:13 2021

@author: netrunner
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
path1="/home/netrunner/Downloads/NIR_vid"
path2="/home/netrunner/Downloads/NIR_frames"
path3="/home/netrunner/Downloads/NIR_faces"
vid_files=os.listdir(path1)
for i in vid_files:
    vod=cv2.VideoCapture(path1 + "/" + i)
    os.mkdir(path2 + "/" + i[:len(i)-4])
    os.mkdir(path3 + "/" + i[:len(i)-4])
    count = 0
    success=True
    while (True):
        success, image = vod.read()
        print (success)
        if success:
            cv2.imwrite(path2+"/"+ i[:len(i)-4]+"/%d.jpg" % count, image)
            count = count + 1
        else:
            print("cat")
            break
        if (count ==9):
            print ("error")
            break 


path1="/home/netrunner/Downloads/NIR_vid"
path2="/home/netrunner/Downloads/NIR_frames"
path3="/home/netrunner/Downloads/NIR_faces"


files = os.listdir(path2)

for i in files:
    path_temp=os.path.join(path2, i)
    # os.mkdir(path3 + "/" + i)
    for ii in os.listdir(path_temp):
        face_temp=os.path.join(path_temp, ii)
        image = cv2.imread(face_temp)
        
        hsv_im=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_im, (36, 25, 25), (70, 255,255))
        imask = mask>0
        green = np.zeros_like(image, np.uint8)
        green[imask] = image[imask]
        
        # image = image[150:250, 100:200]
        green=cv2.cvtColor(green, cv2.COLOR_HSV2RGB)
        # green = green[150:250, 100:200]
        height, width, channels = green.shape
        center = (width / 2, height / 2)
        
        gray = cv2.cvtColor(green, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 50, 240, cv2.THRESH_BINARY)[1]
        
        contours,hierarchy = cv2.findContours(gray.copy(), 1, 2)
        for cnt in contours:
            x_t,y_t,w_t,h_t = cv2.boundingRect(cnt)
         
            if w_t<20 and h_t<20:
                continue
            else:
                x=x_t
                y=y_t
                h=h_t
                w=w_t
        try:
            cropped = image[y:y+h, x:x+w]
            cv2.imwrite(path3 + "/" + i + "/" + ii, cropped )
        except:
            None
        
        # plt.imshow(cropped)
        # plt.show()
        # print(x,y,w,h)


# import numpy as np
# import cv2
# import matplotlib.pyplot as plt

# image=cv2.imread('/home/netrunner/Downloads/frames/frame0.jpg')
# orig_image=image.copy()
# cv2.imshow('original image',orig_image)
# cv2.waitKey(0)
# gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# ret, thresh=cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
# _, contours, hierarchy=cv2.findContours(thresh.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
# for c in contours:
#     x,y,w,h=cv2.boundingRect(c)
#     cv2.rectangle(orig_image,(x,y),(x+w,y+h),(0,0,255),2)
#     cv2.imshow('Bounding rect',orig_image)
# cv2.waitKey(0)
# accuracy=0.03*cv2.arcLength(c,True)
# approx=cv2.approxPolyDP(c,accuracy,True)
# cv2.drawContours(image,[approx],0,(0,255,0),2)
# cv2.imshow('Approx polyDP', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# import cv2
  
# # Function to extract frames
# def FrameCapture(path):
      
#     # Path to video file
#     vidObj = cv2.VideoCapture(path)
  
#     # Used as counter variable
#     count = 0
  
#     # checks whether frames were extracted
#     success = 1
  
#     while success:
  
#         # vidObj object calls read
#         # function extract frames
#         success, image = vidObj.read()
  
#         # Saves the frames with frame-count
#         try:
#             cv2.imwrite("/home/netrunner/Downloads/frames/frame%d.jpg" % count, image)
#         except:
#             print("cat")
#             None
#         count += 1
  
# # Driver Code
# if __name__ == '__main__':
  
#     # Calling the function
#     FrameCapture("/home/netrunner/Downloads/97.mp4")