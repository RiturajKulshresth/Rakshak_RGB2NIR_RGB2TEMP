#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 07:59:30 2021

@author: netrunner
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
path="/home/netrunner/Downloads/frames"
image = cv2.imread("/home/netrunner/Downloads/frames/0.jpg")
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

cropped = image[y:y+h, x:x+w]
plt.imshow(cropped)
plt.show()
print(x,y,w,h)






