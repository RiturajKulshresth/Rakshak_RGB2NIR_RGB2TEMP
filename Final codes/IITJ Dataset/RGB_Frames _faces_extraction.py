#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 00:11:44 2021

@author: netrunner
"""
# RGB frames extraction

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

path1="/home/netrunner/Downloads/RGB_vid"
path2="/home/netrunner/Downloads/RGB_frames"
path3="/home/netrunner/Downloads/RGB_faces"
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
        
        
        
# path1="/home/netrunner/Downloads/NIR_vid"
path2="/home/netrunner/Downloads/NIR_frames"
path3="/home/netrunner/Downloads/NIR_faces"


files = os.listdir(path2)

for i in files:
    path_temp=os.path.join(path2, i)
    count=0
    # os.mkdir(path3 + "/" + i)
    for ii in os.listdir(path_temp):
        # face_temp=os.path.join(path_temp, ii)
        img = cv2.imread(path_temp+"/"+ii)
        # face_save=os.path.join(path3 + "/" + i + "/" + ii)
        
        
        # img = cv2.imread(face_temp)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            faces = img[y:y + h, x:x + w]
            cv2.imwrite(path3 + "/"+i+"/"+"%d.jpg" % count, faces)
        
        count += 1
        
        
path = '/home/netrunner/Downloads/frames'
files = os.listdir(path)
count = 0

for i in tqdm(files):
    img = cv2.imread(path + '/' + i)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        faces = img[y:y + h, x:x + w]
        cv2.imwrite('/home/netrunner/Downloads/faces/%d.jpg' % count, faces)
    
    count += 1
        
        
        
        
        # hsv_im=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # mask = cv2.inRange(hsv_im, (36, 25, 25), (70, 255,255))
        # imask = mask>0
        # green = np.zeros_like(image, np.uint8)
        # green[imask] = image[imask]
        
        # # image = image[150:250, 100:200]
        # green=cv2.cvtColor(green, cv2.COLOR_HSV2RGB)
        # # green = green[150:250, 100:200]
        # height, width, channels = green.shape
        # center = (width / 2, height / 2)
        
        # gray = cv2.cvtColor(green, cv2.COLOR_BGR2GRAY)
        # gray = cv2.threshold(gray, 50, 240, cv2.THRESH_BINARY)[1]
        
        # contours,hierarchy = cv2.findContours(gray.copy(), 1, 2)
        # for cnt in contours:
        #     x_t,y_t,w_t,h_t = cv2.boundingRect(cnt)
         
        #     if w_t<20 and h_t<20:
        #         continue
        #     else:
        #         x=x_t
        #         y=y_t
        #         h=h_t
        #         w=w_t
        # try:
        #     cropped = image[y:y+h, x:x+w]
        #     cv2.imwrite(path3 + "/" + i + "/" + ii, cropped )
        # except:
        #     None
        















