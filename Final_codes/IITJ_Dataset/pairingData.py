import cv2
import dlib
from matplotlib.pyplot import imshow
import numpy as np
import os
from tqdm import tqdm

def cropSquare(img, size, interpolation=cv2.INTER_AREA):
    h, w = img.shape[:2]
    min_size = np.amin([h,w])

    # Centralize and crop
    crop_img = img[int(h/2-min_size/2):int(h/2+min_size/2), int(w/2-min_size/2):int(w/2+min_size/2)]
    resized = cv2.resize(crop_img, (size, size), interpolation=interpolation)

    return resized

def cropRect(img, height, width, interpolation=cv2.INTER_AREA):
    h, w = img.shape[:2]

    # Centralize and crop
    crop_img = img[int(h/2-height/2):int(h/2+height/2), int(w/2-width/2):int(w/2+width/2)]
    resized = cv2.resize(crop_img, (width, height), interpolation=interpolation)

    return resized


def newextraction():
    path1 = 'IITJ-Data/NIR-video'
    path2 = 'IITJ-Data/RGB-video'
    files = os.listdir(path1)
    for i in tqdm(files):
        direc = i.split('.')[0]
        newPath = os.path.join('IITJ-Data/Paired/', direc)
        if not os.path.isdir(newPath):
            os.mkdir(newPath)
        newPath = os.path.join(newPath, 'NIR')
        if not os.path.isdir(newPath):
            os.mkdir(newPath)
        vid = cv2.VideoCapture(path1 + '/' + i)
        count = 0
        while(True):
            ret, frame = vid.read()
            # frame = cropSquare(frame, 300)  # For square cropping 
            # cv2.imwrite("IITJ-Data/Paired/" + direc + "/NIR/" + direc + "_%d.jpg" % count, frame)
            cv2.imwrite("IITJ-Data/Paired/" + direc + "/NIR/%d.jpg" % count, frame)
            count += 1 
            if(count == 10):
                break
    
    files = os.listdir(path2)
    for i in tqdm(files):
        direc = i.split('.')[0]
        newPath = os.path.join('IITJ-Data/Paired/', direc)
        if not os.path.isdir(newPath):
            os.mkdir(newPath)
        newPath = os.path.join(newPath, 'RGB')
        if not os.path.isdir(newPath):
            os.mkdir(newPath)
        vid = cv2.VideoCapture(path2 + '/' + i)
        count = 0
        while(True):
            ret, frame = vid.read()
            # frame = cropSquare(frame, 300)  # For square cropping
            frame = cropRect(frame, 600, 800) # For rectangle cropping 
            # cv2.imwrite("IITJ-Data/Paired/" + direc + "/RGB/" + direc + "_%d.jpg" % count, frame)
            cv2.imwrite("IITJ-Data/Paired/" + direc + "/RGB/%d.jpg" % count, frame)
            count += 1 
            if(count == 10):
                break


def main():
    newextraction()
    
if __name__ == "__main__":
    main()