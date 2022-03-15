#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 08:41:49 2021

@author: netrunner
"""
# filtering and high frequency
# import numpy as np
# import cv2

# #Define the window name
# windowName = "Bilateral Filter"

# #Read in the first test image
# img1 = cv2.imread('test2.png')
# imgorg=cv2.imshow("org",img1)
# #Ensure the image has loaded properly

# #Apply the bilateral filter
# filtered1 = cv2.bilateralFilter(img1,9,75,75)

# #Display the filtered image
# cv2.imshow(windowName,filtered1)
# #Close the window when the user presses any key
# cv2.waitKey(0)
# cv2.destroyWindow(windowName)
# imgorg=cv2.imshow("BF",filtered1)
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import matplotlib.image as img


def gauss(img,spatialKern, rangeKern):    
    gaussianSpatial = 1 / math.sqrt(2*math.pi* (spatialKern**2)) #gaussian function to calcualte the spacial kernel ( the first part 1/sigma * sqrt(2π))
    gaussianRange= 1 / math.sqrt(2*math.pi* (rangeKern**2)) #gaussian function to calcualte the range kernel
    matrix = np.exp(-np.arange(256) * np.arange(256) * gaussianRange)
    xx=-spatialKern + np.arange(2 * spatialKern + 1)
    yy=-spatialKern + np.arange(2 * spatialKern + 1)
    x, y = np.meshgrid(xx , yy )
    spatialGS = gaussianSpatial*np.exp(-(x **2 + y **2) /(2 * (gaussianSpatial**2) ) ) #calculate spatial kernel from the gaussian function. That is the gaussianSpatial variable multiplied with e to the power of (-x^2 + y^2 / 2*sigma^2) 
    return matrix,spatialGS

def padImage(img,spatialKern): #pad array with mirror reflections of itself.
    img=np.pad(img, ((spatialKern, spatialKern), (spatialKern, spatialKern), (0, 0)), 'symmetric')
    return img
    
def jointBilateralFilter(img, img1,spatialKern, rangeKern):
    h, w, ch = img.shape #get the height,width and channel of the image with no flash
    orgImg = padImage(img,spatialKern) #pad image with no flash
    secondImg = padImage(img1,spatialKern)   #pad image with flash  
    matrix,spatialGS=gauss(img,spatialKern, rangeKern) #apply gaussian function

    outputImg = np.zeros((h,w,ch), np.uint8) #create a matrix the size of the image
    summ=1
    for x in range(spatialKern, spatialKern + h):
        for y in range(spatialKern, spatialKern + w):
            for i in range (0,ch): #iterate through the image's height, width and channel
                #apply the equation that is mentioned in the pdf file
                neighbourhood=secondImg[x-spatialKern : x+spatialKern+1 , y-spatialKern : y+spatialKern+1, i] #get neighbourhood of pixels
                central=secondImg[x, y, i] #get central pixel
                res = matrix[ abs(neighbourhood - central) ]  # subtract them                   
                summ=summ*res*spatialGS #multiply them with the spatial kernel
                norm = np.sum(res) #normalization term
                outputImg[x-spatialKern, y-spatialKern, i]= np.sum(res*orgImg[x-spatialKern : x+spatialKern+1, y-spatialKern : y+spatialKern+1, i]) / norm # apply full equation of JBF(img,img1)
    return outputImg

original_rgb=cv2.imread("/home/netrunner/Desktop/Raks/val_256/458_c_rescalexddd.tiff")
original_nir=cv2.imread("/home/netrunner/Desktop/Raks/val_256/458_c_rescalexd.tiff")
conv_nir=cv2.imread("/home/netrunner/Downloads/128_128_128_128_64_64_64.png")
orig_rgb_gray_schan=cv2.cvtColor(original_rgb, cv2.COLOR_BGR2GRAY)
orig_rgb_gray=np.stack((orig_rgb_gray_schan,)*3,axis=-1)

spatialKern = 30 #27
rangeKern =20  #10
# img = cv2.imread('test3a.jpg')
# img1=cv2.imread('test3b.jpg') #read both images, flash and no flash image
# filteredimg = jointBilateralFilter(img, img1, spatialKern, rangeKern)
# filteredimg = jointBilateralFilter(original_nir, conv_nir, spatialKern, rangeKern)
# filteredimg2= jointBilateralFilter(conv_nir, original_nir, spatialKern, rangeKern)
plt.imshow(original_nir)
plt.title("original_nir")
plt.show()
plt.imshow(conv_nir) #show original no flash image
plt.title("conv_nir")
plt.show()

# plt.imshow(filteredimg) #show image after joint bilateral filter is applied
# plt.title("filteredimg")
# plt.show()
# plt.imshow(filteredimg2)
# plt.title("filteredimg2")
# plt.show()

# filteredimg_rgb = jointBilateralFilter(original_rgb, conv_nir, spatialKern, rangeKern)
# filteredimg_rgb2 = jointBilateralFilter(conv_nir, original_rgb, spatialKern, rangeKern)
# plt.imshow(filteredimg_rgb) #show image after joint bilateral filter is applied
# plt.title("filteredimg_rgb")
# plt.show()
# plt.imshow(filteredimg_rgb2)
# plt.title("filteredimg_rgb2")
# plt.show()

filteredimg_gray = jointBilateralFilter(orig_rgb_gray, conv_nir, spatialKern, rangeKern)
# filteredimg_gray2 = jointBilateralFilter(conv_nir, orig_rgb_gray, spatialKern, rangeKern)

plt.imshow(filteredimg_gray) #show image after joint bilateral filter is applied
plt.title("filteredimg_gray")
plt.show()
cv2.imwrite("/home/netrunner/Desktop/Raks/val_256/filtered.png",filteredimg_gray)
# plt.imshow(filteredimg_gray2)
# plt.title("filteredimg_gray2")
# plt.show()

##########################
# change contrast
# 
#-----Converting image to LAB Color model----------------------------------- 
lab= cv2.cvtColor(filteredimg_gray, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl = clahe.apply(l)
limg = cv2.merge((cl,a,b))
final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
plt.imshow(final) #show image after joint bilateral filter is applied
cv2.imwrite("/home/netrunner/Desktop/Raks/val_256/final.png", final)
plt.title("final")
plt.show()


