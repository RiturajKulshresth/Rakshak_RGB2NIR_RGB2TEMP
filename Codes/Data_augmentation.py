#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 07:30:45 2021

@author: netrunner
"""


# example of brighting image augmentation
from numpy import expand_dims
from keras.preprocessing.image import load_img, save_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
# import matplotlib.pyplot as plt
import cv2
import os
from matplotlib import pyplot

# import random
# load the image
# prh=r"/home/netrunner/Music/img_folder2"

# for folder in os.listdir(prh):
#     path2=os.path.join(prh,folder)
#     print(path2)
#     for i in os.listdir(path2):
#         path3=os.path.join(path2, i)
#         img = load_img(path3)
#         data = img_to_array(img)
#         samples = expand_dims(data, 0)
#         datagen = ImageDataGenerator(brightness_range=[0.2,1.0])
#         it = datagen.flow(samples, batch_size=1)
#         for k in range(10):
#             # pyplot.subplot(330 + 1 + k)
#             batch = it.next()
#             image = batch[0].astype('uint8')
            
#             pyplot.imshow(image)
#             pyplot.show()
#             # print(type(image))
#             # image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#             save_img("/home/netrunner/Music/aug/"+folder+"/"+i[:len(i)-4]+"_aug_"+str(k)+".jpg",image)
        
t_c=r"/home/netrunner/Music/img_folder2/98"           
t=r"/home/netrunner/Music/img_folder2/98/2.jpg"
img=load_img(t)
data = img_to_array(img)
samples = expand_dims(data, 0)
datagen = ImageDataGenerator(brightness_range=[0.2,1.0])
it = datagen.flow(samples, batch_size=4, save_to_dir=t_c,save_prefix="aug", save_format="jpg")

batch=it.next()
imag1=batch[0].astype('uint8')
print("vcat")
print(imag1)
save_img("/home/netrunner/Music/img_folder2/98/2_aug_1.jpg",imag1)
pyplot.imshow(imag1)
pyplot.show()

batch=it.next()
imag2=batch[0].astype('uint8')
print("vcat")
print(imag2)
save_img("/home/netrunner/Music/img_folder2/98/2_aug_2.jpg",imag2)
pyplot.imshow(imag2)
pyplot.show()

batch=it.next()
imag3=batch[0].astype('uint8')
print("vcat")
print(imag3)
save_img("/home/netrunner/Music/img_folder2/98/2_aug_3.jpg",imag3)
pyplot.imshow(imag3)
pyplot.show()

for y in range(3):
    batch=it.next()
    imag=batch[0].astype('uint8')
    # print ("image: ", y, imag)
    pyplot.imshow(imag)
    # imag= cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("/home/netrunner/Music/img_folder2/98/2"+str(y)+".jpg",imag)
    pyplot.show()


# import matplotlib.pyplot as plt

datagen = ImageDataGenerator(brightness_range=[0.4,1.5])

# iterator
aug_iter = datagen.flow(img, batch_size=1)

# generate samples and plot
# fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,15))

# generate batch of images
for i in range(3):

	# convert to unsigned integers
	image = next(aug_iter)[0].astype('uint8')
 
	# plot image
 	# ax[i].imshow(image)
 	# ax[i].axis('off')
# from PIL import imread
from scipy import ndimage
datagen = ImageDataGenerator(brightness_range=[0.4,1.5]) 
img_path2="/home/netrunner/Music/img_folder2/98/2.jpg"
img=load_img(img_path2)
data = img_to_array(img)
image = np.expand_dims(data, 0)
save_here="/home/netrunner/Music/aug"
datagen.fit(image)
for x, val in zip(datagen.flow(image, save_to_dir=save_here,save_prefix="aug",save_format='png'),range(10)):
    pass