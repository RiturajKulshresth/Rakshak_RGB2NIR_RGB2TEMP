from spectral import imshow, view_cube
import spectral.io.envi as envi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2
from spectral import open_image

# Size = 256

# img3 = cv2.imread('/home/netrunner/Desktop/Raks/val_256/458_c.tiff', 1)
# img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
# img3 = cv2.resize(img3, (Size, Size))
# cv2.imwrite('/home/netrunner/Desktop/Raks/val_256/458_c_rescaled.tiff')
# # img_data3.append(img_to_array(img3))


# f = open("/home/netrunner/Documents/hyperspectral/4cam_0411-1640-1.raw","r")
# hyper1=envi.open('/home/netrunner/Documents/hyperspectral/4cam_0411-1640-1.hdr','/home/netrunner/Documents/hyperspectral/4cam_0411-1640-1.raw')
# hyper1_nparr=np.array(hyper1.load())
# print(np.amax(hyper1_nparr))
# print(hyper1.shape)
# hyper2=np.array(hyper1_nparr[:,:,253:518])
img1=open_image('/home/netrunner/Documents/hyperspectral/4cam_0411-1640-1.hdr')
print(img1)
img1.shape
imshow(img1[:,:,205])
imshow(img1[:,:,132])
imshow(img1[:,:,58])
view=imshow(img1,(205,132,58))

view=imshow(img1,(332,332,332))
view=imshow(img1,(340,340,340))
view=imshow(img1,(348,348,348))
view=imshow(img1,(356,356,356))
view=imshow(img1,(364,364,364))
view=imshow(img1,(372,372,372))
view=imshow(img1,(380,380,380))
view=imshow(img1,(388,388,388))
view=imshow(img1,(396,396,396))
view=imshow(img1,(404,404,404))
view=imshow(img1,(412,412,412))
view=imshow(img1,(420,420,420))
view=imshow(img1,(428,428,428))
view=imshow(img1,(436,436,436))
view=imshow(img1,(444,444,444))
view=imshow(img1,(452,452,452))
view=imshow(img1,(460,460,460))
view=imshow(img1,(468,468,468))
view=imshow(img1,(476,476,476))
view=imshow(img1,(484,484,484))
view=imshow(img1,(492,492,492))
view=imshow(img1,(500,500,500))
view=imshow(img1,(508,508,508))
view=imshow(img1,(516,516,516))


# hyper_x=np.array(hyper2[:,:,1:4])
# hyper_x=hyper_x.reshape(1304, 1392, 3)
# cv2.imwrite('/home/netrunner/Documents/hyperspectral/new.png',hyper_x)
# imshow(hyper_x)