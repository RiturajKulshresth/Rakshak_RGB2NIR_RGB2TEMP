from matplotlib.pyplot import imshow
import numpy as np
import cv2
from keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential
import os
from tqdm import tqdm

Size = 256

img_data = []

path1 = '/home/netrunner/Desktop/Raks/val_256/RGB'
files = os.listdir(path1)
for i in tqdm(files):
    img = cv2.imread(path1 + '/' + i, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (Size, Size))
    img_data.append(img_to_array(img))
# img = cv2.imread('/home/netrunner/Desktop/Raks/val_256/458_c.tiff',1)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = cv2.resize(img,(Size,Size))
# img_data.append(img_to_array(img))

img_array = np.reshape(img_data, (len(img_data), Size, Size, 3))
img_array = img_array.astype('float32')/255.

img_data2 = []

path2 ='/home/netrunner/Desktop/Raks/val_256/NIR'
files = os.listdir(path2)
for i in tqdm(files):
    img = cv2.imread(path2+'/'+i,1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (Size, Size))
    img_data2.append(img_to_array(img))
# img2 = cv2.imread('/home/netrunner/Desktop/Raks/val_256/458.tiff',1)
# img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
# img2 = cv2.resize (img2,(Size,Size))
# img_data2.append(img_to_array(img2))

img_array2 = np.reshape(img_data2, (len(img_data2), Size, Size, 3))
img_array2 = img_array2.astype('float32')/255.

img_data3 = []

img3 = cv2.imread('/home/netrunner/Desktop/Raks/val_256/458_c.tiff', 1)
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
img3 = cv2.resize(img3, (Size, Size))
img_data3.append(img_to_array(img3))

img_array3 = np.reshape(img_data3, (len(img_data3), Size, Size, 3))
img_array3 = img_array3.astype('float32')/255.


# import time
# start = time.time() 

model = Sequential()
model.add(Conv2D(256, (3, 3), activation = 'relu', padding = 'same', input_shape = (Size, Size, 3)))
model.add(MaxPooling2D((2,2), padding = 'same'))
# model.add(Conv2D(64, (3, 3), activation = 'relu', padding = 'same'))
# model.add(MaxPooling2D((2,2), padding = 'same'))
model.add(Conv2D(128, (3, 3), activation = 'relu', padding = 'same'))
model.add(MaxPooling2D((2,2), padding = 'same'))
model.add(Conv2D(128, (3, 3), activation = 'relu', padding = 'same'))
# model.add(MaxPooling2D((2, 2), padding = 'same'))
# model.add(Conv2D(8, (3, 3), activation = 'relu', padding = 'same'))

model.add(MaxPooling2D((2, 2), padding = 'same'))

# model.add(Conv2D(8, (3, 3), activation = 'relu', padding = 'same'))
# model.add(UpSampling2D((2,2)))
model.add(Conv2D(128, (3, 3), activation = 'relu', padding = 'same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation = 'relu', padding = 'same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(256, (3, 3), activation = 'relu', padding = 'same'))
model.add(UpSampling2D((2,2)))
# model.add(Conv2D(128, (3, 3), activation = 'relu', padding = 'same'))
# model.add(UpSampling2D((2,2)))
model.add(Conv2D(3, (3, 3), activation = 'relu', padding = 'same'))

model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])
model.summary()

model.fit(img_array, img_array2, epochs = 100, batch_size = 10, validation_split=0.1, shuffle = True, verbose = 1)

model.save('/home/netrunner/Desktop/Raks/val_256/model_proper.model') 

print("Output")
output = model.predict(img_array3)

imshow(output[0].reshape(Size, Size, 3))