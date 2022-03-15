from matplotlib.pyplot import imshow
import numpy as np
import cv2
from keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential
import os
from tqdm import tqdm
import spectral.io.envi as envi

Size = 256

def prepareRGBData():
    img_data = []
    
    path1 = 'icvl/rgb'
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
    
    return img_array

def prepareHyperData():
    img_data = []
    
    path2 = 'icvl/hyper'
    # files = os.listdir(path2)
    # for i in tqdm(files):
    #     img = cv2.imread(path2 + '/' + i, 1)
    
    img = envi.open(path2 + '/001.hdr', path2 + '/001.raw')
    sliced = np.array(img[:,:,140])
    z = max(sliced[1150, 800])
    # slice_clip = min(sliced, z)/z
    # img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    imshow(sliced, cmap="gray")
    print(img)
    

def testimage():
    img_data3 = []
    
    img3 = cv2.imread('val_256/457.tiff', 1)
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    img3 = cv2.resize(img3, (Size, Size))
    img_data3.append(img_to_array(img3))
    
    img_array3 = np.reshape(img_data3, (len(img_data3), Size, Size, 3))
    img_array3 = img_array3.astype('float32')/255.
    
    return img_array3

# RGB_array = prepareRGBData()
# print(RGB_array)

def model(arr1, arr2):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(Size, Size, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    # model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    # model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
    # model.add(Conv2D(256, (3,3), activation='relu', padding='same', strides=2))
    # model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
    # model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
    # model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
    
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    # model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    # model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))
    # model.add(UpSampling2D((2, 2)))
    
    model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])
    model.summary()
    
    model.fit(arr1, arr2, epochs = 100, batch_size = 20, validation_split=0.1, verbose = 1)

    model.save('val_256/model1.model') 
    
    return model
    
def output(m1, arr3):
    print("Output")
    output = m1.predict(arr3)

    imshow(output[0].reshape(Size, Size, 3))

prepareHyperData()
