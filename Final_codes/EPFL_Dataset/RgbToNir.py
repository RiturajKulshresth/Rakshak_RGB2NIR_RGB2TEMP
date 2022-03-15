from matplotlib.pyplot import imshow, show
# from scipy.signal import savgol_filter
import numpy as np
import cv2
import tensorflow as tf
from keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential
import os
from tqdm import tqdm

Size = 256

def cropSquare(img, size, interpolation=cv2.INTER_AREA):
    h, w = img.shape[:2]
    min_size = np.amin([h,w])

    # Centralize and crop
    crop_img = img[int(h/2-min_size/2):int(h/2+min_size/2), int(w/2-min_size/2):int(w/2+min_size/2)]
    resized = cv2.resize(crop_img, (size, size), interpolation=interpolation)

    return resized

def prepareRGBData():
    img_data = []
    
    path1 = 'val_256/RGB'
    files = os.listdir(path1)
    for i in tqdm(files):
        img = cv2.imread(path1 + '/' + i, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (Size, Size))
        # img = cropSquare(img, Size)
        # img = savgol_filter(img, window_length=255, polyorder=3, axis=0)
        # img = savgol_filter(img, window_length=255, polyorder=3, axis=1)
        img_data.append(img_to_array(img))
        for j in range(3):
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            # for k in range(2):
            #     img1 = cv2.flip(img, k);
            img_data.append(img_to_array(img))
    # img = cv2.imread('/home/netrunner/Desktop/Raks/val_256/458_c.tiff',1)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img,(Size,Size))
    # img_data.append(img_to_array(img))
    
    img_array = np.reshape(img_data, (len(img_data), Size, Size, 3))
    img_array = img_array.astype('float32')/255.
    
    return img_array

def prepareNIRData():
    img_data2 = []
    
    path2 ='val_256/NIR'
    files = os.listdir(path2)
    for i in tqdm(files):
        img = cv2.imread(path2 + '/' + i, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (Size, Size))
        # img = cropSquare(img, Size)
        # img = savgol_filter(img, window_length=255, polyorder=3, axis=0)
        # img = savgol_filter(img, window_length=255, polyorder=3, axis=1)
        img_data2.append(img_to_array(img))
        for j in range(3):
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            # for k in range(2):
            #     img1 = cv2.flip(img, k);
            img_data2.append(img_to_array(img))
    # img2 = cv2.imread('/home/netrunner/Desktop/Raks/val_256/458.tiff',1)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    # img2 = cv2.resize (img2,(Size,Size))
    # img_data2.append(img_to_array(img2))
    
    img_array2 = np.reshape(img_data2, (len(img_data2), Size, Size, 3))
    img_array2 = img_array2.astype('float32')/255.
    
    
    return img_array2

def testimage():
    img_data3 = []
    
    img = cv2.imread('val_256/0.jpg', 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (Size, Size))
    # img = cropSquare(img, Size)
    img_data3.append(img_to_array(img))
    
    img_array3 = np.reshape(img_data3, (len(img_data3), Size, Size, 3))
    img_array3 = img_array3.astype('float32')/255.
    
    return img_array3

def model(arr1, arr2):
    model = Sequential()
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=(Size, Size, 3)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    # model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    # model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    # model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    # model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
    # model.add(Conv2D(256, (3,3), activation='relu', padding='same', strides=2))
    # model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
    # model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
    # model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
    
    # model.add(Conv2D(16, (3,3), activation='relu', padding='same'))
    # model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    # model.add(UpSampling2D((2, 2)))
    # model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
    # model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    # model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))
    # model.add(UpSampling2D((2, 2)))

    
    model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])
    model.summary()
    
    model.fit(arr1, arr2, epochs = 10, batch_size = 20, validation_split=0.1, verbose = 1)

    model.save('val_256/model1.model') 
    
    return model

    
def output(m1, arr3):
    print("Output")
    output = m1.predict(arr3)

    imshow(output[0].reshape(Size, Size, 3))
    
def groundTruth():
    img = cv2.imread('val_256/457_sol.tiff', 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cropSquare(img, Size)
    imshow(img)

def main():
    NIR_array = prepareNIRData()
    RGB_array = prepareRGBData()
    m1 = model(RGB_array, NIR_array)
    test_array = testimage()
    output(m1, test_array)

def testing():
    t= "val_256/457.tiff"
    img=load_img(t)
    data = img_to_array(img)
    samples = tf.expand_dims(data, 0)
    datagen = ImageDataGenerator(brightness_range=[0.2,1.0])
    it = datagen.flow(samples, batch_size=4)
    for y in range(3):
        batch=it.next()
        imag=batch[0].astype('uint8')
        imshow(imag)
        # save_img("/home/netrunner/Music/img_folder2/98/2"+str(y)+".jpg",imag)
        # show()

if __name__ == "__main__":
    main()
    # testing()
    # groundTruth()
