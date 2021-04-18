import cv2
# import dlib
from matplotlib.pyplot import imshow
import numpy as np
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

def frameRGBExtraction():
    path = '/home/netrunner/Downloads/RGB_vid'
    files = os.listdir(path)
    for i in tqdm(files):
        vid = cv2.VideoCapture("/home/netrunner/Downloads/97_rgb.mp4")
        count = 0
        while(True):
            ret, frame = vid.read()
            if ret:
                cv2.imwrite("/home/netrunner/Downloads/frames/%d.jpg" % count, frame)
                count += 1 
            else:
                break
            if(count == 30):
                break

def faceRGBExtraction():
    # path = '/home/netrunner/Downloads/frames'
    # files = os.listdir(path)
    # count = 0
    path1="/home/netrunner/Downloads/RGB_frames"
    path2="/home/netrunner/Downloads/RGB_frames/97"
    path3_o="/home/netrunner/Downloads/RGB_faces"
    
    for j in os.listdir(path1):
        path2=os.path.join(path1, j)
        path3=os.path.join(path3_o, j)
        count=0
        files = os.listdir(path2)
        for i in tqdm(files):
            img = cv2.imread(path2 + '/' + i)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in faces:
                # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                faces = img[y:y + h, x:x + w]
                cv2.imwrite(path3 +'/%d.jpg' % count, faces)
            
            count += 1
            
                            
# def featureRGBEx():
#     path = 'IITJ-Data/RGB'
#     files = os.listdir(path)
#     detector = dlib.get_frontal_face_detector()
#     predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#     count = 0
    
#     for i in tqdm(files):
#         img = cv2.imread(path + '/' + i)
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         faces = detector(gray)
        
#         for face in faces:
#             # x1 = face.left()
#             # y1 = face.top()
#             # x2 = face.right()
#             # y2 = face.bottom()
            
#             landmarks = predictor(image=gray, box=face)
#             # x3 = landmarks.part(17).x
#             # y3 = landmarks.part(17).y
#             x1 = landmarks.part(26).x
#             y1 = landmarks.part(26).y
#             x2 = landmarks.part(42).x
#             y2 = landmarks.part(42).y
#             x3 = landmarks.part(12).x
#             y3 = landmarks.part(12).y
            
#             # cv2.circle(img=img, center=(x4, y4), radius=5, color=(0, 255, 0), thickness=-1)
#             # cv2.rectangle(img=img, pt1=(x1, y1), pt2=(x4, y4), color=(0, 255, 0), thickness=1)
#             img1 = img[0:y1, 0:x1]
#             img2 = img[y2:y3, x2:x3]
        
#         cv2.imwrite('IITJ-Data/RGB/%d_head.jpg' % count ,img1)
#         cv2.imwrite('IITJ-Data/RGB/%d_cheek.jpg' % count ,img2)
#         imshow(img2)
#         count += 1
        
# def frameNIRExtraction():
#     path = 'IITJ-Data/NIR-video'
#     files = os.listdir(path)
#     for i in tqdm(files):
#         vid = cv2.VideoCapture(path + '/' + i)
#         count = 0
#         while(True):
#             ret, frame = vid.read()
#             cv2.imwrite("IITJ-Data/NIR/%d.jpg" % count, frame)
#             count += 1 
#             if(count == 30):
#                 break

# def faceNIRExtraction():
#     path = 'IITJ-Data/NIR'
#     files = os.listdir(path)
#     count = 0
    
#     for i in tqdm(files):
#         img = cv2.imread(path + '/' + i)
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
#         faces = face_cascade.detectMultiScale(gray, 1.1, 4)
#         for (x, y, w, h) in faces:
#             # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
#             faces = img[y:y + h, x:x + w]
#             cv2.imwrite('IITJ-Data/NIR/%d.jpg' % count, faces)
        
#         count += 1
            
                            
# def featureNIREx():
#     path = 'IITJ-Data/NIR'
#     files = os.listdir(path)
#     detector = dlib.get_frontal_face_detector()
#     predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#     count = 0
    
#     for i in tqdm(files):
#         img = cv2.imread(path + '/' + i)
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         faces = detector(gray)
        
#         for face in faces:
#             # x1 = face.left()
#             # y1 = face.top()
#             # x2 = face.right()
#             # y2 = face.bottom()
            
#             landmarks = predictor(image=gray, box=face)
#             # x3 = landmarks.part(17).x
#             # y3 = landmarks.part(17).y
#             x1 = landmarks.part(26).x
#             y1 = landmarks.part(26).y
#             x2 = landmarks.part(42).x
#             y2 = landmarks.part(42).y
#             x3 = landmarks.part(12).x
#             y3 = landmarks.part(12).y
            
#             # cv2.circle(img=img, center=(x4, y4), radius=5, color=(0, 255, 0), thickness=-1)
#             # cv2.rectangle(img=img, pt1=(x1, y1), pt2=(x4, y4), color=(0, 255, 0), thickness=1)
#             img1 = img[0:y1, 0:x1]
#             img2 = img[y2:y3, x2:x3]
        
#         cv2.imwrite('IITJ-Data/NIR/%d_head.jpg' % count ,img1)
#         cv2.imwrite('IITJ-Data/NIR/%d_cheek.jpg' % count ,img2)
#         imshow(img2)
#         count += 1

def RGB_extraction():
    frameRGBExtraction()
    faceRGBExtraction()
    # featureRGBEx()
    
# def NIR_extraction():
#     frameNIRExtraction()
#     faceNIRExtraction()
#     # featureNIREx()
        
def main():
    RGB_extraction()
    # NIR_extraction()
    
if __name__ == "__main__":
    main()