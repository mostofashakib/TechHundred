import pandas as pd
import cv2
import numpy as np
import tensorflow as tf
import os
import keras
import sklearn
import skimage
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Lambda, MaxPool2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras import regularizers
from tqdm import tqdm
from keras.utils.np_utils import to_categorical
from sklearn.utils import shuffle
from skimage.transform import resize
from keras.applications.vgg16 import VGG16
from sklearn.utils import class_weight
from keras.models import Model


imageSize= 50
train_dir = 'C:/Users/19514/Desktop/hacklahoma/asl-alphabet/asl_alphabet_test/'
test_dir = 'C:/Users/19514/Desktop/hacklahoma/asl-alphabet/asl_alphabet_train/'

def get_data(folder):
    images = []
    labels  = []

    for folderName in os.listdir(folder):
        if not folderName.startswith('.'):
            if folderName in ['A']:
                label = 0
            elif folderName in ['B']:
                label = 1
            elif folderName in ['C']:
                label = 2
            elif folderName in ['D']:
                label = 3
            elif folderName in ['E']:
                label = 4
            elif folderName in ['F']:
                label = 5
            elif folderName in ['G']:
                label = 6
            elif folderName in ['H']:
                label = 7
            elif folderName in ['I']:
                label = 8
            elif folderName in ['J']:
                label = 9
            elif folderName in ['K']:
                label = 10
            elif folderName in ['L']:
                label = 11
            elif folderName in ['M']:
                label = 12
            elif folderName in ['N']:
                label = 13
            elif folderName in ['O']:
                label = 14
            elif folderName in ['P']:
                label = 15
            elif folderName in ['Q']:
                label = 16
            elif folderName in ['R']:
                label = 17
            elif folderName in ['S']:
                label = 18
            elif folderName in ['T']:
                label = 19
            elif folderName in ['U']:
                label = 20
            elif folderName in ['V']:
                label = 21
            elif folderName in ['W']:
                label = 22
            elif folderName in ['X']:
                label = 23
            elif folderName in ['Y']:
                label = 24
            elif folderName in ['Z']:
                label = 25
            elif folderName in ['del']:
                label = 26
            elif folderName in ['nothing']:
                label = 27
            elif folderName in ['space']:
                label = 28           
            else:
                label = 29
            for image_filename in tqdm(os.listdir(folder + folderName)):
                img_file = cv2.imread(folder + folderName + '/' + image_filename)
                if img_file is not None:
                    img_file = skimage.transform.resize(img_file, (imageSize, imageSize, 3))
                    img_arr = np.asarray(img_file)
                    images.append(img_arr)
                    labels.append(label)
    X_train = np.asarray(images)
    y_train = np.asarray(labels)

    return X_train, y_train

X_train, y_train = get_data(train_dir)


X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.15) 

X_train = X_train[:30000]
X_test = X_test[:30000]

weight_path1 = 'C:/Users/19514/Desktop/hacklahoma/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

pretrained_model_1 = VGG16(weights = weight_path1, include_top=False, input_shape=(imageSize, imageSize, 3))

base_model = pretrained_model_1 
x = base_model.output
predictions = Dense(30, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

map_characters = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'del', 27: 'nothing', 28: 'space', 29: 'other'}

def predict(gesture):
    img = cv2.resize(gesture, (50,50))
    img = img.reshape(1,50,50,1)
    img = np.resize(gesture, (1,50,50,3))
    img = img/255.0
    letter = model.predict(img)
    index = letter.argmax()
                        
    return map_characters[index]


Video_Capture = cv2.VideoCapture(0)
rval, frame = Video_Capture.read()
old_text = ''
pred_text = ''
count_frames = 0
total_str = ''
flag = False


while True:

    if frame is not None: 
        
        frame = cv2.flip(frame, 1)
        frame = cv2.resize( frame, (600,600) )
        
        cv2.rectangle(frame, (300,300), (100,100), (0,255,0), 2)
        
        crop_img = frame[100:300, 100:300]
        grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        
        thresh = cv2.threshold(grey,210,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
      
        blackboard = np.zeros(frame.shape, dtype=np.uint8)
        cv2.putText(blackboard, "Translated English  ", (30, 40), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2)
        if count_frames > 20 and pred_text != "":
            total_str += pred_text
            count_frames = 0
            
        if flag == True:
            old_text = pred_text
            pred_text = predict(thresh)
        
            if old_text == pred_text:
                count_frames += 1
            else:
                count_frames = 0
            cv2.putText(blackboard, total_str, (30, 80), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 130))
        
        res = np.hstack((frame, blackboard))
        
        cv2.imshow("image", res)
        
    rval, frame = Video_Capture.read()
    keypress = cv2.waitKey(1)

    if keypress == ord('c'):
        flag = True
    if keypress == ord('q'):
        break

Video_Capture.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
Video_Capture.release()