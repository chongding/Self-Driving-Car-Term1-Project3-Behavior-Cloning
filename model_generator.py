# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 21:30:10 2017

@author: chong
Train Models
"""
import csv
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

data_folder = 'data/'
lines = []
with open(data_folder + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
lines = lines[1:] # remove title line

train_ratio = 0.8 # prcentage of train data
lines = shuffle(lines)
num_train = int(len(lines)*train_ratio)
lines_train = lines[:num_train]
lines_valid = lines[-(len(lines) - num_train):]

def img_load(path, fns, n_img):
    images = [] 
    measurements = []
    img_ind = np.random.randint(0, len(fns), n_img) 
    for ind in img_ind:
        fn_c = path + fns[ind][0] # center image
        fn_l = path + fns[ind][1] # left
        fn_r = path + fns[ind][2] # right
        
        image_c = cv2.imread(fn_c)
        images.append(image_c)
        measurement = float(fns[ind][3]) # steering 
        measurements.append(measurement)
            
        image_l = cv2.imread(fn_l)
        images.append(image_l)
        measurements.append(measurement + 0.09)
    #   
        image_r = cv2.imread(fn_r)
        images.append(image_r)
        measurements.append(measurement - 0.09)  
    return images, measurements

def img_augument(images, measurements):
    aug_images, aug_measurements = [], []
    for image, measurement in zip(images, measurements):
        aug_images.append(image)
        aug_measurements.append(measurement)
        aug_images.append(cv2.flip(image,1))  # left->right flip
        aug_measurements.append(measurement*-1)
    return np.array(aug_images), np.array(aug_measurements)

def batch_generator(path, lines, batch_size): 
    while True:
        images, measurements = img_load(path, lines, batch_size)
        images, measurements = img_augument(images, measurements)      
#       measurements = measurements_*(1+ np.random.uniform(-0.10,0.10)) # non-zero steerring
        yield images, measurements


### Keras import
from keras.models import Sequential
from keras.layers import Flatten, Conv2D, Dense, Dropout, Lambda, Cropping2D
from keras.optimizers import Adam
#from keras.regularizers import l2
        
input_shape = (160,320,3)
batch_size = 128
epochs = 3
keep_prob = 0.4

train_feed = batch_generator(data_folder, lines_train, batch_size)
valid_feed = batch_generator(data_folder, lines_valid, batch_size)

# Model Structure (Simplified Nvidia) this one is perfect
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape = input_shape))
model.add(Cropping2D(cropping=((60,20), (0,0)))) # pixles to remove:((top, bottom), (left, right))
model.add(Conv2D(24, (5, 5), activation='relu', subsample =(2,2)))
model.add(Conv2D(36, (5, 5), activation='relu', subsample =(2,2)))
model.add(Conv2D(48, (5, 5), activation='relu', subsample =(2,2)))
model.add(Conv2D(64, (3, 3), activation="relu", subsample =(1,1)))
model.add(Conv2D(64, (3, 3), activation='relu', subsample =(1,1)))
model.add(Flatten())
#model.add(Dropout(keep_prob))
model.add(Dense(200, activation='relu'))
#model.add(Dropout(keep_prob))
#model.add(Dense(100, activation='relu'))
#model.add(Dropout(keep_prob))
model.add(Dense(50, activation='relu'))
model.add(Dropout(keep_prob))
model.add(Dense(10, activation='relu'))
#model.add(Dropout(keep_prob))
model.add(Dense(1))

adam = Adam(lr = 0.0001)
model.compile(loss = 'mse', optimizer = adam, metrics = [ 'accuracy'])

model.fit_generator(train_feed, samples_per_epoch = 38400, nb_epoch = epochs, validation_data = valid_feed, nb_val_samples = 9600)
#model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, epochs = epochs)
model.save('model.h5')
