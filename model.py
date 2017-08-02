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

images = [] 
measurements = []

for line in lines:
    fn_c = data_folder + line[0] # center image
    fn_l = data_folder + line[1] # left
    fn_r = data_folder + line[2] # right
    
    image_c = cv2.imread(fn_c)
    images.append(image_c)
    measurement = float(line[3]) # steering 
    measurements.append(measurement)
        
    image_l = cv2.imread(fn_l)
    images.append(image_l)
    measurements.append(measurement + 0.09)
#   
    image_r = cv2.imread(fn_r)
    images.append(image_r)
    measurements.append(measurement - 0.09)  


## image augmentation
aug_images, aug_measurements = [], []
for image, measurement in zip(images, measurements):
    aug_images.append(image)
    aug_measurements.append(measurement)
    aug_images.append(cv2.flip(image,1))  # left->right flip
    aug_measurements.append(measurement*-1)
## image filterring, remove some of the images with 0 steering angle

    
X = np.array(aug_images)
y = np.array(aug_measurements)

#X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state = 42)
X_train = X
y_train = y

### Keras import
from keras.models import Sequential
from keras.layers import Flatten, Conv2D, Dense, Dropout, Lambda, Cropping2D
from keras.optimizers import Adam
from keras.regularizers import l2

### batch generator
#def batch_generator(input_shape, batch_size, X_data, y_data):
#    images = np.zeros((batch_size, input_shape[0],input_shape[1],input_shape[2]), dtype = np.float32)
#    measurements = np.zeros((batch_size,), dtype = np.float32)
#    while True:
#        X, y = shuffle(X_data, y_data)
#        for i in range(batch_size):
#          ind = random.randint(0, len(X)-1)
#          images[i] = X[ind]
#          measurements[i] = y[ind]*(1+ np.random.uniform(-0.10,0.10)) # non-zero steerring 
#          
#        yield images, measurements
        
input_shape = (160,320,3)
batch_size = 128
epochs = 3
keep_prob = 0.4

#train_feed = batch_generator(input_shape, batch_size, X_train, y_train)
#valid_feed = batch_generator(input_shape, batch_size, X_valid, y_valid)

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
#model.fit_generator(train_feed, samples_per_epoch = len(X_train), nb_epoch = epochs, validation_data = valid_feed, nb_val_samples = len(X_valid))
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, epochs = epochs)
model.save('model.h5')

