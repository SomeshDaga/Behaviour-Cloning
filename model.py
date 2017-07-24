import os
import csv
import numpy as np

from keras.models import Sequential
from keras.layers import Cropping2D
from keras.layers.core import Flatten, Dense, Activation, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

from keras.optimizers import Adam

import sklearn
from sklearn.model_selection import train_test_split

samples = []
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None) # Skip the header row
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn

scaling = 0.3
height = int(scaling * 160)
width = int(scaling * 320)

crop_top = int(50 * scaling)
crop_bottom  = int(20 * scaling)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                path = '../data/'
                center_image = cv2.imread(path + batch_sample[0])
                left_image = cv2.imread(path + batch_sample[1].lstrip())
                right_image = cv2.imread(path + batch_sample[2].lstrip())

                center_image = cv2.resize(center_image, (width, height), interpolation = cv2.INTER_AREA)
                left_image = cv2.resize(left_image, (width, height), interpolation = cv2.INTER_AREA)
                right_image = cv2.resize(right_image, (width, height), interpolation = cv2.INTER_AREA)

                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)

                center_angle = float(batch_sample[3])
                left_angle = center_angle + 0.25
                right_angle = center_angle - 0.25

                # Flip half of the images in the horizontal axis
                if(np.random.uniform() < 0.5):
                    center_image = cv2.flip(center_image, 1)
                    left_image = cv2.flip(left_image, 1)
                    right_image = cv2.flip(right_image, 1)
                    center_angle = -1*center_angle
                    left_angle = -1*left_angle
                    right_angle = -1*right_angle


                images.append(center_image)
                images.append(left_image)
                images.append(right_image)
                
                angles.append(center_angle)
                angles.append(left_angle)
                angles.append(right_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Cropping2D(cropping=((crop_top, crop_bottom), (0,0)), input_shape=(height, width, 3)))
model.add(Lambda(lambda x: x/255.0 - 0.5)) # Normalization
model.add(Convolution2D(6,2,2,activation='tanh'))
#model.add(Dropout(0.2))
model.add(Convolution2D(6,2,2,activation='tanh'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(100))
model.add(Activation('tanh'))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Activation('tanh'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation('tanh'))


model.compile(loss='mse', optimizer=Adam(lr=5e-4))
model.fit_generator(train_generator, samples_per_epoch= 
            3*len(train_samples), validation_data=validation_generator, 
            nb_val_samples=3*len(validation_samples), nb_epoch=7)
model.save('model.h5')