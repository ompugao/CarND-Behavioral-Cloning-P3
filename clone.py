#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import cv2
import numpy as np
import os

datadir = os.path.expanduser('~/sandbox/udacity')
images = []
measurements = []

with open(os.path.join(datadir, 'driving_log.csv')) as f:
    reader = csv.reader(f)
    for line in reader:
        def getimage(imgpathcolumn, measurementcolumn):
            source_path = line[imgpathcolumn]
            measurement = float(line[measurementcolumn])
            imgpath = os.path.join(datadir, 'IMG', os.path.basename(source_path))
            image = cv2.imread(imgpath)
            return image, measurement
        # center
        correction = 0.2
        image, measurement = getimage(0, 3)
        images.append(image)
        measurements.append(measurement)

        # left
        image, measurement = getimage(1, 3)
        images.append(image)
        measurements.append(measurement + correction)
        
        # right
        image, measurement = getimage(2, 3)
        images.append(image)
        measurements.append(measurement - correction)

# X_train = np.array(images)
# y_train = np.array(measurements)
print("augmenting data...")
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Cropping2D

model = Sequential()
model.add(Lambda(lambda x: (x/255.0 - 0.5), input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0,0))))
model.add(Conv2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)

model.save('model.h5')

