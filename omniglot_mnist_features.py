'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from keras.utils import plot_model

import numpy as np 
import os
import pandas as pd 
from scipy.misc import imshow
from sklearn import preprocessing
from skimage.transform import resize
from skimage.util import invert

input_file = '../CMU/AlphaLinkage/AlphaLinkage/data/mnist/mnist_train.csv'
df = pd.read_csv(input_file, header=None)
x_train = df.iloc[:,1:]
y = df.iloc[:,0]

x_train = np.asarray([np.reshape(x, (28,28)) for x in x_train.values])

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

print(type(x_train))
x_train = x_train.astype('float32')

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1)

model_extract = Model(model.inputs, model.layers[-3].output) # Dense(128,...)

folder = "../CMU/AlphaLinkage/AlphaLinkage/data/omniglot/images_background/"

for f in os.listdir(folder):
    with open(os.path.join(folder, f)) as f_in:
        df = pd.read_csv(f_in, header=None)
        y = df.iloc[:,0]
        image_data_raw = df.iloc[:,1:]
        image_data_raw.astype('float32')
        image_data = np.asarray([np.reshape(x, (105,105)) for x in image_data_raw.values])
        image_data = np.asarray([[[pixel / 255 for pixel in row] for row in img] for img in image_data])
        image_data = np.asarray([resize(img, (28, 28), anti_aliasing=True) for img in image_data])
        image_data = np.asarray([invert(img) for img in image_data])
        image_data = np.asarray([[[pixel * 255 for pixel in row] for row in img] for img in image_data])
        if K.image_data_format() == 'channels_first':
            image_data = image_data.reshape(image_data.shape[0], 1, img_rows, img_cols)
        else:
            image_data = image_data.reshape(image_data.shape[0], img_rows, img_cols, 1)
        image_data = image_data.astype('float32')
        cnn_features = model_extract.predict(image_data)
        with open(os.path.join("../CMU/AlphaLinkage/AlphaLinkage/data/omniglot/images_background_cnn_features/", f), 'w+') as f_out:
            for label, feature_vector in zip(y, cnn_features):
                data = np.insert(feature_vector, 0, int(label))
                out = ','.join([str(num) for num in data])
                f_out.write(out + '\n')