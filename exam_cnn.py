# encoding:utf-8

# import numpy as np
# import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.optimizers import Adam

from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.utils import plot_model
from keras.callbacks import TensorBoard

from keras.datasets import cifar10
from keras.utils import np_utils


# NUM_CLASSES = 50

img_rows, img_cols = 32, 32
img_channels = 3
nb_classes = 10
nb_epoch = 3

batch_size = 128

# cifar-10のダウンロード
(X_train, y_train),(X_test, y_test) = cifar10.load_data()
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# 画素値を0-1に変換する
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.0
X_test /= 255.0

# one-hotエンコーディング
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


### add for TensorBoard
import keras.callbacks
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf

old_session = KTF.get_session()
session = tf.Session('')
KTF.set_session(session)
KTF.set_learning_phase(1)
###

# モデルの定義
model = Sequential()

model.add(Conv2D(32,3,input_shape=(32,32,3)))
model.add(Activation('relu'))
model.add(Conv2D(32,3))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(64,3))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(1.0))

model.add(Dense(nb_classes, activation='softmax'))

adam = Adam(lr=1e-4)

model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=["accuracy"])

### add for Tensorflow

tb_cb = keras.callbacks.TensorBoard('./logs/', histogram_freq=1)
cbks = [tb_cb]
###

# モデルのサマリを表示
# model.summary()
plot_model(model, to_file='./model2.png')

history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_split=0.1, callbacks=cbks)

# 学習履歴をプロット
TensorBoard(log_dir='./logs')

### add for TensorBoard

KTF.set_session(old_session)

###