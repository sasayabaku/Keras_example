
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers import Input

from keras.applications.vgg16 import VGG16

input_tensor = Input(shape=(150, 150,3))
model = VGG16(include_top=True, weights='imagenet', input_tensor=input_tensor, input_shape=None)


# FC層を構築

top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))

# 学習済みの全結合層の重みをロード
top_model.load_weights(os)