# coding: utf-8

from keras.models import model_from_json
from keras.utils.vis_utils import plot_model

from keras.models import Model, Sequential
from keras.layers.core import Dropout, Flatten, Dense
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam

import cv2
import numpy as np

# モデルのロード
json_string = open('./keras_data/cnn_model.json').read()
model = model_from_json(json_string)

model.load_weights('./keras_data/cnn_model_weight.hdf5')

# モデルの確認
plot_model(model, show_shapes=True)

# 中間層モデルの作成
layer_name = 'activation_4'
hidden_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

# 出力層の確認
img = cv2.imread('./keras_data/face.jpg')
img = cv2.resize(img, (100, 100))
target = np.reshape(img, (1, img.shape[0], img.shape[1], img.shape[2])).astype('float') / 255.0

hidden_output = hidden_layer_model.predict(target)

print(hidden_output.shape)


# 追加レイヤのモデルを生成
add_model = Sequential()
add_model.add(Flatten(input_shape=hidden_layer_model.input_shape[1:]))
add_model.add(Dense(256, activation='relu'))
add_model.add(Dropout(0.5))
add_model.add(Dense(5, activation='softmax'))

new_model = Model(inputs=model.input, outputs=add_model(model.output))

adam = Adam(lr=1e-5)

new_model.compute(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

# 現在のモデル構成を確認
for i in range(len(new_model.layers)):
    print(i, new_model.layers[i], new_model.layers[i].trainable)

for layer in model.layers[:12]:
    layer.trainable = False

print(new_model.summary())
