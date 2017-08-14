# encoding: utf-8

# 未完成

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam

import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def sin(x, T=100):
    return np.sin(2.0 * np.pi * x / T)

def toy_problem(T=100, ampl=0.05):
    x = np.arange(0, 2 * T + 1)
    noise = ampl * np.random.uniform(low=-1.0, high=1.0, size=len(x))
    return sin(x) + noise

def make_dataset(low_data, n_prev=100):

    data, target = [], []
    maxlen = 25

    for i in range(len(low_data)-maxlen):
        data.append(low_data[i:i + maxlen])
        target.append(low_data[i + maxlen])

    re_data = np.array(data).reshape(len(data), maxlen, 1)
    re_target = np.array(target).reshape(len(data), 1)

    return re_data, re_target



f = toy_problem()

g, h = make_dataset(f)

X_train, X_test, Y_train, Y_test = train_test_split(g, h, test_size=0.1)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
print(len(f))

length_of_sequence = len(g)
in_out_neurons = 1
n_hidden = 300

model = Sequential()
model.add(LSTM(n_hidden, batch_input_shape=(None, length_of_sequence, in_out_neurons), return_sequences=False))
model.add(Dense(in_out_neurons))
model.add(Activation("linear"))

optimizer = Adam(lr=0.001)
model.compile(loss="mean_squared_error", optimizer=optimizer)

# model.fit(X_train, Y_train, batch_size=600, nb_epoch=15, validation_data=(X_test, Y_test), callbacks=["early_stopping"])
model.fit(g, h, batch_size=300, nb_epoch=15, validation_split=0.1, callbacks=["early_stopping"])

# ノイズ付きサイン波をプロット
# plt.figure()
# plt.plot(f)
# plt.show()

