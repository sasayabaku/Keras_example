# encoding: utf-8

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils

# kerasのMNISTデータの取得

(X_train, y_train), (X_test, y_test) = mnist.load_data()


# 配列の整形と，色の範囲を0-255 -> 0-1に変換
X_train = X_train.reshape(60000, 784) / 255
X_test = X_test.reshape(10000, 784) / 255


# 正解ラベルをダミー変数に変換
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# ネットワークの定義
model = Sequential([
        Dense(512, input_shape=(784,)),
        Activation('sigmoid'),
        Dense(10),
        Activation('softmax')
        ])

# 損失関数，最適化アルゴリズムなどの設定 + モデルのコンパイルを行う
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# 学習処理の実行 -> 変数histに進捗の情報が格納される
hist = model.fit(X_train, y_train, batch_size=200, verbose=1, epochs=3, validation_split=0.1)

# 予測
score = model.evaluate(X_test, y_test, verbose=1)
print("")
print('test accuracy : ', score[1])

# Chapter2 学習状況の可視化
import matplotlib.pyplot as plt
import seaborn

loss = hist.history['loss']
val_loss = hist.history['val_loss']

# lossのグラフ
plt.plot(range(3), loss, marker='.', label='loss')
plt.plot(range(3), val_loss, marker='.', label='val_loss')
plt.legend(loc='best', fontsize=10)
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

acc = hist.history['acc']
val_acc = hist.history['val_acc']

# accuracyのグラフ
plt.plot(range(3), acc, marker='.', label='acc')
plt.plot(range(3), val_acc, marker='.', label='val_acc')
plt.legend(loc='best', fontsize=10)
plt.grid()
plt.xlabel('epoch')
plt.ylabel('acc')
plt.show()

# ネットワークの可視化
from keras.utils import plot_model
plot_model(model, to_file='./model.png')
