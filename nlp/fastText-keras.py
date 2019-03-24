# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Embedding, GlobalAveragePooling1D, Dense

VOCAB_SIZE = 2000
EMBEDDING_DIM = 100
MAX_WORD = 500
CLASS_NUM = 5


def build_fastText():
    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_WORD))  # 根据词序号取词向量
    model.add(GlobalAveragePooling1D())  # 对词向量求均值
    model.add(Dense(CLASS_NUM, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    model = build_fastText()
    print(model.summary())
