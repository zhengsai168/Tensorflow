import numpy as np
np.random.seed(1337)
#from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import RMSprop
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("D:/data/mnist_data",one_hot = True)
X_train,Y_train,X_test,Y_test=mnist.train.images,mnist.train.labels,mnist.test.images,mnist.test.labels
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
#y_train = np_utils.to_categorical(y_train, num_classes=10)
#y_test = np_utils.to_categorical(y_test, num_classes=10)
#转化为one-hot 向量

model=Sequential([Dense(32,input_dim=784),
                 Activation('relu'),
                 Dense(10),
                 Activation('softmax')
                ])
rmsprop=RMSprop(lr=0.001,rho=0.9,epsilon=None,decay=0.0)
model.compile(optimizer=rmsprop,loss='categorical_crossentropy',metrics=['accuracy'])
#categorical_crossentropy 需要label是(n_samples,n_classes) 这样的shape（交叉熵的定义可知）
model.fit(X_train,Y_train,epochs=2,batch_size=32)
loss,accuracy=model.evaluate(X_test,Y_test)
print('loss :%g ,accuracy :%g'%(loss,accuracy))
