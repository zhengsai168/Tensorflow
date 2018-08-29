from keras.layers import Dense,Activation,Conv2D,MaxPooling2D,Flatten
from tensorflow.examples.tutorials.mnist import input_data
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np

mnist = input_data.read_data_sets("D:/data/mnist_data",one_hot = True)
X_train,Y_train,X_test,Y_test=mnist.train.images,mnist.train.labels,mnist.test.images,mnist.test.labels
X_train=np.reshape(X_train,[-1,1,28,28])
X_test=np.reshape(X_test,[-1,1,28,28])

model=Sequential()
model.add(Conv2D(input_shape=(1,28,28),#Conv2D作为第一层时提供input_shape参数
                 filters=32,#卷积核数目
                 kernel_size=5,
                 strides=1,
                 padding='same',
                 data_format='channels_first' #当'channels_first'时 （samples,channels,row,column）
                 ))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2,
                       strides=2,
                       padding='same',
                       data_format='channels_first'
                       ))
model.add(Conv2D(64,5,strides=1,padding='same',data_format='channels_first'))
model.add(Activation('relu'))
model.add(MaxPooling2D(2,2,'same',data_format='channels_first'))
model.add(Conv2D(64, 5, strides=1, padding='same', data_format='channels_first'))
model.add(Activation('relu'))
model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))
model.add(Flatten())#展平，到1D
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))
model.compile(optimizer=Adam(),#Adam使用默认参数
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=1, batch_size=32,)
loss,accuracy=model.evaluate(X_test,Y_test)
print('loss :%g,accuracy: %g'%(loss,accuracy))

