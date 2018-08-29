import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

np.random.seed(1337)

X=np.linspace(-1,1,200)  #[-1,1]生成200个数, ndarray
np.random.shuffle(X)    #位置随机排列
Y=X*0.5+2+np.random.normal(0,0.05,(200,))
#plt.plot(X,Y)
plt.scatter(X,Y)
plt.show()

X_train,Y_train=X[:160],Y[:160]
X_test,Y_test=X[160:],Y[160:]

model=Sequential()
model.add(Dense(output_dim=1,input_dim=1))  #Dense全连接层
#choose loos and optimizer
model.compile(loss='mse',optimizer='sgd')
#training
print('training------------------')
for step in range(301):
    cost=model.train_on_batch(X_train,Y_train)
    if step%100==0:
        print('cost: %g'%(cost))

#test
print('testing-------------------')
cost=model.evaluate(X_test,Y_test,batch_size=40)
print('test cost :',cost)
W,b=model.layers[0].get_weights()
print('W: %g ,b: %g'%(W,b))