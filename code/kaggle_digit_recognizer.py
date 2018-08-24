import tensorflow as tf
import numpy as np
import pandas as pd

train=pd.read_csv('D:/data/Digit_recognizer/train.csv').values
test_x=pd.read_csv('D:/data/Digit_recognizer/test.csv').values
n_samples=train.shape[0]
#print(n_samples)
#n_samples=1000
train_x=train[0:n_samples,1:]
train_y=np.zeros((n_samples,10))
test_y=np.zeros((test_x.shape[0],10))

#one_hot coding
for i in range(n_samples):
	train_y[i][train[i][0]]=1

x=tf.placeholder(tf.float32,[None,784])
W=tf.Variable(tf.random_normal([784,10]))
b=tf.Variable(tf.random_normal([10]))
pred=tf.matmul(x,W)+b

y=tf.placeholder(tf.float32,[None,10])
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=pred))
optimizer=tf.train.GradientDescentOptimizer(0.5).minimize(loss)
accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,1),tf.argmax(pred,1)),tf.float32))

init=tf.global_variables_initializer()

#result=np.array()
with tf.Session() as sess:
	sess.run(init)
	for i in range(420):
		sess.run(optimizer,feed_dict={x:train_x[i*100:i*100+100],y:train_y[i*100:i*100+100]})
		print(sess.run(accuracy,feed_dict={x:train_x[i*100:i*100+100],y:train_y[i*100:i*100+100]}))	
	result=sess.run(pred,feed_dict={x:test_x,y:test_y})
frame=pd.DataFrame()
frame['ImageId']=np.arange(1,test_x.shape[0]+1)
result=np.argmax(result,axis=1)
frame['Label']=result
frame.to_csv('D:/test_y.csv',index=None)

