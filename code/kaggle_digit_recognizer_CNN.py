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

for i in range(n_samples):
	train_y[i][train[i][0]]=1

my_keep_conv=0.8
my_keep_fc=0.5
batch_size=128
test_size=256

X=tf.placeholder(tf.float32,[None,784])
Y=tf.placeholder(tf.float32,[None,10])
keep_conv=tf.placeholder(tf.float32)
keep_fc=tf.placeholder(tf.float32)

def conv2d(x,W,b,strides=1):
	x=tf.nn.conv2d(x,W,[1,strides,strides,1],padding='SAME')+b
	return tf.nn.relu(x)
def maxpool2d(x,k=2):
	return tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME')

Weights={
	'wc1':tf.Variable(tf.random_normal([3,3,1,32])),
	'wc2':tf.Variable(tf.random_normal([3,3,32,64])),
	'wc3':tf.Variable(tf.random_normal([3,3,64,128])),
	'wfc1':tf.Variable(tf.random_normal([4*4*128,625])),
	'out':tf.Variable(tf.random_normal([625,10]))
}

biases={
	'bc1':tf.Variable(tf.random_normal([32])),
	'bc2':tf.Variable(tf.random_normal([64])),
	'bc3':tf.Variable(tf.random_normal([128])),
	'bfc1':tf.Variable(tf.random_normal([625])),
	'out':tf.Variable(tf.random_normal([10]))
}

def Net(x,Weights,biases,keep_conv,keep_fc):
	x=tf.reshape(x,[-1,28,28,1])
	#x [?,28,28,1]
	
	conv1=conv2d(x,Weights['wc1'],biases['bc1'])
	conv1=maxpool2d(conv1)
	conv1=tf.nn.dropout(conv1,keep_conv)
	#conv1 [?,14,14,32]
	
	conv2=conv2d(conv1,Weights['wc2'],biases['bc2'])
	conv2=maxpool2d(conv2)
	conv2=tf.nn.dropout(conv2,keep_conv)
	#conv2 [?,7,7,64]
	
	conv3=conv2d(conv2,Weights['wc3'],biases['bc3'])
	conv3=maxpool2d(conv3)
	#conv3 [?,4,4,128]
	
	conv3=tf.reshape(conv3,[-1,Weights['wfc1'].get_shape().as_list()[0]])
	conv3=tf.nn.dropout(conv3,keep_conv)
	#conv3 [?,4*4*128]
	
	fc1=tf.matmul(conv3,Weights['wfc1'])+biases['bfc1']
	fc1=tf.nn.relu(fc1)
	fc1=tf.nn.dropout(fc1,keep_fc)
	#fc1 [?,625]

	out=tf.matmul(fc1,Weights['out'])+biases['out']
	#out [?,10]

	return out

pred=Net(X,Weights,biases,keep_conv,keep_fc)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=pred))
optimizer=tf.train.AdamOptimizer(learning_rate=0.002).minimize(cost)
correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(Y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init=tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	for i in range(420):
		print(i)
		#print('!!!!!!!!!!!!!')
		sess.run(optimizer,feed_dict={X:train_x[i*100:i*100+100],Y:train_y[i*100:i*100+100],keep_conv:my_keep_conv,keep_fc:my_keep_fc})
		print(sess.run(accuracy,feed_dict={X:train_x[i*100:i*100+100],Y:train_y[i*100:i*100+100],keep_conv:1.0,keep_fc:1.0}))	
	result=sess.run(pred,feed_dict={X:test_x,Y:test_y,keep_conv:1.0,keep_fc:1.0})
frame=pd.DataFrame()
frame['ImageId']=np.arange(1,test_x.shape[0]+1)
result=np.argmax(result,axis=1)
frame['Label']=result
frame.to_csv('D:/test_y.csv',index=None)

