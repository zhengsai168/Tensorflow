import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

my_keep_conv=0.8
my_keep_fc=0.5
batch_size=128
test_size=256

mnist = input_data.read_data_sets("D:/code/py/mnist_data",one_hot = True) 
trX,trY,teX,teY=mnist.train.images,mnist.train.labels,mnist.test.images,mnist.train.labels

#trX=tf.reshape(trX,[-1,28,28,1])
#teX=tf.reshape(teX,[-1,28,28,1])

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
optimzer=tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(Y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

with tf.Session() as sess:
	tf.global_variables_initializer().run()
	for i in range(100):
		#batch_x,batch_y=mnist.train.next_batch(batch_size)
		training_batch=zip(range(0,trX.shape[0],batch_size),range(batch_size,trY.shape[0],batch_size))
		for start,end in training_batch:
			sess.run(optimzer,feed_dict={X:trX[start:end],Y:trY[start:end],keep_conv:my_keep_conv,keep_fc:my_keep_fc})
		test_ind=np.arange(teX.shape[0])
		np.random.shuffle(test_ind)
		test_ind=test_ind[0:test_size]
		print(sess.run(accuracy,feed_dict={X:teX[test_ind],Y:teY[test_ind],keep_conv:1.0,keep_fc:1.0}))

