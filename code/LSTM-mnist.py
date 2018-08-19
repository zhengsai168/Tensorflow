import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("D:/code/py/mnist_data",one_hot = True) 

lr=0.001
training_iters=100000
batch_size=128

n_inputs=28
n_steps=28
n_hidden_units=128
n_classes=10

X=tf.placeholder(tf.float32,[None,n_steps,n_inputs])
Y=tf.placeholder(tf.float32,[None,n_classes])

weights={
	'in':tf.Variable(tf.random_normal([n_inputs,n_hidden_units])),
	'out':tf.Variable(tf.random_normal([n_hidden_units,n_classes]))
}

biases={
	'in':tf.Variable(tf.constant(0.1,shape=[n_hidden_units,])),
	'out':tf.Variable(tf.constant(0.1,shape=[n_classes,]))
}

def RNN(X,weights,biases):
	X=tf.reshape(X,[-1,n_inputs])
	X_in=tf.matmul(X,weights['in'])+biases['in']
	X_in=tf.reshape(X_in,[-1,n_steps,n_hidden_units])

	lstm_cell=tf.contrib.rnn.BasicLSTMCell(n_hidden_units,forget_bias=1.0,state_is_tuple=True)
	init_state=lstm_cell.zero_state(batch_size,dtype=tf.float32)
	ouputs,final_state=tf.nn.dynamic_rnn(lstm_cell,X_in,initial_state=init_state,time_major=False)
	results=tf.matmul(final_state[1],weights['out'])+biases['out']
	return results

pred=RNN(X,weights,biases)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=pred))
train_op=tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred=tf.equal(tf.argmax(Y,1),tf.argmax(pred,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

with tf.Session() as sess:
	tf.global_variables_initializer().run()
	step=0
	while step*batch_size <training_iters:
		batch_xs,batch_ys=mnist.train.next_batch(batch_size)
		batch_xs=batch_xs.reshape([batch_size,n_steps,n_inputs])
		sess.run(train_op,feed_dict={X:batch_xs,Y:batch_ys})
		if step%5 ==0:
			print(sess.run(accuracy,feed_dict={X:batch_xs,Y:batch_ys}))
		step+=1

