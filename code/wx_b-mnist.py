import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("D:/code/py/mnist_data",one_hot = True)

x=tf.placeholder(tf.float32,[None,784])
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
pred=tf.matmul(x,W)+b

y=tf.placeholder(tf.float32,[None,10])
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=pred))
optimizer=tf.train.GradientDescentOptimizer(0.5).minimize(loss)
accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,1),tf.argmax(pred,1)),tf.float32))

init=tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	for i in range(1000):
		batch_x,batch_y=mnist.train.next_batch(50)
		sess.run(optimizer,feed_dict={x:batch_x,y:batch_y})
		if i%10==0:
			print('step %d,accuracy: %g'%(i,sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})))

