import tensorflow as tf
import numpy as np

'''
a=tf.constant([1.0,2.0])
b=tf.constant([3.0,4.0])
c=a*b
sess=tf.Session()
print(sess.run(c))
sess.close()
'''



'''
a=tf.constant([[-1.0,2.0,3.0,4.0]])
with tf.Session() as sess:
	b=tf.nn.dropout(a,0.5,noise_shape=[1,4])
	print(sess.run(b))
	b=tf.nn.dropout(a,0.5,noise_shape=[1,1])
	print(sess.run(b))	
'''




'''
x1 = tf.placeholder(tf.int16)
x2 = tf.placeholder(tf.int16)
y = tf.add(x1, x2)
# 用Python产生数据
li1 = [2, 3, 4]
li2 = [4, 0, 1]
# 打开一个session --> 喂数据 --> 计算y
with tf.Session() as sess:
    print(sess.run(y, feed_dict={x1: li1, x2: li2}))
'''




'''
x_data=np.random.rand(100).astype(np.float32)
y_data=x_data*0.1+0.3

Weights=tf.Variable(tf.random_uniform([1],-1.0,1.0))
bias=tf.Variable(tf.zeros([1]))

y=Weights*x_data+bias

loss=tf.reduce_mean(tf.square(y-y_data))
optimizer=tf.train.GradientDescentOptimizer(0.5)
train=optimizer.minimize(loss)


#init=tf.global_variables_initializer()
#sess.run(init)

#for i in range(201):
#	sess.run(train)
#	if i%20==0:
#		print(i,sess.run(Weights),sess.run(bias))

with tf.Session() as sess:
	tf.global_variables_initializer().run()
	for i in range(201):
		#sess.run(train)
		train.run()
		if i%20==0:
			print(i,sess.run(Weights),sess.run(bias))
'''






'''
a=tf.Variable(0,name='counter')
#print(a.name)
one=tf.constant(1)
new_a=tf.add(a,one)
update=tf.assign(a,new_a)

with tf.Session() as sess:
	tf.global_variables_initializer().run()
	for i in range(20):
		sess.run(update)
		print(sess.run(a))
'''






'''
a=tf.placeholder(tf.float32)
b=tf.placeholder(tf.float32)

c=tf.multiply(a,b)

with tf.Session() as sess:
	print(sess.run(c,feed_dict={a:2.0,b:4.0}))
'''


def add_layer(inputs,in_size,out_size,activation_function=None):
	Weights=tf.Variable(tf.random_normal([in_size,out_size()]))
	bias=tf.Variable(tf.zeros([1,out_size]+0.1))
	Wx_plus_b=tf.matmul(inputs,Weights)+bias
	if activation_function is None:
		outputs=Wx_plus_b
	else:
		outputs=activation_function(Wx_plus_b)
	return output