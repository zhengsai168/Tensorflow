# -*- coding: utf-8 -*-
# 6个简单ResUnit（Weight Layer1->relu->Weight Layer)
# 其中一个ResUnit由于y和x的size不一样（卷积层stride=2导致），方法即ResNet论文中提到的y=F(x,{Wi})+Wx
# 2个全连接层
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def conv2d(x,w,stride=1):
    return tf.nn.conv2d(x,w,strides=[1,stride,stride,1],padding='SAME')

def maxpool2d(x,k):
    return tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME')

def res_identity(x,w1,w2):  #朴素ResUnit
    x1=tf.nn.relu(conv2d(x,w1))
    x2=tf.nn.relu(conv2d(x1,w2))
    return tf.nn.relu(x+x2)

def res_change(x,w1,w2,w3):  #size变化的ResUnit
    x1 = tf.nn.relu(conv2d(x, w1,2))
    x2 = tf.nn.relu(conv2d(x1, w2))
    return tf.nn.relu(x2+conv2d(x,w3,2))

mnist = input_data.read_data_sets("D:/data/mnist_data",one_hot = True)

batch_size = 100
learning_rate = 0.003
learning_rate_decay = 0.97

Weights_in=tf.Variable(tf.random_normal([3,3,1,32]))

Weights_conv1=[]
for i in range(6):
    Weights_conv1.append(tf.Variable(tf.random_normal([3,3,32,32])))

Weights_change=tf.Variable(tf.random_normal([1,1,32,64]))

Weights_conv2=[]
Weights_conv2.append(tf.Variable(tf.random_normal([3,3,32,64])))
for i in range(5):
    Weights_conv2.append(tf.Variable(tf.random_normal([3, 3, 64, 64])))

Weights_fc1=tf.Variable(tf.random_normal([7*7*64,4096]))
Weights_out1=tf.Variable(tf.random_normal([4096,10]))

x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])
global_step=tf.Variable(0,trainable=False)

x1=tf.reshape(x,[-1,28,28,1])
conv1=conv2d(x1,Weights_in)
pool1=maxpool2d(conv1,2)
unit1=res_identity(pool1,Weights_conv1[0],Weights_conv1[1])
unit2=res_identity(unit1,Weights_conv1[2],Weights_conv1[3])
unit3=res_identity(unit2,Weights_conv1[4],Weights_conv1[5])
unit4=res_change(unit3,Weights_conv2[0],Weights_conv2[1],Weights_change)
unit5=res_identity(unit4,Weights_conv2[2],Weights_conv2[3])
unit6=res_identity(unit5,Weights_conv2[4],Weights_conv2[5])
fc1=tf.reshape(unit6,[-1,Weights_fc1.get_shape().as_list()[0]])
#fc1=tf.nn.dropout(fc1,0.75)
fc1=tf.matmul(fc1,Weights_fc1)
fc1=tf.nn.tanh(fc1)
fc1=tf.nn.dropout(fc1,0.75)
pred=tf.matmul(fc1,Weights_out1)

entropy =tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)
loss = tf.reduce_mean(entropy)
rate = tf.train.exponential_decay(learning_rate, global_step, 200, learning_rate_decay)
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(5001):
        x_b,y_b=mnist.train.next_batch(batch_size)
        train_op_, loss_, step, accuracy_, rate_ = sess.run([train_op, loss, global_step, accuracy, rate],
                                                            feed_dict={x: x_b, y: y_b})
        if i % 50 == 0:
            print("training step {0}, loss {1},accuracy {2},learning rate {3}".format(step, loss_, accuracy_, rate_))

print('ok')