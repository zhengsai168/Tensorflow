#程序参考的是黄文坚的《tensorflow实战》
#输入为 28*28
#第一个卷积层的卷积核尺寸为5*5，通道数为1（灰度图像），32个卷积核，进行2*2的最大池话，由于卷积的padding参数选的是same，所以第一次卷积并池化后
#的尺寸是14*14*32
#第二层的卷积和池化，和第一层基本一致，唯一不同是卷积核数量变成64，输出为7*7*64，然后将其向量化
#然后连接一个1024个节点的全连接层，用ReLU激活函数
#书中还在之后使用了dropout层，基本原理是给神经元设置概率p，其在运算的时候以p概率舍弃，这样可以减轻过拟合。
#最后使用softmax回归得到概率，以此判断数字。

import tensorflow as tf   
from tensorflow.examples.tutorials.mnist import input_data  
import os

mnist = input_data.read_data_sets("D:/code/py/mnist_data",one_hot = True)  
sess = tf.InteractiveSession()  

def weight_variable(shape):  
    initial = tf.truncated_normal(shape,stddev=0.1)  
    return tf.Variable(initial)  
  
def bias_variable(shape):  
    initial = tf.constant(0.1,shape = shape)  
    return tf.Variable(initial)  
  
def conv2d(x,W):  
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')  
  
def max_pool_2x2(x):  
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')  
  
x = tf.placeholder(tf.float32,[None,784])  
y_ = tf.placeholder(tf.float32,[None,10])  
x_image = tf.reshape(x,[-1,28,28,1])  

# Conv1 Layer  
W_conv1 = weight_variable([5,5,1,32])  
b_conv1 = bias_variable([32])  
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)  
h_pool1 = max_pool_2x2(h_conv1)  
  
# Conv2 Layer  
W_conv2 = weight_variable([5,5,32,64])  
b_conv2 = bias_variable([64])  
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)  
h_pool2 = max_pool_2x2(h_conv2)  
  
W_fc1 = weight_variable([7*7*64,1024])  
b_fc1 = bias_variable([1024])  
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])  
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)  
  
keep_prob = tf.placeholder(tf.float32)  
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)  
  
W_fc2 = weight_variable([1024,10])  
b_fc2 = bias_variable([10])  
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)  
  
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv),reduction_indices=[1]))  
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  
  
correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))  
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))  
  
ckpt_dir='./ckpt_dir'
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
global_step=tf.Variable(0,name='global_step',trainable=False)
saver=tf.train.Saver()

with tf.Session() as sess:

    #tf.global_variables_initializer().run()  
    sess.run(tf.global_variables_initializer())
    for i in range(2000):  
        batch = mnist.train.next_batch(50)  
        if i % 2 == 0:  
            train_accuracy = accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})  
            print("step %d, training accuracy %g"%(i,train_accuracy))  
        train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})  
        #sess.run(train_step,feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})
        global_step.assign(i).eval()

        saver.save(sess,ckpt_dir+'/model.ckpt',global_step=global_step)
print("test accuracy %g"%accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))  