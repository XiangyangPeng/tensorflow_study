#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 12:00:37 2019

@author: xyz
"""

import tensorflow as tf
import input_data as data

mnist=data.read_data_sets("MNIST_data",one_hot=True)
sess=tf.Session()#none in the book
#build the graph---------------------------------
x=tf.placeholder(tf.float32,[None,784])
#10 classes, so yhe ouput is ten probabilities
w=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
y=tf.nn.softmax(tf.matmul(x,w)+b)
y_=tf.placeholder(tf.float32,[None,10])
cross_entropy=-tf.reduce_sum(y_*tf.log(y))
#an opimizer
train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#initialize--------------------------------------
init=tf.global_variables_initializer()
sess.run(init)#none in the book
#train-------------------------------------------
for i in range(3000):
    #Batch processing
    batch_xs,batch_ys=mnist.train.next_batch(50)
    feed_dict={x:batch_xs,y_:batch_ys}
    sess.run(train_step,feed_dict)
#evaluate--------------------------------------
#in tf.argmax(input_tensor,axis) axis:0-col 1-row    
correct_prediction=tf.equal(tf.math.argmax(y,1),tf.math.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))

print(sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))
