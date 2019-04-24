#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 22:13:37 2019

@author: xyz
copy from page 90-96 of the book <<TensorFlow 進階指南>>
"""

import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
import input_data

mnist = input_data.read_data_sets("MNIST_data/")
img_size=mnist.train.images[0].shape[0]
noise_size=100
g_units=128
d_units=128
alpha=0.01
learning_rate=0.001
smooth=0.1
batch_size=128
epoches=500
n_sample=25

class GAN():
    def __init__(self,img_size,noise_size):
        self.real_img=tf.placeholder(tf.float32,[None,img_size],name="real_img")
        self.noise_img=tf.placeholder(tf.float32,[None,noise_size],name="noise_img")
        
    '''
    静态方法只是名义上归属类管理，但是不能使用类变量和实例变量，是类的工具包
    放在函数前（该函数不传入self或者cls），所以不能访问类属性和实例属性
    '''        
    @staticmethod 
    def get_generator(self,noise_img,n_units,out_dim,reuse=False,alpha=0.01):
        with tf.variable_scope("generator",reuse=reuse):
            #hidden layer
            hidden1=tf.layers.dense(noise_img,n_units)  #.dense and .Dense
            #leaky ReLU
            hidden1=tf.maximum(alpha*hidden1,hidden1)   #activation function
            #drop out
            hidden1=tf.layers.dropout(hidden1,rate=0.2) #drop out some neurons to avoid overfitting
            ###i am thinking that maybe  a class Dense or something can be used to replace all these above
            
            
            
        