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
            
            #logist & outputs
            logist = tf.layers.dense(hidden1,out_dim)
            outputs=tf.atanh(logist)
            return logist,outputs
        
    @staticmethod
    def get_discriminator(self,img,n_units,reuse=False,alpha=0.01):
        with tf.variable_scope("discriminator",reuse=reuse):
            #hidden layer
            hidden1=tf.layers.dense(img,n_units)
            hidden1=tf.maximum(alpha*hidden1,hidden1)
            
            #logist & outputs
            logist=tf.layers.dense(hidden1,1)
            #the purpose of discriminator is to diffrentiate original data and generated data,
            #so "out_dim" = 1
            outputs=tf.sigmoid(logist)
            return logist,outputs
        
    @staticmethod
    def view_samples(self,epoch,samples):
        fig,axes=plt.subplots(figsize=(7,7),nrows=5,ncols=5,sharey=True,sharex=True)
        for ax,img in zip(axes.flatten(),samples[epoch][1]):
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.imshow(img.reshape(28,28),cmap='Grays_r')
            
        return fig,axes
    
    def inference(self):
        g_logist,g_outputs=self.get_generator(self,self.noise_img,g_units,img_size)
        d_logist_real,d_outputs_real=self.get_discriminator(self,self.real_img,d_units)
        d_logist_fake,d_outputs_fake=self.get_discriminator(self,g_outputs,d_units,reuse=True)
        #cross entropy loss function:logist+targets 
        #smooth ???
        self.d_loss_real=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logist=d_logist_real,labels=tf.ones_like(d_logist_real))*(1-smooth))
        self.d_loss_fake=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logist=d_logist_fake,labels=tf.ones_like(d_logist_fake)))
        self.d_loss=tf.add(self.d_loss_fake,self.d_loss_real)
        
        #generator should fraud the discriminator, so the logists and targets loss function is special
        self.g_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logist=d_logist_fake,labels=tf.ones_like(d_logist_fake))*(1-smooth))
        
        #get the trainable vars
        train_vars=tf.trainable_variables()
        self.g_vars=[var for var in train_vars if var.name.startwith("generator")]
        self.d_vars=[var for var in train_vars if var.name.startwith("discriminator")]
        
        #optimizer
        d_train_opt=tf.train.AdamOptimizer(learning_rate).minimize(self.d_loss,var_list=self.d_vars)
        g_train_opt=tf.train.AdamOptimizer(learning_rate).minimize(self.g_loss,var_list=self.g_vars)
        
        self.saver=tf.train.Saver(var_list=self.g_vars)
        
        return d_train_opt,g_train_opt
            
            
            
            
        