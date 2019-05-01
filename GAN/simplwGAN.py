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
    def get_generator(self,noise_img,n_units,out_dim,reuse=tf.AUTO_REUSE,alpha=0.01):#set reuse as True or tf.AUTO_REUSE, or None(subscope)
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
    def get_discriminator(self,img,n_units,reuse=tf.AUTO_REUSE,alpha=0.01):
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
        self.d_loss_real=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logist_real,labels=tf.ones_like(d_logist_real))*(1-smooth))
            #wrong on the book:logist--logits
        self.d_loss_fake=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logist_fake,labels=tf.ones_like(d_logist_fake)))
        self.d_loss=tf.add(self.d_loss_fake,self.d_loss_real)
        
        #generator should fraud the discriminator, so the logists and targets loss function is special
        self.g_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logist_fake,labels=tf.ones_like(d_logist_fake))*(1-smooth))
        
        #get the trainable vars
        train_vars=tf.trainable_variables()
        self.g_vars=[var for var in train_vars if var.name.startswith("generator")]
        self.d_vars=[var for var in train_vars if var.name.startswith("discriminator")]
        
        #optimizer
        #for two optimizers, var_list is necessary.
        d_train_opt=tf.train.AdamOptimizer(learning_rate).minimize(self.d_loss,var_list=self.d_vars)
        g_train_opt=tf.train.AdamOptimizer(learning_rate).minimize(self.g_loss,var_list=self.g_vars)
        
        self.saver=tf.train.Saver(var_list=self.g_vars)
        
        return d_train_opt,g_train_opt
    
    def training(self,d_train_opt,g_train_opt):
        samples=[]
        losses=[]
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for e in range(epoches):
                for batch_i in range(mnist.train.num_examples//batch_size):
                    batch=mnist.train.next_batch(batch_size)
                    batch_images=batch[0].reshape((batch_size,784))
                    #maybe 0<= batch_images <=1,and -1<= tanh() <=1,so...
                    batch_images=batch_images*2-1
                    
                    #the inputs of generator is random numbers in [-1,1]
                    batch_noise=np.random.uniform(-1,1,size=(batch_size,noise_size))
                    
                    #feed_dict of optimizers should only include the inputs that it depende on
                    sess.run(d_train_opt,feed_dict={self.real_img:batch_images,self.noise_img:batch_noise})
                    sess.run(g_train_opt,feed_dict={self.noise_img:batch_noise})
                    
                #collect information for evaluation
                train_loss_d=sess.run(self.d_loss,feed_dict={self.real_img:batch_images,self.noise_img:batch_noise})
                train_loss_d_real=sess.run(self.d_loss_real,feed_dict={self.real_img:batch_images,self.noise_img:batch_noise})
                train_loss_d_fake=sess.run(self.d_loss_fake,feed_dict={self.real_img:batch_images,self.noise_img:batch_noise})
                train_loss_g=sess.run(self.g_loss,feed_dict={self.noise_img:batch_noise})
                
                if(e%100==0):
                    print("Epoch {}/{}...".format(e+1,epoches),
                          "Discriminator Loss: {:.4f}(Real: {:.4f} + Fake: {:.4f})...".format(train_loss_d,train_loss_d_real,train_loss_d_fake),
                          "Generator Loss: {:.4f}".format(train_loss_g))
                
                #record th values of losses
                losses.append((train_loss_d,train_loss_d_real,train_loss_d_fake,train_loss_g))
                
                #record samples
                sample_noise=np.random.uniform(-1,1,size=(n_sample,noise_size))
                gen_samples=sess.run(self.get_generator(self,self.noise_img,g_units,img_size,reuse=True),feed_dict={self.noise_img:sample_noise})
                samples.append(gen_samples)
                
                #save models in every batch in a '.ckpt' file
                self.saver.save(sess,'./checkpoints/generator.ckpt')
                    
        with open('train_samples.pkl','wb') as f:
            pickle.dump(samples,f)#a Python object hierarchy is converted into a byte stream, and be saved in a '.pkl' file
        return losses
        
    def draw_loss(self,losses):
        fig,ax=plt.subplots(figsize=(20,7))
        losses=np.array(losses)
        plt.plot(losses.T[0],label="Discriminator Total Loss")
        plt.plot(losses.T[1],label="Discriminator Real Loss")
        plt.plot(losses.T[2],label="Discriminator Fake Loss")
        plt.plot(losses.T[3],label="Generator")
        plt.title("Training Losses")
        plt.legend()

    def draw_samples(self):
        epoch_index=[0,5,10,20,40,60,80,100,150,250]
        show_imgs=[]
        with open('train_samples.pkl','rb') as f:
            samples=pickle.load(f)
        for i in epoch_index:
            show_imgs.append(samples[i][1])
        
        rows,cols=10,25
        fig,axes=plt.subplots(figsize=(30,12),nrows=rows,ncols=cols,sharex=True,sharey=True)
        
        for sample,ax_row in zip(show_imgs,axes):
            for img,ax in zip(sample,ax_row):
                ax.imshow(img.reshape((28,28)),cmap='Grays_r')
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
                    
    def test(self):
        with tf.Session() as sess:
            self.saver.restore(sess,tf.train.latest_checkpoint('checkpoints'))
            sample_noise=np.random.uniform(-1,1,size=(25,100))#size=(n_sample,noise_size)
            gen_samples=sess.run(self.get_generator(self,self.noise_img,g_units,img_size,reuse=True),feed_dict={self.noise_img:sample_noise})
        self.view_samples(self,0,[gen_samples])
tf.reset_default_graph()      
#if you want to execute the graph creation again,you must add the code above        
gan=GAN(img_size=784,noise_size=100)
#train
d_train_opt,g_train_opt=gan.inference()
losses=gan.training(d_train_opt,g_train_opt)
gan.draw_loss(losses)
#test
gan.draw_samples()
gan.test()
                
                
                    
                    
                    
                    
                    
            
            
            
            
        