#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 11:50:31 2019

@author: xyz
"""

import input_data
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
print("train set:%d"%(mnist.train.num_examples))
print("test set:%d"%(mnist.test.num_examples))