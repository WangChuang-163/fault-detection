# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 20:05:32 2019

@author: wangc
"""
import torch
 
import scipy.io as scio   #导入库函数
from scipy.fftpack import fft
import numpy as np
from torch.autograd import Variable

from torch.utils import data
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
from torchvision import transforms

import os

import dataset
cifar = dataset.FlameSet('48DriveEndFault', '1772', 4096, '2D', 'net_classifier')

from torch import nn
import model
net = model.CNN2d_classifier()

import torch
PATH = 'trained_model/net_classifier_2D_4096_unnormal_10label.pkl'
#torch.save(net, PATH)

# Model class must be defined somewhere
net = torch.load(PATH)
net.eval()

class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
#            if name is "linear1": x = x.view(x.size(0), -1)
#            x = module(x)  # last layer output put into current layer input
            print(name)
            if name in self.extracted_layers:
                x = module(x)
                outputs.append(x)
        return outputs

myresnet=net
exact_list=["conv1", "conv2", "conv3", "conv4"]
myexactor=FeatureExtractor(myresnet,exact_list)

x,y = cifar[4005]
print (x.shape, y)
x=x.unsqueeze(0)

y=myexactor(x)    # 5x64x112x112  5x64x56x56  5x512x1x1
print (myexactor)

for i in range(len(y)):
    print (y[i].shape)

plt.figure(figsize=(5,5))
ax = plt.subplot(1,1,1)
ax.set_title("Sample ")
plt.imshow(x[0][0].data.numpy())

plt.figure(figsize=(25,6))
for i in range(len(y[0][0])):
    ax = plt.subplot(3,12,i+1)
    ax.set_title("feature '{}'".format(i))
    ax.axis('off')
    ax.imshow(y[0][0][i].data.numpy())
    
plt.figure(figsize=(25,12))
for i in range(len(y[1][0])):
    ax = plt.subplot(6,12,i+1)
    ax.set_title("feature '{}'".format(i))
    ax.axis('off')
    ax.imshow(y[1][0][i].data.numpy())

plt.figure(figsize=(25,24))
for i in range(len(y[2][0])):
    ax = plt.subplot(11,12,i+1)
    ax.set_title("feature '{}'".format(i))
    ax.axis('off')
    ax.imshow(y[2][0][i].data.numpy())

plt.show()