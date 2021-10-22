# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 08:23:27 2019

@author: wangc
"""

import dataset
cifar = dataset.FlameSet('48DriveEndFault', '1772', 4096, '2D', 'net_classifier')

import model
net = model.CNN1d()

import torch
PATH = 'trained_model/net_classifier_2D_4096_unnormal_10label.pkl'
#torch.save(net, PATH)

# Model class must be defined somewhere
net = torch.load(PATH)
net.eval()

from torch import nn
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
#            print(name)
            if name in self.extracted_layers:
                x = module(x)
                outputs.append(x)
        return outputs

exact_list=["conv1", "conv2", "conv3", "conv4"]
myexactor=FeatureExtractor(net,exact_list)

data,label = cifar[1]
print (data.shape, label)
data=data.unsqueeze(0)

feature = myexactor(data)    # 5x64x112x112  5x64x56x56  5x512x1x1
#print (myexactor)

for i in range(len(feature)):
    print (feature[i].shape)

import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(20,5))
ax = plt.subplot(1,1,1)
ax.set_title("Sample ")  
print (data.shape)  
index = np.linspace(1,len(data[0][0]),len(data[0][0]))  # 训练结束，绘制损失值变化图
plt.plot(index, data[0][0])
plt.show()
'''
for i in range(3):
    plt.figure(figsize=(20,2.5*len(feature[i][0])))
    for j in range(len(feature[i][0])):
        ax = plt.subplot(len(feature[i][0]),4,j+1)
        ax.set_title("feature '{}'".format(j))
#        ax.axis('off')
        index = np.linspace(1,len(feature[i][0][j]),len(feature[i][0][j]))
        ax.plot(index, feature[i][0][j].data.numpy())
    plt.show()
'''
for i in range(4):
    plt.figure(figsize=(20,5))
    for j in range(10):
        ax = plt.subplot(1,1,1)
        ax.set_title("feature '{}'".format(j))
#        ax.axis('off')
        index = np.linspace(1,len(feature[i][0][j]),len(feature[i][0][j]))
        ax.plot(index, feature[i][0][j].data.numpy())
    plt.show()
