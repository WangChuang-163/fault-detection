# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 21:58:00 2019

@author: wangc
"""


import numpy as np
import torch

import matplotlib.pyplot as plt # plt 用于显示图片

## load the dataset 
import dataset
cifar = dataset.FlameSet('48DriveEndFault','1772',4096 ,'2D', 'net_classifier')


traindata_id, testdata_id= cifar._shuffle()

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

## create training and validation sampler objects
tr_sampler = SubsetRandomSampler(traindata_id)
val_sampler = SubsetRandomSampler(testdata_id)
## create iterator objects for train and valid datasets
trainloader = DataLoader(cifar, batch_size=50, sampler=tr_sampler, shuffle=False)  #dataset就是Torch的Dataset格式的对象；batch_size即每批训练的样本数量，默认为；
validloader = DataLoader(cifar, batch_size=1, sampler=val_sampler, shuffle=False)  #shuffle表示是否需要随机取样本；num_workers表示读取样本的线程数。

#print (len(trainloader)*50)
#print (len(validloader)*50)

# Define model
from torch import nn

import model
net_classifier = model.CNN2d_classifier()  #选择神经网络模型
net_fitting_ball = model.CNN2d_fitting()
net_fitting_inner = model.CNN2d_fitting()
net_fitting_outer = model.CNN2d_fitting()

PATH = 'trained_model/net2.pkl' #net1为1D卷积神经网络模型，net2为2D卷积神经网络模型
# Model class must be defined somewhere
net_classifier = torch.load(PATH)     # 加载训练过的模型
net_classifier.eval()

PATH = 'trained_model/net_fitting_inner_2D_4096_unnormal.pkl' #net1为1D卷积神经网络模型，net2为2D卷积神经网络模型
# Model class must be defined somewhere
net_fitting_inner = torch.load(PATH)     # 加载训练过的模型
net_fitting_inner.eval()


total_correct = 0
pred_correct = 0
i = 0
for x,y in validloader:             # 训练误差
    #x = x.view(x.size(0), 3*32*32)
    out = net_classifier(x)
    # out:[b, 10]
    pred = out.argmax(dim=1)
    
    correct = pred.eq(y).sum().float().item()
    total_correct += correct
    
    if pred in [0, 1, 2]:
        print ('ball error')
    elif pred in [3, 4, 5]:
        print ('InnerRace error')
        out = net_fitting_inner(x)
        print("size of fualt: '{}'".format(out))
        #if abs(out-1)<0.1:
            
    elif pred in [6, 7, 8]:
        print ('OuterRace6 error')
    elif pred in [9]:
        print ('normal')
    else:
        print ('unexpected error')
        
    i += 1
    if i >20:
        break
    #loss = loss_function(out, y)
    #print(loss.item())
    
total_num = 20 # len(validloader)*1
acc = total_correct/total_num
print('test_acc', acc)


