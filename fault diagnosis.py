# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 17:35:43 2019

@author: wangc
"""

import scipy.io as scio  # 导入库函数
from scipy.fftpack import fft
import numpy as np
import torch
from torch.autograd import Variable

from torch.utils import data
import matplotlib.pyplot as plt  # plt 用于显示图片
import matplotlib.image as mpimg  # mpimg 用于读取图片
from torchvision import transforms

## load the dataset 
import dataset

cifar = dataset.FlameSet('48DriveEndFault', '1772', 4096, '2D', 'net_classifier')

traindata_id, testdata_id = cifar._shuffle()

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

## create training and validation sampler objects
tr_sampler = SubsetRandomSampler(traindata_id)
val_sampler = SubsetRandomSampler(testdata_id)
## create iterator objects for train and valid datasets
trainloader = DataLoader(cifar, batch_size=50, sampler=tr_sampler,
                         shuffle=False)  # dataset就是Torch的Dataset格式的对象；batch_size即每批训练的样本数量，默认为；
validloader = DataLoader(cifar, batch_size=1, sampler=val_sampler,
                         shuffle=False)  # shuffle表示是否需要随机取样本；num_workers表示读取样本的线程数。

# print (len(trainloader)*50)
# print (len(validloader)*50)

# Define model
from torch import nn

import model

net = model.CNN2d_classifier()  # 选择神经网络模型

import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr=0.01, weight_decay=1e-6, momentum=0.9, nesterov=True)
loss_function = nn.NLLLoss()
train_loss, valid_loss = [], []

for epoch in range(10):
    net.train()
    for batch_idx, (x, y) in enumerate(trainloader):

        out = net(x)
        # print(out, y)
        loss = loss_function(out, y)

        loss.backward()  # 计算倒数
        optimizer.step()  # w' = w - Ir*grad 模型参数更新
        optimizer.zero_grad()

        if batch_idx % 20 == 0:  # 训练过程，输出并记录损失值
            print(epoch, batch_idx, loss.item())

        train_loss.append(loss.item())  # loss仍然有一个图形副本。在这种情况中，可用.item()来释放它.(提高训练速度技巧)

index = np.linspace(1, len(train_loss), len(train_loss))  # 训练结束，绘制损失值变化图
plt.figure()
plt.plot(index, train_loss)
plt.show()

PATH = 'trained_model/net_classifier_2D_4096_unnormal_10label.pkl'  # net1为1D卷积神经网络模型，net2为2D卷积神经网络模型
torch.save(net, PATH)

# Model class must be defined somewhere
# net = torch.load(PATH)     # 加载训练过的模型

net.eval()

total_correct = 0
for x, y in trainloader:  # 训练误差
    # x = x.view(x.size(0), 3*32*32)
    # print(x,y)
    out = net(x)
    # out:[b, 10]
    pred = out.argmax(dim=1)
    correct = pred.eq(y).sum().float().item()
    total_correct += correct

    # loss = loss_function(out, y)
    # print(loss.item())

total_num = len(trainloader) * 50
acc = total_correct / total_num
print('train_acc', acc)

total_correct = 0
for x, y in validloader:  # 测试误差
    # x = x.view(x.size(0), 3*32*32)
    out = net(x)
    # print(out)
    # out:[b, 10]
    pred = out.argmax(dim=1)
    correct = pred.eq(y).sum().float().item()
    total_correct += correct

    # loss = loss_function(out, y)
    # print(loss.item())

total_num = len(validloader) * 1
acc = total_correct / total_num
print('test_acc', acc)
