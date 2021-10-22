# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 08:01:21 2019

@author: wangc
"""

import torch

import scipy.io as scio  # 导入库函数
import numpy as np
from torch.utils import data

import os


def process1(datasetdata, length):  # [3,1024]->[3,32,32] list->array->tensor
    # Max = max(datasetdata)
    # Min = min(datasetdata)
    # for j in range(len(datasetdata)):
    # datasetdata[j] = (datasetdata[j]-Min)/(Max-Min)-0.5
    if length == 1024:
        traindata = torch.tensor(np.array(datasetdata).reshape(32, 32), dtype=torch.float)
    elif length == 4096:
        traindata = torch.tensor(np.array(datasetdata).reshape(64, 64), dtype=torch.float)
    return traindata


def process2(datasetdata, length):  # [3,1024] list->tensor
    #    Max = max(datasetdata)
    #    Min = min(datasetdata)
    #    for j in range(len(datasetdata)):
    #        datasetdata[j] = (datasetdata[j]-Min)/(Max-Min)-0.5
    traindata = torch.tensor(datasetdata, dtype=torch.float)
    return traindata


# 定义自己的数据集合
class FlameSet(data.Dataset):
    def __init__(self, exp, rpm, length, dimension, kind):

        if exp not in ('12DriveEndFault', '12FanEndFault', '48DriveEndFault'):
            print("wrong experiment name: '{}'".format(exp))
            exit(1)
        if rpm not in ('1797', '1772', '1750', '1730'):
            print("wrong rpm value: '{}'".format(rpm))
            exit(1)
        if kind not in ('net_classifier', 'net_fitting_ball', 'net_fitting_inner', 'net_fitting_outer'):
            print("wrong rpm value: '{}'".format(kind))
            exit(1)
        self.length = length
        self.data_id = 0
        self.dataset = np.zeros((0, self.length))
        self.label = []
        self.traindata_id = []
        self.testdata_id = []
        # root directory of all data 
        #        print (self.dataset)
        rdir = 'Datasets/48DriveEndFault'
        load = ['1730', '1750', '1772', '1797']

        #
        if kind == 'net_classifier':
            mydatalist = ['0.007-Ball.mat', '0.014-Ball.mat', '0.021-Ball.mat',
                          '0.007-InnerRace.mat', '0.014-InnerRace.mat', '0.021-InnerRace.mat',
                          '0.007-OuterRace6.mat', '0.014-OuterRace6.mat', '0.021-OuterRace6.mat',
                          'Normal.mat']

            mylabellist = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        #            mylabellist = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3]
        elif kind == 'net_fitting_ball':
            mydatalist = ['Normal.mat', '0.007-Ball.mat', '0.014-Ball.mat', '0.021-Ball.mat']
            mylabellist = [0, 1, 2, 3]
        elif kind == 'net_fitting_inner':
            mydatalist = ['Normal.mat', '0.007-InnerRace.mat', '0.014-InnerRace.mat', '0.021-InnerRace.mat']
            mylabellist = [0, 1, 2, 3]
        elif kind == 'net_fitting_outer':
            mydatalist = ['Normal.mat', '0.007-OuterRace6.mat', '0.014-OuterRace6.mat', '0.021-OuterRace6.mat']
            mylabellist = [0, 1, 2, 3]
        else:
            print("wrong rpm value: '{}'".format(kind))
            exit(1)

        #        mydatalist = ['0.007-Ball.mat', '0.007-InnerRace.mat', '0.007-OuterRace6.mat', 'Normal.mat']

        for idx in range(len(mydatalist)):  # 遍历故障形式
            for idy in range(len(load)):  # 遍历载荷种类
                rdir2 = os.path.join(rdir, load[idy])  # 载荷文件夹路径
                matdata_path = os.path.join(rdir2, mydatalist[idx])  # mat 文件路径
                mat_dict = scio.loadmat(matdata_path, squeeze_me=True)  # 读取mat文件，并将其存放到列表中，参数squeeze_me=True，自动压缩数据维度
                key = list(filter(lambda x: 'DE_time' in x, mat_dict.keys()))[0]
                time_series = mat_dict[key][:]  # take out 'DE_time' from mat file
                idx_last = -(time_series.shape[
                                 0] % self.length)  # according to the length defined,cut the data into segment
                clips = time_series[:idx_last].reshape(-1, self.length)
                print(clips.shape)
                n = clips.shape[0]
                print(idx)
                n_split = 4 * n // 5
                #                print (n_split)
                self.dataset = np.vstack((self.dataset, clips))
                #                self.label += [idx] * n
                self.label += [mylabellist[idx]] * n
                self.traindata_id += list(range(self.data_id, (self.data_id + n_split)))
                self.testdata_id += list(range((self.data_id + n_split), (self.data_id + n)))
                self.data_id += n

        if dimension == '2D':
            self.transforms = process1
        elif dimension == '1D':
            self.transforms = process2
        else:
            print('input a wrong dimension')

        # print (len(self.traindata_id))
        # print (len(self.testdata_id))
        # print (self.dataset.shape)
        # print (len(self.label))
        # print (self.data_id)

    def _shuffle(self):

        return self.traindata_id, self.testdata_id

    def __getitem__(self, index):
        pil_img = self.dataset[index]  # 根据索引，读取一个3X32X32的列表
        # print(np.array(pil_img).shape)
        data = self.transforms(pil_img, self.length)
        data = data.unsqueeze(0)  # 输入数据为1通道时，在第一维度进行升维，确保训练数据x具有3个维度
        # print(data.shape)
        label = self.label[index]

        return data, label

    def __len__(self):
        return len(self.dataset)


def loaddata(cifar):
    traindata_id, testdata_id = cifar._shuffle()

    from torch.utils.data import DataLoader
    from torch.utils.data.sampler import SubsetRandomSampler

    ## create training and validation sampler objects
    tr_sampler = SubsetRandomSampler(traindata_id)
    val_sampler = SubsetRandomSampler(testdata_id)
    ## create iterator objects for train and valid datasets
    trainloader = DataLoader(cifar, batch_size=50, sampler=tr_sampler,
                             shuffle=False)  # dataset就是Torch的Dataset格式的对象；batch_size即每批训练的样本数量，默认为；
    validloader = DataLoader(cifar, batch_size=50, sampler=val_sampler,
                             shuffle=False)  # shuffle表示是否需要随机取样本；num_workers表示读取样本的线程数。
    return trainloader, validloader


'''
import matplotlib.pyplot as plt
cifar = FlameSet('48DriveEndFault', '1772', 4096, '1D', 'net_classifier')
target = 0
data = []
label = []
plt.figure(figsize=(20, 5*10))
for i in range(len(cifar)):
    x, y = cifar[i]
    if y == target:
        data.append(x)
        label.append(y)
        target += 1
        ax = plt.subplot(10, 1, target)
        ax.set_title("condation '{}'".format(y))
        ax.plot(x[0].numpy())
    i += 102

plt.show()
'''

'''
import matplotlib.pyplot as plt
cifar = FlameSet('48DriveEndFault','1772',4096,'2D', 'net_classifier')
target = 0
data=[]
label=[]
for i in range(len(cifar)):
    x,y = cifar[i]
    if y==target:
        data.append(x)
        label.append(y)
        target += 1
        plt.figure(figsize=(5,5))
        ax = plt.subplot(1,1,1)
        ax.set_title("condation '{}'".format(y))
        ax.imshow(x[0])
    i +=102

plt.show()
'''

# print (x)
# x = np.transpose(x.numpy(), (1, 2, 0))
# print (x.shape)
# idx = 1600
# plt.figure()        # 二维数据的可视化

# idx = 6000
# for i in range(3):
#    x,y = cifar[idx+i]
#    ax = plt.subplot(1, 3, i+1)
#    ax.imshow(x[0])
#    print(cifar[idx+i])
# plt.show()
