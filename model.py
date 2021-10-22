# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 09:01:13 2019

@author: wangc
"""

# Define model
import torch
from torch import nn
import torch.nn.functional as F


class CNN2d_classifier(nn.Module):
    
    def __init__(self):
         super(CNN2d_classifier, self).__init__()
         self.conv1 = nn.Conv2d(1, 32, 5, padding=2)      # x*32
         self.conv2 = nn.Conv2d(32, 64, 3, padding=1)     # x*2
         self.conv3 = nn.Conv2d(64, 128, 3, padding=1)    # x*2
         self.conv4 = nn.Conv2d(128, 256, 3, padding=1)   # x*2
         self.pool = nn.MaxPool2d(2, stride=2)            # x/(2*2)
         self.linear1 = nn.Linear(4096, 512)
         self.linear2 = nn.Linear(512, 128)
         self.linear3 = nn.Linear(128, 10)
         self.softmax = nn.LogSoftmax(dim=1)
         
    def forward(self, x):
        #print (x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        #print (x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        #print (x.shape)
        x = self.pool(F.relu(self.conv3(x)))
        #print (x.shape)
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 4096) ## reshaping 
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        x = self.softmax(x)
        return x

class CNN2d_fitting(nn.Module):
    
    def __init__(self):
         super(CNN2d_fitting, self).__init__()
         self.conv1 = nn.Conv2d(1, 32, 5, padding=2)      # x*32
         self.conv2 = nn.Conv2d(32, 64, 3, padding=1)     # x*2
         self.conv3 = nn.Conv2d(64, 128, 3, padding=1)    # x*2
         self.conv4 = nn.Conv2d(128, 256, 3, padding=1)   # x*2
         self.pool = nn.MaxPool2d(2, stride=2)            # x/(2*2)
         self.linear1 = nn.Linear(4096, 512)
         self.linear2 = nn.Linear(512, 128)
         self.linear3 = nn.Linear(128, 1)
         
         
    def forward(self, x):
        #print (x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        #print (x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        #print (x.shape)
        x = self.pool(F.relu(self.conv3(x)))
        #print (x.shape)
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 4096) ## reshaping 
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        
        return x

class CNN1d(nn.Module):
    
    def __init__(self):
         super(CNN1d, self).__init__()
         self.conv1 = nn.Conv1d(1, 32, 9, padding=4)   # x*32
         self.conv2 = nn.Conv1d(32, 64, 5, padding=2)  # x*2
         self.conv3 = nn.Conv1d(64, 128, 5, padding=2)  # x*2
         self.conv4 = nn.Conv1d(128, 256, 3, padding=1) # x*2
         self.pool = nn.MaxPool1d(4, stride=4)          # x/4
         self.linear1 = nn.Linear(4096, 512)
         self.linear2 = nn.Linear(512, 128)
         self.linear3 = nn.Linear(128, 3)
         self.softmax = nn.LogSoftmax(dim=1)
         
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))        # x*32/4=x*8
        x = self.pool(F.relu(self.conv2(x)))        # x*8*2/4=x*4
        x = self.pool(F.relu(self.conv3(x)))        # x*4*2/4=x*2
        x = self.pool(F.relu(self.conv4(x)))        # x*2*2/4=x
        x = x.view(-1, 4096) ## reshaping 
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        x = self.softmax(x)
        return x
'''
import sys
import torch
import tensorwatch as tw
import torchvision.models
model = torchvision.models.alexnet()
tw.draw_model(model, [1, 3, 224, 224])
'''

