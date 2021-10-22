# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 21:47:08 2019

@author: wangc
"""

def train(net, trainloader, kind):
    
    from torch import nn
    import torch.optim as optim
    import matplotlib.pyplot as plt
    import numpy as np
    
    optimizer = optim.SGD(net.parameters(), lr=0.01, weight_decay= 1e-6, momentum = 0.9, nesterov = True)
    if kind =='classifier':
        loss_function = nn.NLLLoss()
    elif kind == 'fitting':
        loss_function = nn.MSELoss()
    else:
        print("input a wrong kind:'{}'".format(kind))
    
    train_loss = []
    
    for epoch in range(100):
        net.train()
        for batch_idx, (x, y) in enumerate(trainloader):
            
            out = net(x)
            y = y.float().unsqueeze(1)
            #print(out,y)
            loss = loss_function(out, y)
                   
            loss.backward()     # 计算倒数     
            optimizer.step()    # w' = w - Ir*grad 模型参数更新
            optimizer.zero_grad()
            
            if batch_idx % 10==0:    # 训练过程，输出并记录损失值
                print(epoch, batch_idx, loss.item())
            
            train_loss.append(loss.item())  #loss仍然有一个图形副本。在这种情况中，可用.item()来释放它.(提高训练速度技巧)
    
    index = np.linspace(1,len(train_loss),len(train_loss))  # 训练结束，绘制损失值变化图
    plt.figure()
    plt.plot(index, train_loss)
    plt.show()
    
    return net
