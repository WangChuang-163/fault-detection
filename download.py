# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 21:02:56 2019

@author: wangc
"""

import os
#import glob
import errno
import urllib
import urllib.request

class Download():

    def __init__(self, exp, rpm):
        if exp not in ('12DriveEndFault', '12FanEndFault', '48DriveEndFault'):
            print ("wrong experiment name: '{}'".format(exp))
            exit(1)
        if rpm not in ('1797', '1772', '1750', '1730'):
            print ("wrong rpm value: '{}'".format(rpm))
            exit(1)
        ## root directory of all data 
        rdir = os.path.join('', 'Datasets')
        
        ## 从metdata.txt文件中读取所有的文件名及相应的网址
        fmeta = os.path.join(os.path.dirname(__file__), '3metadata.txt')
        all_lines = open(fmeta).readlines()
        lines = []
        for line in all_lines:
            l = line.split()
            if (l[0] == exp or l[0] == 'NormalBaseline') and l[1] == rpm: 
                lines.append(l)     # 读取某种故障位置与正常情况下，在某种载荷下的振动信号的文件名及网址

        self._load(rdir, lines)


    def _mkdir(self, path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else:
                print ("can't create directory '{}'".format(path))
                exit(1)

    def _download(self, fpath, link):
        print ("Downloading to: '{}'".format(fpath))
        urllib.request.urlretrieve(link, fpath)

    def _load(self, rdir, infos):
        
        for idx, info in enumerate(infos):    # 遍历 matdata.txt, take out file name and URL
            # directory of this file
            fdir = os.path.join(rdir, info[0], info[1])  # file folder name
            self._mkdir(fdir)
            fpath = os.path.join(fdir, info[2] + '.mat')  # file name
            if not os.path.exists(fpath):
                self._download(fpath, info[3].rstrip('\n'))

    
cifar=Download('48DriveEndFault','1730')
