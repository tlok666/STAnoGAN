#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 17:15:15 2019

@author: dragon
"""

import os
import cv2 
import scipy
import imageio as imageio
import torch as t
import numpy as np
import torchvision as tv
from PIL import Image
from random import randint
from skimage import img_as_float
from utils.config import config
import scipy.ndimage

class FlowData(t.utils.data.Dataset):
    def __init__(self, root):
        imgs = os.listdir(root)  # Full path        
        self.imgs = [os.path.join(root, img) for img in imgs]   
        
    def __getitem__(self, index):
        img_path = self.imgs[index]
        cap = cv2.VideoCapture(img_path) 
        Data3d = np.zeros([int(config.iDepth), config.iChanel, int(config.iHeight), int(config.iWidth)])
        jCnt = 0
        jjCnt= 0
        try:
            jStart = randint(0, int(cap.get(7))-int(config.iDepth))
        except:
            print(img_path)
                
        while(cap.isOpened() & (jCnt<int(cap.get(7))) & (jCnt<int(config.iDepth))): 
            ret, frame = cap.read() 
            if jjCnt > jStart:
                try:
                    frame = scipy.ndimage.zoom(frame, zoom=(config.iHeight/frame.shape[0], config.iWidth/frame.shape[1], 1), order=1) # 360.0 640.0 # 600.0 800.0
                except:
                    print(img_path)
                try:
                    Data3d[jCnt,:,:,:] = np.transpose(img_as_float(frame), (2, 0, 1))
                except:
                    print(img_path)
                jCnt = jCnt+1
            jjCnt = jjCnt+1
        cap.release()     
        if jCnt < int(config.iDepth):
            print(jCnt)
        return t.from_numpy(Data3d).float()
    
    def __len__(self):
        return len(self.imgs)
    
class FlowFileData(t.utils.data.Dataset):
    def __init__(self, root):
        #imgs = os.listdir(root)  # Full path        
        #self.imgs = [os.path.join(root, img) for img in dirs]    #for root, dirs, files in os.walk(file_dir)
        
        for root1, dirs, files in os.walk(root):
            #print("root", root)  # 当前目录路径
            print("dirs", dirs)  # 当前路径下所有子目录
            break
        self.imgs = [os.path.join(root, img) for img in dirs]
            #print("files", files)  # 当前路径下所有非目录子文件
    def __getitem__(self, index):
        img_path = self.imgs[index]
        cap = os.listdir(img_path)
        Data3d = np.zeros([config.iChanel, int(config.iDepth), int(config.iHeight), int(config.iWidth)])
        jCnt = 0
        jjCnt= 0
        jStart = randint(0, int(len(cap)-int(config.iDepth)))
        while((jCnt<len(cap)) & (jCnt<int(config.iDepth))):  
            #print(os.path.join(img_path, cap[jStart + jCnt]))
            frame = imageio.mimread(os.path.join(img_path, cap[jStart + jCnt]))  
            if jjCnt > jStart:
                #frame = scipy.ndimage.zoom(frame, zoom=(config.iHeight/frame[0].shape[0], config.iWidth/frame[0].shape[1]), order=1) # 360.0 640.0
                frame = scipy.ndimage.zoom(frame[0], zoom=(config.iHeight/frame[0].shape[0], config.iWidth/frame[0].shape[1]), order=1)
                #Data3d[:,jCnt,:,:] = np.transpose(img_as_float(frame), (2, 0, 1))
                Data3d[:,jCnt,:,:] = (frame - 127.0)/127.0#frame
                jCnt = jCnt+1     
            jjCnt = jjCnt+1
        if jCnt < int(config.iDepth):
            print(jCnt)
        return t.from_numpy(Data3d).float()
    
    def __len__(self):
        return len(self.imgs)
   
class FlowFileTestData(t.utils.data.Dataset):
    def __init__(self, root):    
        imgs = os.listdir(root) 
        self.rootpath = root
        self.imgs_len = len(imgs) - config.iDepth
        
    def __getitem__(self, index):
        cap = os.listdir(self.rootpath)
        cap.sort()
        Data3d = np.zeros([config.iChanel, int(config.iDepth), int(config.iHeight), int(config.iWidth)])
        jCnt = 0
        while((jCnt<len(cap)) & (jCnt-index<int(config.iDepth)-1)): 
            try:
                frame = imageio.mimread(os.path.join(self.rootpath, cap[jCnt]))
            except:
                print(cap)
                print(self.rootpath)
            if (jCnt >= index) & (jCnt-index < int(config.iDepth)):
                frame = scipy.ndimage.zoom(frame[0], zoom=(config.iHeight/frame[0].shape[0], config.iWidth/frame[0].shape[1]), order=1)
                Data3d[:,jCnt-index,:,:] = (frame - 127.0)/127.0     # np.transpose(img_as_float(frame), (2, 0, 1)) # frame
                #print(frame)
            jCnt = jCnt+1     
            #print(jCnt, index)
        return t.from_numpy(Data3d).float()
    
    def __len__(self):
        return self.imgs_len
    
class FlowTestData(t.utils.data.Dataset):
    def __init__(self, root):
        imgs = cv2.VideoCapture(root)     
        self.rootpath = root
        self.imgs_len = imgs.get(7) - config.iDepth
        
    def __getitem__(self, index):
        print(index)
        cap = cv2.VideoCapture(self.rootpath) 
        Data3d = np.zeros([int(config.iDepth), config.iChanel, int(config.iHeight), int(config.iWidth)])
        jCnt = 0
        while(cap.isOpened() & (jCnt<int(cap.get(7)))): 
            ret, frame = cap.read() 
            if (jCnt >= index) & (jCnt-index < int(config.iDepth)):
                frame = scipy.ndimage.zoom(frame, zoom=(config.iHeight/float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), config.iWidth/float(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 1), order=1)
                Data3d[jCnt-index,:,:,:] = np.transpose(img_as_float(frame), (2, 0, 1))
            jCnt = jCnt+1
        cap.release()         
        return t.from_numpy(Data3d).float()
    
    def __len__(self):
        return self.imgs_len
    
    
if __name__ == '__main__':
    # Load all the 3D data
    root_path = '/home/dragon/Downloads/trails/FlowBasedVideoAnomaly/dataset/Normal'
    Video_Paths = os.listdir(root_path)
    Videos_objs = [os.path.join(video_path, video_path) for video_path in Video_Paths]   
    NumVideo = len(Video_Paths)
    for iCnt in range(1): # range(config.data_num):
        video_dir = os.path.join(root_path, Video_Paths[iCnt])     
        cap = cv2.VideoCapture(video_dir) 
        Data3d = np.zeros([3, int(cap.get(7)), 64, 64])
        jCnt = 0
        while(cap.isOpened() & (jCnt<int(cap.get(7)))): 
            ret, frame = cap.read() 
            frame = scipy.ndimage.zoom(frame, zoom=(64.0/600.0, 64.0/800.0, 1), order=1)
            Data3d[:,jCnt,:,:] = np.transpose(img_as_float(frame), (2, 0, 1))
            jCnt = jCnt+1
            cv2.imshow('image', frame) 
            k = cv2.waitKey(20) 
        cap.release() 
        
        
        
        
        
        