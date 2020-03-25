#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 17:18:16 2019

@author: dragon
"""

import torch as t
import numpy as np

class config(object):
    env = 'FlowVideoAnomaly'         # visdom的env
    video_path = 'dataset/Cardiac_Normal'
    validate_path = '/home/dragon/Downloads/trails/FlowBasedVideoAnomaly/dataset/abnormal/19-01-22-082431_HEYULAN-64-F-0017727933_20190122_152959_0046.MP4'
    iDepth = 11.0 #31.0
    iWidth = 256.0
    iHeight= 256.0
    iChanel= 3
    batch_size = 1
    num_workers = 16
    netd_path = None#'cpkt/netd_PED1.pth'#'cpkt/netd_400.pth' #'cpkt/netd_900.pth' # 'cpkt/netd_900.pth'
    netg_path = None#'cpkt/netg_PED1.pth'#'cpkt/netg_400.pth'#'cpkt/netg_900.pth' # 'cpkt/netg_900.pth'
    net_st_fusion = None
    max_epoch = 10000
    adjust_lr = 100
    isize = 32
    nc = 23  # input image channels
    nz =100 # size of the latent z vector
    ngf =64
    ndf =64
    extralayers = 0 # Number of extra layers on gen and disc
    gpu  = True  # If GPU is adopted
    ngpu = 1     # number of GPUs to use
    lr_g = 2e-4  # Learning rate of generator
    lr_d = 2e-4  # Learning rate of discriminator
    beta = 0.5   # beta1 parameter of Adam optimizer
    w_bce = 1    # alpha to weight bce loss
    w_rec = 50   # alpha to weight reconstruction loss
    w_enc = 1    # alpha to weight encoder loss
    gpu_ids = [0]   
