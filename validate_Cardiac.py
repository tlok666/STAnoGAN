#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 09:28:04 2019
 
@author: dragon
"""
import os
import tqdm
import torch as t
import numpy as np
import scipy.io as sio
from model.loss import l2_loss
from utils.config import config
from data.FlowData import FlowTestData
from torch.autograd import Variable
from model.networks import NetG, NetD
from torchnet.meter import AverageValueMeter
from utils.visualize import Visualizer
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
from model_tv.network import model
from model_tv.train_options import arguments

if __name__ == '__main__':
    args = arguments().parse()
    args.data_size = [config.batch_size, config.iChanel, config.iWidth, config.iHeight]
    netg, netd = NetG(config), NetD(config)
    net_st_fusion = model(args).cuda()
    
    error_meter = AverageValueMeter()
    map_location=lambda storage, loc: storage
    netg.load_state_dict(t.load('cpkt/netg_1500.pth', map_location = map_location))  
    netd.load_state_dict(t.load('cpkt/netd_1500.pth', map_location = map_location)) 
    net_st_fusion.load_state_dict(t.load('cpkt/netfusion_1500.pth', map_location = map_location)) 
    if config.gpu:
        netg.cuda()
        netd.cuda()
        net_st_fusion.cuda()
    criterion_L2 = l2_loss
    tsne=TSNE()
    
    vis = Visualizer(config.env)    
    root_path = 'dataset/Normal'
    Video_Paths = os.listdir(root_path)
    Videos_objs = [os.path.join(video_path, video_path) for video_path in Video_Paths]
    for iCnt in range(len(Videos_objs)):
        video_dir = os.path.join(root_path, Video_Paths[iCnt])
        datapair = FlowTestData(video_dir)
        dataloader = t.utils.data.DataLoader(datapair,
                                             batch_size = config.batch_size,
                                             shuffle = False,
                                             num_workers= config.num_workers,
                                             drop_last=True
                                             )   
        #latent_vector = np.zeros(int(datapair.imgs_len))
        latent_vector1 = np.zeros(int(datapair.imgs_len)) 
        error_meter.reset()
        for ii,(videos) in tqdm.tqdm(enumerate(dataloader)):
            img_3d = Variable(videos)
            if config.gpu: 
                img_3d = img_3d.cuda()
            img_st = net_st_fusion.forward(img_3d, need_result=True)

            fake, latent_i, latent_o = netg.forward(img_st)       
            error = criterion_L2(latent_i, latent_o)
            error_meter.add(error.data.cpu().numpy())
        #latent_vector[ii] = error_meter.value()[0]
            latent_vector1[ii] = error.data.cpu().numpy()
            vis.plot('Normal_%s' %iCnt, error.data.cpu().numpy())
            #vis.plot('Cardiac%s' %Video_Paths[iCnt], error.data.cpu().numpy())
            
            #vis.images(Attention_img[0,:,16,:,:].detach().cpu().numpy()*127.0 + 127.0, win='Input')
            #vis.images(fake[0,:,16,:,:].detach().cpu().numpy()*127.0 + 127.0, win='Reconstruct')
            #vis.images((Attention_img[0,:,16,:,:] - fake[0,:,16,:,:]).detach().cpu().numpy()*127.0 + 127.0, win='Residual')
            #s  = Mask_list[0][0,0,16,:,:].detach().cpu().numpy().squeeze()
            #s_m  = np.rot90(s, 2)
            #vis.heatmap(np.fliplr(s_m) , win='Mask')
            #tsne.fit_transform(latent_i.detach().cpu().view(1, 29*29*100).numpy())
           
        
        #latent_mat_path = os.path.join('result', 'Abnormal_%s.mat' %iCnt)
        #sio.savemat(latent_mat_path, {'latent': latent_vector})
        latent_mat_path = os.path.join('result/Cardiac', 'Cardiac_%s.mat' %Video_Paths[iCnt])
        sio.savemat(latent_mat_path, {'latent': latent_vector1})
    
        #vec_Normal = latent_vector
        #latent_mat_path = os.path.join('result', 'abnormal_result.mat')
        #sio.loadmat(latent_mat_path, {'latent': latent_vector})
        #latent_vector = np.concatenate((latent_vector, vec_Normal), axis=0)
    
        #latent_label1 = np.zeros(int(datapair.imgs_len))
        #latent_label2 = np.ones(int(datapair.imgs_len))
        #latent_label = np.concatenate((latent_label1, latent_label2), axis=0)
    
    #print "AUC is ",roc_auc_score(latent_label, latent_vector)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        