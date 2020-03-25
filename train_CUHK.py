#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 17:03:53 2019

@author: dragon
"""
import tqdm
import torch as t
from model.loss import l2_loss
from utils.config import config
from data.FlowData import FlowData
from torch.autograd import Variable
from model.networks import NetD, NetG,weights_init
from torchnet.meter import AverageValueMeter
from utils.visualize import Visualizer
from model_tv.network import model
from model_tv.train_options import arguments

if __name__ == '__main__':
    vis = Visualizer('CUHK')    
    args = arguments().parse()
    args.data_size = [config.batch_size, config.iChanel, config.iWidth, config.iHeight]
    datapair = FlowData('dataset/Avenue_Dataset/training_videos')
    dataloader = t.utils.data.DataLoader(datapair,
                                         batch_size = config.batch_size,
                                         shuffle = True,
                                         num_workers= config.num_workers,
                                         drop_last=True
                                         )   
    netg, netd = NetG(config), NetD(config)
    net_st_fusion = model(args).cuda()
    
    map_location=lambda storage, loc: storage
    if config.netd_path:
        netd.load_state_dict(t.load(config.netd_path, map_location = map_location)) 
    if config.netg_path:
        netg.load_state_dict(t.load(config.netg_path, map_location = map_location))  
    if config.net_st_fusion:
        net_st_fusion.load_state_dict(t.load(config.net_st_fusion, map_location = map_location))  
    # Optimizer definition
    optimizer_f = t.optim.Adam(list(net_st_fusion.parameters()), config.lr_g, betas=(config.beta, 0.999))
    optimizer_g = t.optim.Adam(list(netg.parameters()), config.lr_g, betas=(config.beta, 0.999))
    optimizer_d = t.optim.Adam(netd.parameters(), config.lr_d, betas=(config.beta, 0.999))
    criterion_L1 = t.nn.L1Loss().cuda() 
    criterion_BCE= t.nn.BCELoss().cuda()
    criterion_L2 = l2_loss
    
    # Loss Plot
    errord_meter = AverageValueMeter()
    errorg_meter = AverageValueMeter()
    errorLatent_meter = AverageValueMeter()
    
    if config.gpu:
        netd.cuda()
        netg.cuda()
        
    epochs = range(config.max_epoch)
    y_real_, y_fake_ = t.ones(841).cuda(), t.zeros(841).cuda()
    for epoch in iter(epochs):
        for ii,(videos) in tqdm.tqdm(enumerate(dataloader)):
            #imgs = imgs.permute(0, 3, 1, 2)
            #imgt = imgt.permute(0, 3, 1, 2)
            img_3d = Variable(videos)
            if config.gpu: 
                img_3d = img_3d.cuda()
            img_st = net_st_fusion.forward(img_3d, need_result=True)
            
            #--update_netd--    Update D network: Ladv = |f(real) - f(fake)|_2
            #self.pred_real, self.feat_real = self.netd(self.input)
            #self.pred_fake, self.feat_fake = self.netd(self.fake.detach())
            netd.zero_grad()
            fake, latent_i, latent_o = netg(img_st)
            out_d_real, feat_true = netd(img_st)
            out_d_fake, feat_fake = netd(fake.detach())
            err_d = .5*criterion_BCE(out_d_real, y_real_) + .5*criterion_BCE(out_d_fake, y_fake_) #+ criterion_L2(feat_real, feat_fake)
            err_d.backward(retain_graph=True)
            optimizer_d.step()
            optimizer_f.step()
            errord_meter.add(err_d.data.cpu().numpy())
            vis.plot('errord',errord_meter.value()[0])
            # If D loss is zero, then re-initialize netD
            if err_d.item() < 1e-5:
                netd.apply(weights_init)
            
            #--update_netg--    Update G network: log(D(G(x)))  + ||G(x) - x|| 
            netg.zero_grad()
            #out_g, _ = netd(fake)
            err_g_bce = criterion_L2(feat_true, feat_fake)  # l_adv
            err_g_l1l = criterion_L1(fake, img_st)          # l_con
            err_g_enc = criterion_L2(latent_i, latent_o)    # l_enc
            err_g = err_g_bce * config.w_bce + err_g_l1l * config.w_rec + err_g_enc * config.w_enc
            err_g.backward()
            optimizer_g.step()
            optimizer_f.step()
            errorg_meter.add(err_g.data.cpu().numpy())
            vis.plot('errorg',errorg_meter.value()[0])
            
            err_Latent = err_g_enc
            errorLatent_meter.add(err_Latent.data.cpu().numpy())
            vis.plot('errorLatent',errorLatent_meter.value()[0])
            #vis.images(((t.squeeze(fake[:,:,1,:,:],0).detach().cpu().numpy())), win='Fake')
            #vis.images(((t.squeeze(img_3d[:,:,1,:,:],0).detach().cpu().numpy())), win='Real')
            
            if epoch % config.adjust_lr == 0:
                t.save(net_st_fusion.state_dict(),'cpkt_CUHK/netfusion_%s.pth' %epoch)
                t.save(netd.state_dict(),'cpkt_CUHK/netd_%s.pth' %epoch)
                t.save(netg.state_dict(),'cpkt_CUHK/netg_%s.pth' %epoch)
                errord_meter.reset()
                errorg_meter.reset()

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        