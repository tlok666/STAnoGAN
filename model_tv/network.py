import torch
import torch.nn as nn
import numpy as np
from .losses.flow_loss import flow_loss
from .net.tvnet import TVNet
from torch.autograd import Variable
from models import video_transforms
import models


class model(nn.Module):
    def __init__(self, args):
        super(model, self).__init__()

        self.args = args
        self.data_size = args.data_size

        ## Define neural networks...
        self.flow_net = TVNet(args)
        
        ## ...with their losses...
        #self.fusion_spatial = encoder3d(self, 3, 1)
        #self.fusion_temporal = encoder3d(self, 2, 1)
        self.fusion_spatial = models.__dict__['rgb_resnet152'](pretrained=True, num_classes=101)
        self.fusion_temporal = models.__dict__['flow_resnet152'](pretrained=True, num_classes=101)
        
        self.normalize_st = video_transforms.Normalize(mean=[0.485, 0.456, 0.406] * 1, std=[0.229, 0.224, 0.225] * 1)
        self.normalize_tp = video_transforms.Normalize(mean=[0.5, 0.5] * 10, std=[0.226, 0.226] * 10)

        ## ... and optimizers
        self.flow_optmizer = torch.optim.SGD(self.flow_net.parameters(), 
                                             lr=args.learning_rate,
                                             momentum=0.5)
    
    def forward(self, x, need_result=False):
        U = torch.zeros([x.shape[0], (x.shape[1]-1)*2, x.shape[3], x.shape[4]])
        for i in range(x.shape[1]-1):
            u1, u2, rho = self.flow_net(x[:,i,:,:,:], x[:,i+1,:,:,:])
            U[:, 2*i + 0, :, :] = u1.squeeze(0)
            U[:, 2*i + 1, :, :] = u2.squeeze(0)
        #--Spatial feature extraction
        #x = self.normalize_st(x[:,11//2,:,:,:])
        #Spatial_feature = self.fusion_spatial(x)
        
        #--Temporal feature extraction
        #U = self.normalize_tp(U)
        #Temporal_feature = self.fusion_temporal(U.cuda())
        
        if need_result:
            return torch.cat((x[:,11//2,:,:,:], U.cuda()), 1)
        
        
def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
            bias=False, batchnorm=False):
    if batchnorm:
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
    else:
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU())
    return layer

def encoder3d(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
            bias=False, batchnorm=False):
    if batchnorm:
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm3d(out_channels),
            nn.ReLU())
    else:
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU())
    return layer