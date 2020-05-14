import sys
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import revtorch.revtorch as rv
import random
from models.networks_other import init_weights

import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'





id = random.getrandbits(64)



class ResidualInner(nn.Module):
    """F or G function module
    Extends:
        nn.Module
    """
    def __init__(self, channels):
        """
        Arguments:
            channels {int} -- Number of in and out channels (in_chan = out_chan)
        """
        super(ResidualInner, self).__init__()
        self.gn1 = nn.BatchNorm3d(channels)
        self.gn2 = nn.BatchNorm3d(channels)
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.gn1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self.gn2(x)
        x = F.relu(x)
        
        return x

def makeReversibleSequence(channels):
    """Create a reversible block
    Arguments:
        channels {int} -- Number of in and out channels (in_chan = out_chan)
    
    Returns:
        [nn.Module] -- reversible block
    """
    innerChannels = channels // 2
    fBlock = ResidualInner(innerChannels)
    gBlock = ResidualInner(innerChannels)
    return rv.ReversibleBlock(fBlock, gBlock)

def makeReversibleComponent(channels, blockCount):
    """Create a reversible sequence of blockCount blocks
    Arguments:
        channels {int} -- Number of in and out channels (in_chan = out_chan)
        blockCount {int} -- Number of blocks in the sequence
    
    Returns:
        [nn.Module] -- Reversible sequence
    """
    modules = []
    for i in range(blockCount):
        modules.append(makeReversibleSequence(channels))
    return rv.ReversibleSequence(nn.ModuleList(modules))


class DownSampler(nn.Module):
    """Down Sample module
    
    Decrease the spatial size and increase the channel size of tensor
    
    Extends:
        nn.Module
    """
    def __init__(self, inChannels, outChannels):
        """
        Arguments:
            inChannels {int} -- Number of in channels
            outChannels {int} -- Number of out channels
        """
        super(DownSampler, self).__init__()

        self.inChannels = inChannels

        # add_chan = out_chan - in_chan            
        # self.pad_front, self.pad_back = [add_chan//2]*2
        self.conv = nn.Conv3d(inChannels, outChannels, 1)
        self.maxpool = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.bn = nn.BatchNorm3d(outChannels)


        # if inChannels%2 != 0:
        #     self.pad_front += 1
        
    def forward(self, x):
        x = self.maxpool(x) # Reduce the spatial size
        x = self.conv(x) # increase the number of channels
        x = F.relu(x)
        x = self.bn(x)

        # pad = 1*(np.array(x.shape )% 2 != 0)
        #x = F.pad(x, (pad[4],0,pad[3],0,pad[2],0)) # add pad of size 1 in the non-divisible by 2 dimension

        # x = x.permute((0,2,1,3,4))
        # x = F.pad(x, (0,0,0,0,self.pad_front,self.pad_back))
        # x = x.permute((0,2,1,3,4))
        
        return x

class UpSampler(nn.Module):
    """Up Sample module
    
    Decrease the channels size and increase the spatial size of tensor
    
    Extends:
        nn.Module
    """
    def __init__(self, inChannels, outChannels, spatial_size):
        """
        Arguments:
            inChannels {int} -- Number of in channels
            outChannels {int} -- Number of out channels
            spatial_size {tuple} -- Spatial size to get
        """
        super(UpSampler, self).__init__()

        self.spatial_size = spatial_size


        self.conv = nn.Conv3d(inChannels, outChannels, 1)
        # self.bn = nn.BatchNorm3d(outChannels)
        
    def forward(self, x):
        x = self.conv(x) # decrease the number of channels
        # x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False) # increase spatial size
        x = F.interpolate(x, self.spatial_size, mode="trilinear", align_corners=False) # increase spatial size
        return x

class revunet_3D(nn.Module):
    """Partialy reversible UNet 3D module
    
    Extends:
        nn.Module

    Arguments:
        channels {int} -- List of number of channels at each level
        depth {int} -- Number of block in a reversible sequence
        n_class {int} -- Number of classes to predict
        spatial_sizes {list} -- Spatial sizes
    """
    #def __init__(self, channels, depth, n_class, spatial_sizes):
    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True, in_channels=3, is_batchnorm=True):
        """
        Arguments:
            channels {int} -- List of number of channels at each level
            depth {int} -- Number of block in a reversible sequence
            n_class {int} -- Number of classes to predict
            spatial_sizes {list} -- Spatial sizes
        """
        super(revunet_3D, self).__init__()


        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        channels = [64, 128, 256, 512, 1024]
        channels = [int(x / self.feature_scale) for x in channels]
        self.depth = 1


        images_dim = [160,160,96]
        tmp = [1] + images_dim
        spatial_sizes = [(tmp[1]   ,
                          tmp[2]   ,
                          tmp[3]    ) ]

        for i in range(len(channels)-1):
            spatial_sizes.append(   (spatial_sizes[i][0]//(2)    ,
                                     spatial_sizes[i][1]//(2)    ,
                                     spatial_sizes[i][2]//(2)   ) 
                                )



        

        self.firstConv = nn.Conv3d(self.in_channels, channels[0], 3, padding=1, bias=False)
        self.lastConv = nn.Conv3d(channels[0], n_classes, 1, bias=True)

        self.conv1 = makeReversibleComponent(channels[0], self.depth)
        self.down1 = DownSampler(channels[0], channels[1])

        self.conv2 = makeReversibleComponent(channels[1], self.depth)
        self.down2 = DownSampler(channels[1], channels[2])

        self.conv3 = makeReversibleComponent(channels[2], self.depth)
        self.down3 = DownSampler(channels[2], channels[3])

        self.conv4 = makeReversibleComponent(channels[3], self.depth)
        self.down4 = DownSampler(channels[3], channels[4])

        self.center = makeReversibleComponent(channels[4], self.depth)

        self.up1    = UpSampler(channels[4], channels[3], spatial_sizes[3])
        self.dconv1 = makeReversibleComponent(channels[3], self.depth)

        self.up2    = UpSampler(channels[3], channels[2], spatial_sizes[2])
        self.dconv2 = makeReversibleComponent(channels[2], self.depth)

        self.up3    = UpSampler(channels[2], channels[1], spatial_sizes[1])
        self.dconv3 = makeReversibleComponent(channels[1], self.depth)

        self.up4    = UpSampler(channels[1], channels[0], spatial_sizes[0])
        self.dconv4 = makeReversibleComponent(channels[0], self.depth)
        

        #create encoder levels
        # encoderModules = []
        # self.skip_index = []
        # idx = 0
        # for i in range(self.levels):
        #     if i != 0:
        #         encoderModules.append(DownSampler(channels[i-1], channels[i]))
        #         idx+=1
        #     encoderModules.append(makeReversibleComponent(channels[i], self.depth))
        #     if i != self.levels-1:
        #         self.skip_index.append(idx)
        #     idx+=1
        # self.encoders = nn.ModuleList(encoderModules)

        # #create decoder levels
        # decoderModules = []
        
        # for i in range(self.levels):
        #     decoderModules.append(makeReversibleComponent(channels[self.levels - 1 - i], self.depth))
        #     if i != self.levels - 1:
        #         decoderModules.append(UpSampler(channels[self.levels - 1 - i], channels[self.levels -2 - i], spatial_sizes[self.levels - i - 2]))
        # self.decoders = nn.ModuleList(decoderModules)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, x):
        x = self.firstConv(x)

        x1 = self.conv1(x)
        del x
        x2 = self.down1(x1)

        x2 = self.conv2(x2)
        x3 = self.down2(x2)

        x3 = self.conv3(x3)
        x4 = self.down3(x3)

        x4 = self.conv4(x4)
        c  = self.down4(x4)

        c  = self.center(c)

        y = self.up1(c)
        del c
        y = self.dconv1(x4 + y)
        del x4

        y = self.up2(y)
        y = self.dconv2(x3 + y)
        del x3

        y = self.up3(y)
        y = self.dconv3(x2 + y)
        del x2

        y = self.up4(y)
        y = self.dconv4(x1 + y)
        del x1

        y = self.lastConv(y)




        # inputStack = []
        # for i in range(len(self.encoders)):
        #     x = self.encoders[i](x)

        #     if i in self.skip_index:
        #         inputStack.append(x)

        # for i in range(len(self.encoders)):
        #     x = self.decoders[i](x)
        #     if i in [sk + 1 for sk in self.skip_index]:
        #         x = x + inputStack.pop()

        # x = self.lastConv(x)
        # x = F.softmax(x, dim=1)

        # x = nn.Sigmoid()(x)

        return y

    @staticmethod
    def apply_argmax_softmax(pred):
        # log_p = F.softmax(pred, dim=1)



        return pred