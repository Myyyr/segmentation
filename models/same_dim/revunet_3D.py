import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import revtorch.revtorch as rv
import random
from models.networks_other import init_weights


class ResidualInner(nn.Module):
    def __init__(self, channels, groups):
        super(ResidualInner, self).__init__()
        # self.gn = nn.BatchNorm3d(channels)
        self.gn = nn.GroupNorm(groups, channels)
        self.conv = nn.Conv3d(channels, channels, 3, padding=1, bias=False)

        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, x):
        x = F.leaky_relu(self.gn(self.conv(x)), inplace=True)
        return x

# class ResidualInner(nn.Module):
#     def __init__(self, channels, groups):
#         super(ResidualInner, self).__init__()
#         # self.gn = nn.BatchNorm3d(channels)
#         self.conv = nn.Conv3d(channels, channels, 3, padding=1, bias=False)

#     def forward(self, x):
#         x = F.relu(self.conv(x), inplace=True)
#         return x

def makeReversibleSequence(channels):
    innerChannels = channels // 2
    groups = 2#CHANNELS[0] // 2
    fBlock = ResidualInner(innerChannels, groups)
    gBlock = ResidualInner(innerChannels, groups)
    #gBlock = nn.Sequential()
    return rv.ReversibleBlock(fBlock, gBlock)

def makeReversibleComponent(channels, blockCount):
    modules = []
    for i in range(blockCount):
        modules.append(makeReversibleSequence(channels))
    return rv.ReversibleSequence(nn.ModuleList(modules))

def getChannelsAtIndex(index, channels):
    if index < 0: index = 0
    if index >= len(channels): index = len(channels) - 1
    return channels[index]

class EncoderModule(nn.Module):
    def __init__(self, inChannels, outChannels, depth, downsample=True):
        super(EncoderModule, self).__init__()
        self.downsample = downsample
        if downsample:
            self.conv = nn.Conv3d(inChannels, outChannels, 1)
        self.reversibleBlocks = makeReversibleComponent(outChannels, depth)
        for m in self.children():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
    def forward(self, x):
        if self.downsample:
            x = F.max_pool3d(x, 2)
            x = self.conv(x) #increase number of channels
        x = self.reversibleBlocks(x)
        return x

class DecoderModule(nn.Module):
    def __init__(self, inChannels, outChannels, depth, upsample=True):
        super(DecoderModule, self).__init__()
        self.reversibleBlocks = makeReversibleComponent(inChannels, depth)
        self.upsample = upsample
        if self.upsample:
            self.conv = nn.Conv3d(inChannels, outChannels, 1)
        for m in self.children():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
                
    def forward(self, x):
        x = self.reversibleBlocks(x)
        if self.upsample:
            x = self.conv(x)
            x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
        return x

class RevUnet3D(nn.Module):
    def __init__(self, inchannels = 1, channels = [4*16, 8*16, 16*16, 32*16, 64*16], out_size = 2, depth = 1, interpolation = None):
        super(RevUnet3D, self).__init__()
        self.levels = len(channels)
        channels = [64, 128, 256, 512, 1024]

        self.firstConv = nn.Conv3d(inchannels, channels[0], 3, padding=1, bias=False)
        #self.dropout = nn.Dropout3d(0.2, True)
        self.lastConv = nn.Conv3d(channels[0], out_size, 1)#, bias=True)

        #create encoder levels
        encoderModules = []
        for i in range(self.levels):
            encoderModules.append(EncoderModule(getChannelsAtIndex(i - 1, channels), getChannelsAtIndex(i, channels), depth, i != 0))
        self.encoders = nn.ModuleList(encoderModules)

        #create decoder levels
        decoderModules = []
        for i in range(self.levels):
            decoderModules.append(DecoderModule(getChannelsAtIndex(self.levels - i - 1, channels), getChannelsAtIndex(self.levels - i - 2, channels), depth, i != (self.levels -1)))
        self.decoders = nn.ModuleList(decoderModules)


        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            


    def forward(self, x):
        tibo_in_shape = x.shape[-3:]
        x = self.firstConv(x)
        #x = self.dropout(x)

        inputStack = []
        for i in range(self.levels):
            x = self.encoders[i](x)
            if i < self.levels - 1:
                inputStack.append(x)

        for i in range(self.levels):
            x = self.decoders[i](x)
            if i < self.levels - 1:
                x = x + inputStack.pop()

        x = self.lastConv(x)

        # x = F.softmax(x, dim=1)
        return x

    @staticmethod
    def apply_argmax_softmax(pred):
        pred = F.softmax(pred, dim=1)
        return pred