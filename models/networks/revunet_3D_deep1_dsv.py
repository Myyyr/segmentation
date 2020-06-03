import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import revtorch.revtorch as rv
import random
from .utils import UnetDsv3

id = random.getrandbits(64)

#restore experiment
#VALIDATE_ALL = False
#PREDICT = True
#RESTORE_ID = 7420189804603519207
#RESTORE_EPOCH = 6
#LOG_COMETML_EXISTING_EXPERIMENT = ""

#general settings
SAVE_CHECKPOINTS = False #set to true to create a checkpoint at every epoch
EXPERIMENT_TAGS = ["bugfreeFinalDrop"]
EXPERIMENT_NAME = "Reversible NO_NEW60, dropout"
EPOCHS = 1000
BATCH_SIZE = 1
VIRTUAL_BATCHSIZE = 1
VALIDATE_EVERY_K_EPOCHS = 1
INPLACE = True

#hyperparameters
#CHANNELS = [36, 72, 144, 288, 576] #normal doubling strategy
# CHANNELS = [60, 120, 240, 360, 480]
CHANNELS = [16, 32, 64, 128, 256, 512, 1024]
#CHANNELS = [int(x / 4) for x in CHANNELS]
INITIAL_LR = 1e-4
L2_REGULARIZER = 1e-5

#logging settings
LOG_EVERY_K_ITERATIONS = 5 #0 to disable logging
LOG_MEMORY_EVERY_K_ITERATIONS = False
LOG_MEMORY_EVERY_EPOCH = True
LOG_EPOCH_TIME = True
LOG_VALIDATION_TIME = True
LOG_HAUSDORFF_EVERY_K_EPOCHS = 0 #must be a multiple of VALIDATE_EVERY_K_EPOCHS
LOG_COMETML = False
LOG_PARAMCOUNT = True
LOG_LR_EVERY_EPOCH = True

#data and augmentation
TRAIN_ORIGINAL_CLASSES = False #train on original 5 classes
DATASET_WORKERS = 1
SOFT_AUGMENTATION = False #Soft augmetation directly works on the 3 classes. Hard augmentation augments on the 5 orignal labels, then takes the argmax
NN_AUGMENTATION = True #Has priority over soft/hard augmentation. Uses nearest-neighbor interpolation
DO_ROTATE = True
DO_SCALE = True
DO_FLIP = True
DO_ELASTIC_AUG = True
DO_INTENSITY_SHIFT = True
#RANDOM_CROP = [128, 128, 128]

ROT_DEGREES = 20
SCALE_FACTOR = 1.1
SIGMA = 10
MAX_INTENSITY_SHIFT = 0.1



class ResidualInner(nn.Module):
    def __init__(self, channels, groups):
        super(ResidualInner, self).__init__()
        self.gn1 = nn.BatchNorm3d(channels)
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)

        self.gn2 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x):
        x = F.leaky_relu(self.gn1(self.conv1(x)), inplace=INPLACE)
        x = F.leaky_relu(self.gn2(self.conv2(x)), inplace=INPLACE)
        return x

def makeReversibleSequence(channels):
    innerChannels = channels // 2
    groups = CHANNELS[0] // 2
    fBlock = ResidualInner(innerChannels, groups)
    gBlock = ResidualInner(innerChannels, groups)
    #gBlock = nn.Sequential()
    return rv.ReversibleBlock(fBlock, gBlock)

def makeReversibleComponent(channels, blockCount):
    modules = []
    for i in range(blockCount):
        modules.append(makeReversibleSequence(channels))
    return rv.ReversibleSequence(nn.ModuleList(modules))

def getChannelsAtIndex(index):
    if index < 0: index = 0
    if index >= len(CHANNELS): index = len(CHANNELS) - 1
    return CHANNELS[index]

class EncoderModule(nn.Module):
    def __init__(self, inChannels, outChannels, depth, downsample=True):
        super(EncoderModule, self).__init__()
        self.downsample = downsample
        if downsample:
            self.conv = nn.Conv3d(inChannels, outChannels, 1)
        self.reversibleBlocks = makeReversibleComponent(outChannels, depth)

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

    def forward(self, x):
        x = self.reversibleBlocks(x)
        if self.upsample:
            x = self.conv(x)
            x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
        return x

class NoNewReversible_deep_dsv(nn.Module):
    def __init__(self):
        super(NoNewReversible_deep_dsv, self).__init__()
        depth = 1
        self.levels = 6
        n_classes = 2

        self.firstConv = nn.Conv3d(1, CHANNELS[0], 3, padding=1, bias=False)
        #self.dropout = nn.Dropout3d(0.2, True)
        self.lastConv = nn.Conv3d(n_classes*(self.levels - 1), n_classes, 1, bias=True)

        #create encoder levels
        encoderModules = []
        for i in range(self.levels):
            encoderModules.append(EncoderModule(getChannelsAtIndex(i - 1), getChannelsAtIndex(i), depth, i != 0))
        self.encoders = nn.ModuleList(encoderModules)

        #create decoder levels
        decoderModules = []
        for i in range(self.levels):
            decoderModules.append(DecoderModule(getChannelsAtIndex(self.levels - i - 1), getChannelsAtIndex(self.levels - i - 2), depth, i != (self.levels -1)))
        self.decoders = nn.ModuleList(decoderModules)

        self.dsv5 = UnetDsv3(in_size=CHANNELS[4], out_size=n_classes, scale_factor=16)
        self.dsv4 = UnetDsv3(in_size=CHANNELS[3], out_size=n_classes, scale_factor=8)
        self.dsv3 = UnetDsv3(in_size=CHANNELS[2], out_size=n_classes, scale_factor=4)
        self.dsv2 = UnetDsv3(in_size=CHANNELS[1], out_size=n_classes, scale_factor=2)
        self.dsv1 = nn.Conv3d(in_channels=CHANNELS[0], out_channels=n_classes, kernel_size=1)

    def forward(self, x):
        x = self.firstConv(x)
        #x = self.dropout(x)

        inputStack = []
        for i in range(self.levels):
            x = self.encoders[i](x)
            if i < self.levels - 1:
                inputStack.append(x)

        up = []
        for i in range(self.levels):
            x = self.decoders[i](x)
            if i < self.levels - 1:
                up.append(x)
                x = x + inputStack.pop()

        dsv5 = self.dsv5(up[0])
        dsv4 = self.dsv4(up[1])
        dsv3 = self.dsv3(up[2])
        dsv2 = self.dsv2(up[3])
        dsv1 = self.dsv1(up[4])


        x = self.lastConv(torch.cat([dsv1,dsv2,dsv3,dsv4,dsv5], dim=1))
        #x = torch.sigmoid(x)
        return x

    @staticmethod
    def apply_argmax_softmax(pred):
        pred = F.softmax(pred, dim=1)
        return pred