import math
import torch.nn as nn
from models.networks.utils import UnetConv3, UnetUp3, UnetUp3_CT
import torch.nn.functional as F
from models.networks_other import init_weights
import torch

class unet_3D(nn.Module):

    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True, in_channels=3, is_batchnorm=True, im_dim = None):
        super(unet_3D, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        self.im_dim = im_dim

        # downsampling
        self.conv1 = UnetConv3(self.in_channels, filters[0], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2 = UnetConv3(filters[0], filters[1], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = UnetConv3(filters[1], filters[2], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = UnetConv3(filters[2], filters[3], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center = UnetConv3(filters[3], filters[4], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))

        # upsampling
        self.up_concat4 = UnetUp3_CT(filters[4], filters[3], is_batchnorm)
        self.up_concat3 = UnetUp3_CT(filters[3], filters[2], is_batchnorm)
        self.up_concat2 = UnetUp3_CT(filters[2], filters[1], is_batchnorm)
        self.up_concat1 = UnetUp3_CT(filters[1], filters[0], is_batchnorm)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], n_classes, 1)

        # interpolation
        self.interpolation = nn.Upsample(size = (512,512,256), mode = "trilinear")


        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, X):
        # print("||start|| memory :",convert_bytes(torch.cuda.max_memory_allocated()))
        # print("||start|| cur memory :", convert_bytes(torch.cuda.memory_allocated()))
        # print("|||| X size :", convert_bytes(X.element_size() * X.nelement()))
        # if self.im_dim != None:
        #     with torch.no_grad():
        #         # print("|||| INPUT SHAPE", inputs.shape)
        #         inputs = nn.functional.interpolate(X, self.im_dim, mode='trilinear')
                # print("|||| INPUT SHAPE", inputs.shape)

        # print("||interpolate|| memory :",convert_bytes(torch.cuda.max_memory_allocated()))
        # print("||interpolate|| cur memory :", convert_bytes(torch.cuda.memory_allocated()))
        # print("|||| inputs size :", convert_bytes(inputs.element_size() * inputs.nelement()))
        # del X
        # print("||del|| memory :",convert_bytes(torch.cuda.max_memory_allocated()))
        # print("||del|| cur memory :", convert_bytes(torch.cuda.memory_allocated()))

        conv1 = self.conv1(X)
        del X
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)
# 
        # print("||down/up|| memory :",convert_bytes(torch.cuda.max_memory_allocated()))
        # print("||down/up|| cur memory :", convert_bytes(torch.cuda.memory_allocated()))

        del maxpool1, maxpool2, maxpool3, maxpool4, center
        del conv1,conv2,conv3,conv4
        del up4, up3, up2
        # print("||del maxpool|| memory :",convert_bytes(torch.cuda.max_memory_allocated()))
        # print("||del maxpool|| cur memory :", convert_bytes(torch.cuda.memory_allocated()))

        Y = self.final(up1)
        del up1
        # print("||final|| memory :",convert_bytes(torch.cuda.max_memory_allocated()))
        # print("||final|| cur memory :", convert_bytes(torch.cuda.memory_allocated()))
        final = self.interpolation(Y)
        del Y
        # print("||interpolation|| memory :",convert_bytes(torch.cuda.max_memory_allocated()))
        # print("||interpolation|| cur memory :", convert_bytes(torch.cuda.memory_allocated()))
        # exit(0)
        return final

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p



def convert_bytes(size):
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return "%3.2f %s" % (size, x)
        size /= 1024.0

    return size












