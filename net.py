import torch.nn as nn
import torch.nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils import weight_norm


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


class Channel_Attention(nn.Module):

    def __init__(self, channel, r):
        super(Channel_Attention, self).__init__()

        self.__avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.__max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.__fc = nn.Sequential(
            nn.Conv2d(channel, channel//r, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(channel//r, channel, 1, bias=False),
        )
        self.__sigmoid = nn.Sigmoid()

    def forward(self, x):
        y1 = self.__avg_pool(x)
        y1 = self.__fc(y1)

        y2 = self.__max_pool(x)
        y2 = self.__fc(y2)

        y = self.__sigmoid(y1+y2)
        return x * y

class Spartial_Attention(nn.Module):

    def __init__(self, kernel_size):
        super(Spartial_Attention, self).__init__()

        assert kernel_size % 2 == 1, "kernel_size = {}".format(kernel_size)
        padding = (kernel_size - 1) // 2

        self.__layer = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding),
            nn.Sigmoid(),
        )

    def forward(self, x):
        avg_mask = torch.mean(x, dim=1, keepdim=True)
        max_mask, _ = torch.max(x, dim=1, keepdim=True)
        mask = torch.cat([avg_mask, max_mask], dim=1)

        mask = self.__layer(mask)
        return x * mask

#WIDE Multi-scale Residual Block
class WRN(nn.Module):
    def __init__(self,nin=32,use_GPU=True):
        super(WRN, self).__init__()
        self.use_GPU = use_GPU
        ksize1 = 3
        ksize2 = 5
        pad1=int((ksize1-1)/2)
        pad2=int((ksize2-1)/2)
        self.conv1 = nn.Conv2d(nin, nin, 1, 1)
        self.conv2 = nn.Conv2d(nin, nin, ksize1, 1,pad1)
        self.conv3 = nn.Conv2d(nin, nin, ksize2, 1,pad2)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, input):
        x = input
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x))
        x3 = self.lrelu(self.conv3(x))

        y1 = self.conv1(x1 + x2 + x3)
        y2 = self.conv2(x1 + x2 + x3)
        y3 = self.conv3(x1 + x2 + x3)

        y1 = self.lrelu(y1)
        y2 = self.lrelu(y2)
        y2 = self.lrelu(y2)

        out = self.conv1(y1 + y2 + y3)
        out = input+out
        return out

#Multi-scale residual block
class MRN(nn.Module):
    def __init__(self,nin=32,use_GPU=True):
        super(MRN, self).__init__()
        self.use_GPU = use_GPU
        ksize1 = 3
        ksize2 = 5
        pad1=int((ksize1-1)/2)
        pad2=int((ksize2-1)/2)
        self.conv1 = nn.Conv2d(nin, nin, ksize1, 1,pad1)
        self.conv2 = nn.Conv2d(nin, nin, ksize2, 1,pad2)
        self.conv3 = nn.Conv2d(nin, nin, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, input):
        x = input
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x))

        y1 = self.conv1(x1 + x2 )
        y2 = self.conv2(x1 + x2 )

        y1 = self.lrelu(y1)
        y2 = self.lrelu(y2)

        out = self.conv3(y1 + y2 )
        out = input+out
        return out

#Long Multi-scale Residual Block
class LRN(nn.Module):
    def __init__(self,nin=32,use_GPU=True):
        super(LRN, self).__init__()
        self.use_GPU = use_GPU
        ksize1 = 3
        ksize2 = 5
        pad1=int((ksize1-1)/2)
        pad2=int((ksize2-1)/2)
        self.conv1 = nn.Conv2d(nin, nin, 1, 1)
        self.conv2 = nn.Conv2d(nin, nin, ksize1, 1,pad1)
        self.conv3 = nn.Conv2d(nin, nin, ksize2, 1,pad2)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, input):
        x = input
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x))

        y1 = self.conv2(x1 + x2)
        y2 = self.conv3(x1 + x2)

        z1 = self.conv3(y1 + y2)
        z2 = self.conv1(y1 + y2)

        z1 = self.lrelu(z1)
        z2 = self.lrelu(z2)

        out = self.conv1(z1 + z2)
        out = input+out
        return out

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class Resblock(nn.Module):
    def __init__(self, n_feats=32):
        super(Resblock, self).__init__()
        self.PR = nn.PReLU()
        self.conv = nn.Conv2d(n_feats, n_feats, 3, padding=1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(n_feats, n_feats, 3, padding=1, stride=1, bias=False)

    def forward(self, x):
        res = x
        x1 = self.PR(self.conv(x))
        # cat = torch.cat([res, x1], 1)
        x2 = self.conv2(x1)
        out = self.PR(x2 + res)
        return out


class Fuse(nn.Module):

    def __init__(self, n_feats=32):
        super(Fuse, self).__init__()

        self.PR = nn.PReLU()
        self.conv1 = nn.Sequential(nn.Conv2d(n_feats, n_feats, 3, padding=1, dilation=1))
        self.conv3 = nn.Sequential(nn.Conv2d(n_feats, n_feats, 3, padding=3, dilation=3))
        self.conv5 = nn.Sequential(nn.Conv2d(n_feats, n_feats, 3, padding=5, dilation=5))
        self.fuse = nn.Sequential(eca_layer(channel=32))

    def forward(self, x):
        residual = x
        x1 = self.conv1(x)
        x2 = self.conv3(x)
        x3 = self.conv5(x)
        x4 = x1 + x2 + x3
        x5 = self.fuse(x4)
        out = residual + x5
        output = self.PR(out)
        return output


class eca_layer(nn.Module):

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class NET(nn.Module):
    def __init__(self, input_channel=32, use_GPU=True):
        super(NET, self).__init__()
        ksize = 3
        self.use_GPU = use_GPU
        self.PR = nn.PReLU()
        self.fuse2 = eca_layer(channel=32)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, input_channel, ksize, padding=ksize // 2, stride=1, bias=False),
            nn.PReLU(),
        )

        # WIDE Multi-scale Residual Block
        self.block1 = nn.Sequential(WRN(input_channel),
                                    nn.PReLU())
        # Long Multi-scale Residual Block
        self.block2 = nn.Sequential(LRN(input_channel),
                                    nn.PReLU())
        # Multi-scale Residual Block
        self.block3 = nn.Sequential(MRN(input_channel),
                                    nn.PReLU())
        self.ca = Channel_Attention(32, 16)
        self.sa = Spartial_Attention(5)
        self.RB1 = nn.Sequential(
            Resblock(n_feats=32),
            Resblock(n_feats=32),
            Resblock(n_feats=32),
            Fuse(n_feats=32)

        )
        self.RB2 = nn.Sequential(
            Resblock(n_feats=32),
            Resblock(n_feats=32),
            Resblock(n_feats=32),
            Fuse(n_feats=32),
        )
        self.RB3 = nn.Sequential(
            Resblock(n_feats=32),
            Resblock(n_feats=32),
            Resblock(n_feats=32),
            Fuse(n_feats=32),
        )
        self.RB4 = nn.Sequential(
            Resblock(n_feats=32),
            Resblock(n_feats=32),
            Resblock(n_feats=32),
            Fuse(n_feats=32),
        )
        self.RB5 = nn.Sequential(
            Resblock(n_feats=32),
            Resblock(n_feats=32),
            Resblock(n_feats=32),
            Fuse(n_feats=32),
        )
        self.RB6 = nn.Sequential(
            Resblock(n_feats=32),
            Resblock(n_feats=32),
            Resblock(n_feats=32),
            Fuse(n_feats=32),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(input_channel, 3, ksize, padding=ksize // 2, stride=1, bias=False)
        )

    def forward(self, input):
        #the first
        x1 = self.conv1(input)
        x2 = self.conv1(input)
        x3 = self.conv1(input)
        for j in range(2):
            x11 = x1
            for i in range(3):
                x1 = self.block1(x1)
                x2 = self.block2(x2)
                x3 = self.block3(x3)
            x1 = x11 + x1 + x2 + x3
        x1 = self.ca(x1)
        x1 = self.sa(x1)
        x1 = self.conv2(x1)
        out1 = x1

        #the second
        x4 = self.conv1(input)
        x4 = self.RB1(x4)
        x4 = self.RB2(x4)
        x4 = self.RB3(x4)
        x4 = self.RB4(x4)
        x4 = self.RB5(x4)
        x4 = self.RB6(x4)
        x4 = self.PR(self.fuse2(x4))
        x4 = self.conv2(x4)
        out2 = x4

        # residual overlay
        out = out1 + out2
        return out
