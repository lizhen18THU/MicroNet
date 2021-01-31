import torch
import torch.nn as nn
import torch.nn.functional as F
import DyShiftMax as DyshiftMax
import numpy as np


class stemLayers(nn.Module):
    # midchannel是C/R
    def __init__(self, kernelSize, inchannel, outchannel, midchannel, G, stride, droprate):
        super(stemLayers, self).__init__()
        self.R = outchannel // midchannel
        # 3*1conv
        self.conv1 = nn.Conv2d(inchannel, midchannel, (kernelSize, 1), stride=(stride, 1),
                               padding=((kernelSize - 1) // 2, 0), groups=G[0], bias=False)
        self.bn1 = nn.BatchNorm2d(midchannel)
        # 1*3conv
        self.conv2 = nn.Conv2d(midchannel, outchannel, (1, kernelSize), stride=(1, stride),
                               padding=(0, (kernelSize - 1) // 2), groups=G[1], bias=False)
        self.bn2 = nn.BatchNorm2d(outchannel)
        self.relu = nn.ReLU(inplace=True)
        self.droprate = droprate

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        if self.droprate > 0:
            x = F.dropout(x, p=self.droprate, training=self.training)
        x = self.bn2(self.conv2(x))
        x = self.relu(x)
        return x


# 完成整数的分解，用于深度分离卷积扩展通道数时的数分解
def crack(integer):
    start = int(np.sqrt(integer))
    factor = integer / start
    while int(factor) != factor:
        start += 1
        factor = integer / start
    return int(factor), start


class channelShuffle(nn.Module):
    def __init__(self, groups):
        super(channelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        batchsize, channels, height, width = x.data.size()
        channels_per_group = int(channels / self.groups)

        x = x.view(batchsize, self.groups, channels_per_group, height, width)

        x = x.transpose(1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)

        return x


class MicroBlockA(nn.Module):
    def __init__(self, kernelSize, inchannel, outchannel, midchannel, G, stride, droprate):
        super(MicroBlockA, self).__init__()
        self.R = outchannel // midchannel
        expandsion_ratio = outchannel / inchannel
        assert expandsion_ratio == int(expandsion_ratio)
        r1, r2 = crack(expandsion_ratio)
        # macro-factorized depthwise conv expansion
        self.facdep_conv1 = nn.Conv2d(inchannel, inchannel * r1, (kernelSize, 1), stride=(stride, 1),
                                      padding=((kernelSize - 1) // 2, 0), groups=inchannel, bias=False)
        self.bn1 = nn.BatchNorm2d(inchannel * r1)
        self.facdep_conv2 = nn.Conv2d(inchannel * r1, inchannel * r1 * r2, (1, kernelSize), stride=(1, stride),
                                      padding=(0, (kernelSize - 1) // 2), groups=inchannel * r1, bias=False)
        self.bn2 = nn.BatchNorm2d(outchannel)
        self.dysmax1 = DyshiftMax.DyShiftMax(outchannel, inchannel, min(outchannel // 4, 16))

        # macro-factorized pointwise conv
        self.facpoint_conv = nn.Conv2d(outchannel, midchannel, kernel_size=1, groups=G[0], bias=False)
        self.point_chanShuffle = channelShuffle(G[0])
        self.bn3 = nn.BatchNorm2d(midchannel)
        self.dysmax2 = DyshiftMax.DyShiftMax(midchannel, G[0], min(midchannel // 4, 16))
        self.droprate = droprate

    def forward(self, x):
        x = self.bn1(self.facdep_conv1(x))
        x = self.bn2(self.facdep_conv2(x))
        x = self.dysmax1(x)
        if self.droprate > 0:
            x = F.dropout(x, p=self.droprate, training=self.training)
        x = self.bn3(self.point_chanShuffle(self.facpoint_conv(x)))
        x = self.dysmax2(x)
        return x


class MicroBlockB(nn.Module):
    def __init__(self, kernelSize, inchannel, outchannel, midchannel, G, stride, droprate):
        super(MicroBlockB, self).__init__()
        self.R = outchannel // midchannel
        expandsion_ratio = outchannel / inchannel
        assert expandsion_ratio == int(expandsion_ratio)
        r1, r2 = crack(expandsion_ratio)
        # macro-factorized depthwise conv expansion
        self.facdep_conv1 = nn.Conv2d(inchannel, inchannel * r1, (kernelSize, 1), stride=(stride, 1),
                                      padding=((kernelSize - 1) // 2, 0), groups=inchannel, bias=False)
        self.bn1 = nn.BatchNorm2d(inchannel * r1)
        self.facdep_conv2 = nn.Conv2d(inchannel * r1, inchannel * r1 * r2, (1, kernelSize), stride=(1, stride),
                                      padding=(0, (kernelSize - 1) // 2), groups=inchannel * r1, bias=False)
        self.bn2 = nn.BatchNorm2d(outchannel)
        self.dysmax1 = DyshiftMax.DyShiftMax(outchannel, inchannel, min(outchannel // 4, 16))

        # macro-factorized pointwise conv
        self.facpoint_conv1 = nn.Conv2d(outchannel, midchannel, kernel_size=1, groups=G[0], bias=False)
        self.point_chanShuffle = channelShuffle(G[0])
        self.bn3 = nn.BatchNorm2d(midchannel)
        self.dysmax2 = DyshiftMax.DyShiftMax(midchannel, G[0], min(midchannel // 4, 16))
        self.facpoint_conv2 = nn.Conv2d(midchannel, outchannel, kernel_size=1, groups=G[1], bias=False)
        self.bn4 = nn.BatchNorm2d(outchannel)
        self.dysmax3 = DyshiftMax.DyShiftMax(outchannel, G[1], min(outchannel // 4, 16))
        self.droprate = droprate

    def forward(self, x):
        x = self.bn1(self.facdep_conv1(x))
        x = self.bn2(self.facdep_conv2(x))
        x = self.dysmax1(x)
        if self.droprate > 0:
            x = F.dropout(x, p=self.droprate, training=self.training)
        x = self.bn3(self.point_chanShuffle(self.facpoint_conv1(x)))
        x = self.dysmax2(x)
        x = self.bn4(self.facpoint_conv2(x))
        x = self.dysmax3(x)
        return x


class MicroBlockC(nn.Module):
    def __init__(self, kernelSize, inchannel, outchannel, midchannel, G, stride, droprate):
        super(MicroBlockC, self).__init__()
        self.R = outchannel // midchannel
        self.inchannel = inchannel
        self.outchannel = outchannel
        # macro-factorized depthwise conv expansion
        self.facdep_conv1 = nn.Conv2d(inchannel, inchannel, (kernelSize, 1), stride=(stride, 1),
                                      padding=((kernelSize - 1) // 2, 0), groups=inchannel, bias=False)
        self.bn1 = nn.BatchNorm2d(inchannel)
        self.facdep_conv2 = nn.Conv2d(inchannel, inchannel, (1, kernelSize), stride=(1, stride),
                                      padding=(0, (kernelSize - 1) // 2), groups=inchannel, bias=False)
        self.bn2 = nn.BatchNorm2d(inchannel)
        self.dysmax1 = DyshiftMax.DyShiftMax(inchannel, inchannel, min(inchannel // 4, 16))

        # macro-factorized pointwise conv
        assert int(inchannel / G[0]) == inchannel / G[0]
        self.facpoint_conv1 = nn.Conv2d(inchannel, midchannel, kernel_size=1, groups=G[0], bias=False)
        self.point_chanShuffle = channelShuffle(G[0])
        self.bn3 = nn.BatchNorm2d(midchannel)
        self.dysmax2 = DyshiftMax.DyShiftMax(midchannel, G[0], min(midchannel // 4, 16))
        self.facpoint_conv2 = nn.Conv2d(midchannel, outchannel, kernel_size=1, groups=G[1], bias=False)
        self.bn4 = nn.BatchNorm2d(outchannel)
        self.dysmax3 = DyshiftMax.DyShiftMax(outchannel, G[1], min(outchannel // 4, 16))
        self.droprate = droprate

    def forward(self, x):
        out = self.bn1(self.facdep_conv1(x))
        out = self.bn2(self.facdep_conv2(out))
        out = self.dysmax1(out)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.bn3(self.point_chanShuffle(self.facpoint_conv1(out)))
        out = self.dysmax2(out)
        out = self.bn4(self.facpoint_conv2(out))
        out = self.dysmax3(out)

        if self.inchannel == self.outchannel and out.shape[2] == x.shape[2]:
            out += x

        return out
