from BasicBlock import *
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


# 关于BN和dropout加不加，怎么加，还有

class MicroNet_M0(nn.Module):
    def __init__(self, droprate, classNum=1000):
        super(MicroNet_M0, self).__init__()

        self.stem = stemLayers(kernelSize=3, inchannel=3, outchannel=6, midchannel=3, G=(1, 3), stride=2)
        self.stage1 = MicroBlockA(3, 6, 24, 8, (2, 0), 2)
        self.stage2 = MicroBlockA(3, 8, 32, 16, (4, 0), 2)
        self.stage3 = nn.Sequential(MicroBlockB(5, 16, 96, 16, (4, 4), 2), MicroBlockC(5, 96, 192, 32, (4, 8), 1))
        self.stage4 = nn.Sequential(MicroBlockC(5, 192, 384, 64, (8, 8), 2), MicroBlockC(3, 384, 576, 96, (8, 12), 1))
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.nchannels = 576
        self.fc1 = nn.Linear(576, 576 // 4)
        # 只在全连接层加上dropout
        self.droprate = droprate
        self.fc2 = nn.Linear(576 // 4, classNum)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        x = self.stem(x)
        x = self.stage2(self.stage1(x))
        x = self.stage4(self.stage3(x))
        x = self.avg_pooling(x)
        x = x.view(-1, self.nchannels)
        x = self.fc1(x)
        x = F.dropout(x, p=self.droprate, training=self.training)
        x = self.fc2(x)

        return x


class MicroNet_M1(nn.Module):
    def __init__(self, droprate, classNum=1000):
        super(MicroNet_M1, self).__init__()

        self.stem = stemLayers(kernelSize=3, inchannel=3, outchannel=8, midchannel=4, G=(1, 4), stride=2)
        self.stage1 = MicroBlockA(3, 8, 32, 12, (4, 0), 2)
        self.stage2 = nn.Sequential(MicroBlockA(3, 12, 48, 16, (4, 0), 2), MicroBlockB(3, 16, 144, 24, (4, 6), 1))
        self.stage3 = nn.Sequential(self._make_Layers(2, 144, 2, 5, 192, 32, (4, 8)),
                                    MicroBlockC(5, 192, 384, 64, (8, 8), 1))
        self.stage4 = nn.Sequential(MicroBlockC(5, 384, 576, 96, (8, 12), 2), MicroBlockC(3, 576, 768, 128, (8, 16), 1))
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.nchannels = 768
        self.fc1 = nn.Linear(768, 768 // 4)
        self.droprate = droprate
        self.fc2 = nn.Linear(768 // 4, classNum)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_Layers(self, numbers, inchannel, stride, kernelSize, outchannel, midchannel, G):
        layers = []
        layers.append(MicroBlockC(kernelSize, inchannel, outchannel, midchannel, G, stride))
        for i in range(1, numbers):
            layers.append((MicroBlockC(kernelSize, outchannel, outchannel, midchannel, G, 1)))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage2(self.stage1(x))
        x = self.stage4(self.stage3(x))
        x = self.avg_pooling(x)
        x = x.view(-1, self.nchannels)
        x = self.fc1(x)
        x = F.dropout(x, p=self.droprate, training=self.training)
        x = self.fc2(x)

        return x


class MicroNet_M2(nn.Module):
    def __init__(self, droprate, classNum=1000):
        super(MicroNet_M2, self).__init__()

        self.stem = stemLayers(kernelSize=3, inchannel=3, outchannel=12, midchannel=4, G=(1, 4), stride=2)
        self.stage1 = MicroBlockA(3, 12, 48, 16, (4, 0), 2)
        self.stage2 = nn.Sequential(MicroBlockA(3, 16, 64, 24, (4, 0), 2), MicroBlockB(3, 24, 144, 24, (4, 6), 1))
        self.stage3 = nn.Sequential(self._make_Layers(2, 144, 2, 5, 192, 32, (4, 8)),
                                    MicroBlockC(5, 192, 288, 48, (6, 8), 1),
                                    self._make_Layers(2, 288, 1, 5, 480, 80, (8, 10)))
        self.stage4 = nn.Sequential(MicroBlockC(5, 480, 720, 120, (10, 12), 2),
                                    MicroBlockC(3, 720, 720, 120, (10, 12), 1),
                                    MicroBlockC(3, 720, 864, 144, (12, 12), 1))
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.nchannels = 864
        self.fc1 = nn.Linear(864, 864 // 4)
        self.droprate = droprate
        self.fc2 = nn.Linear(864 // 4, classNum)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_Layers(self, numbers, inchannel, stride, kernelSize, outchannel, midchannel, G):
        layers = []
        layers.append(MicroBlockC(kernelSize, inchannel, outchannel, midchannel, G, stride))
        for i in range(1, numbers):
            layers.append((MicroBlockC(kernelSize, outchannel, outchannel, midchannel, G, 1)))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage2(self.stage1(x))
        x = self.stage4(self.stage3(x))
        x = self.avg_pooling(x)
        x = x.view(-1, self.nchannels)
        x = self.fc1(x)
        x = F.dropout(x, p=self.droprate, training=self.training)
        x = self.fc2(x)

        return x


class MicroNet_M3(nn.Module):
    def __init__(self, droprate, classNum=1000):
        super(MicroNet_M3, self).__init__()

        self.stem = stemLayers(kernelSize=3, inchannel=3, outchannel=16, midchannel=4, G=(1, 4), stride=2)
        self.stage1 = MicroBlockA(3, 16, 64, 32, (4, 0), 2)
        self.stage2 = nn.Sequential(MicroBlockB(3, 32, 192, 48, (6, 8), 2),
                                    self._make_Layers(2, 192, 1, 3, 192, 48, (6, 8)))
        self.stage3 = nn.Sequential(self._make_Layers(2, 192, 2, 5, 256, 64, (8, 8)),
                                    self._make_Layers(2, 256, 1, 5, 384, 96, (8, 12)),
                                    self._make_Layers(2, 384, 1, 5, 576, 144, (12, 12)))
        self.stage4 = nn.Sequential(self._make_Layers(2, 576, 2, 5, 768, 192, (12, 16)),
                                    MicroBlockC(5, 768, 768, 192, (12, 16), 1),
                                    MicroBlockC(3, 768, 1024, 256, (16, 16), 1))
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.nchannels = 1024
        self.fc1 = nn.Linear(1024, 1024 // 4)
        self.droprate = droprate
        self.fc2 = nn.Linear(1024 // 4, classNum)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_Layers(self, numbers, inchannel, stride, kernelSize, outchannel, midchannel, G):
        layers = []
        layers.append(MicroBlockC(kernelSize, inchannel, outchannel, midchannel, G, stride))
        for i in range(1, numbers):
            layers.append((MicroBlockC(kernelSize, outchannel, outchannel, midchannel, G, 1)))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage2(self.stage1(x))
        x = self.stage4(self.stage3(x))
        x = self.avg_pooling(x)
        x = x.view(-1, self.nchannels)
        x = self.fc1(x)
        x = F.dropout(x, p=self.droprate, training=self.training)
        x = self.fc2(x)

        return x


def M0_Net(droprate=0.05):
    return MicroNet_M0(droprate=droprate)


def M1_Net(droprate=0.05):
    return MicroNet_M1(droprate=droprate)


def M2_Net(droprate=0.1):
    return MicroNet_M2(droprate=droprate)


def M3_Net(droprate=0.1):
    return MicroNet_M3(droprate=droprate)
