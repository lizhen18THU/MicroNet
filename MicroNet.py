from BasicBlock import *
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class MicroNet_M0(nn.Module):
    def __init__(self, droprate, droprate_fc, classNum):
        super(MicroNet_M0, self).__init__()

        self.stem = stemLayers(kernelSize=3, inchannel=3, outchannel=6, midchannel=3, G=(1, 3), stride=2,
                               droprate=droprate)
        self.stage1 = MicroBlockA(3, 6, 24, 8, (2, 0), 2, droprate)
        self.stage2 = MicroBlockA(3, 8, 32, 16, (4, 0), 2, droprate)
        self.stage3 = nn.Sequential(MicroBlockB(5, 16, 96, 16, (4, 4), 2, droprate),
                                    MicroBlockC(5, 96, 192, 32, (4, 8), 1, droprate))
        self.stage4 = nn.Sequential(MicroBlockC(5, 192, 384, 64, (8, 8), 2, droprate),
                                    MicroBlockC(3, 384, 576, 96, (8, 12), 1, droprate))
        self.avg_pooling = nn.AvgPool2d(7)
        self.nchannels = 576

        # 修改全连接层的规模大小以满足FLOPs的要求，
        self.fc1 = nn.Linear(576, int(576 // 0.5))
        self.droprate_fc = droprate_fc
        self.fc2 = nn.Linear(int(576 // 0.5), classNum)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        x = self.stem(x)
        x = self.stage2(self.stage1(x))
        x = self.stage4(self.stage3(x))
        x = self.avg_pooling(x)
        x = x.view(-1, self.nchannels)
        x = self.fc1(x)
        if self.droprate_fc > 0:
            x = F.dropout(x, p=self.droprate_fc, training=self.training)
        x = self.fc2(x)

        return x


class MicroNet_M1(nn.Module):
    def __init__(self, droprate, droprate_fc, classNum):
        super(MicroNet_M1, self).__init__()

        self.stem = stemLayers(kernelSize=3, inchannel=3, outchannel=8, midchannel=4, G=(1, 4), stride=2,
                               droprate=droprate)
        self.stage1 = MicroBlockA(3, 8, 32, 12, (4, 0), 2, droprate)
        self.stage2 = nn.Sequential(MicroBlockA(3, 12, 48, 16, (4, 0), 2, droprate),
                                    MicroBlockB(3, 16, 144, 24, (4, 6), 1, droprate))
        self.stage3 = nn.Sequential(self._make_Layers(2, 144, 2, 5, 192, 32, (4, 8), droprate),
                                    MicroBlockC(5, 192, 384, 64, (8, 8), 1, droprate))
        self.stage4 = nn.Sequential(MicroBlockC(5, 384, 576, 96, (8, 12), 2, droprate),
                                    MicroBlockC(3, 576, 768, 128, (8, 16), 1, droprate))
        self.avg_pooling = nn.AvgPool2d(7)
        self.nchannels = 768
        self.fc1 = nn.Linear(768, int(768 // 0.77))
        self.droprate_fc = droprate_fc
        self.fc2 = nn.Linear(int(768 // 0.77), classNum)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_Layers(self, numbers, inchannel, stride, kernelSize, outchannel, midchannel, G, droprate):
        layers = []
        layers.append(MicroBlockC(kernelSize, inchannel, outchannel, midchannel, G, stride, droprate))
        for i in range(1, numbers):
            layers.append(MicroBlockC(kernelSize, outchannel, outchannel, midchannel, G, 1, droprate))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage2(self.stage1(x))
        x = self.stage4(self.stage3(x))
        x = self.avg_pooling(x)
        x = x.view(-1, self.nchannels)
        x = self.fc1(x)
        if self.droprate_fc > 0:
            x = F.dropout(x, p=self.droprate_fc, training=self.training)
        x = self.fc2(x)

        return x


class MicroNet_M2(nn.Module):
    def __init__(self, droprate, droprate_fc, classNum):
        super(MicroNet_M2, self).__init__()

        self.stem = stemLayers(kernelSize=3, inchannel=3, outchannel=12, midchannel=4, G=(1, 4), stride=2,
                               droprate=droprate)
        self.stage1 = MicroBlockA(3, 12, 48, 16, (4, 0), 2, droprate)
        self.stage2 = nn.Sequential(MicroBlockA(3, 16, 64, 24, (4, 0), 2, droprate),
                                    MicroBlockB(3, 24, 144, 24, (4, 6), 1, droprate))
        self.stage3 = nn.Sequential(self._make_Layers(2, 144, 2, 5, 192, 32, (4, 8), droprate),
                                    MicroBlockC(5, 192, 288, 48, (6, 8), 1, droprate),
                                    self._make_Layers(2, 288, 1, 5, 480, 80, (8, 10), droprate))
        self.stage4 = nn.Sequential(MicroBlockC(5, 480, 720, 120, (10, 12), 2, droprate),
                                    MicroBlockC(3, 720, 720, 120, (10, 12), 1, droprate),
                                    MicroBlockC(3, 720, 864, 144, (12, 12), 1, droprate))
        self.avg_pooling = nn.AvgPool2d(7)
        self.nchannels = 864
        self.fc1 = nn.Linear(864, int(864 // 0.88))
        self.droprate_fc = droprate_fc
        self.fc2 = nn.Linear(int(864 // 0.88), classNum)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_Layers(self, numbers, inchannel, stride, kernelSize, outchannel, midchannel, G, droprate):
        layers = []
        layers.append(MicroBlockC(kernelSize, inchannel, outchannel, midchannel, G, stride, droprate))
        for i in range(1, numbers):
            layers.append(MicroBlockC(kernelSize, outchannel, outchannel, midchannel, G, 1, droprate))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage2(self.stage1(x))
        x = self.stage4(self.stage3(x))
        x = self.avg_pooling(x)
        x = x.view(-1, self.nchannels)
        x = self.fc1(x)
        if self.droprate_fc > 0:
            x = F.dropout(x, p=self.droprate_fc, training=self.training)
        x = self.fc2(x)

        return x


class MicroNet_M3(nn.Module):
    def __init__(self, droprate, droprate_fc, classNum):
        super(MicroNet_M3, self).__init__()

        self.stem = stemLayers(kernelSize=3, inchannel=3, outchannel=16, midchannel=4, G=(1, 4), stride=2,
                               droprate=droprate)
        self.stage1 = MicroBlockA(3, 16, 64, 32, (4, 0), 2, droprate)
        self.stage2 = nn.Sequential(MicroBlockB(3, 32, 192, 48, (6, 8), 2, droprate),
                                    self._make_Layers(2, 192, 1, 3, 192, 48, (6, 8), droprate))
        self.stage3 = nn.Sequential(self._make_Layers(2, 192, 2, 5, 256, 64, (8, 8), droprate),
                                    self._make_Layers(2, 256, 1, 5, 384, 96, (8, 12), droprate),
                                    self._make_Layers(2, 384, 1, 5, 576, 144, (12, 12), droprate))
        self.stage4 = nn.Sequential(self._make_Layers(2, 576, 2, 5, 768, 192, (12, 16), droprate),
                                    MicroBlockC(5, 768, 768, 192, (12, 16), 1, droprate),
                                    MicroBlockC(3, 768, 1024, 256, (16, 16), 1, droprate))
        self.avg_pooling = nn.AvgPool2d(7)
        self.nchannels = 1024
        self.fc1 = nn.Linear(1024, int(1024 // 1.05))
        self.droprate_fc = droprate_fc
        self.fc2 = nn.Linear(int(1024 // 1.05), classNum)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_Layers(self, numbers, inchannel, stride, kernelSize, outchannel, midchannel, G, droprate):
        layers = []
        layers.append(MicroBlockC(kernelSize, inchannel, outchannel, midchannel, G, stride, droprate))
        for i in range(1, numbers):
            layers.append(MicroBlockC(kernelSize, outchannel, outchannel, midchannel, G, 1, droprate))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage2(self.stage1(x))
        x = self.stage4(self.stage3(x))
        x = self.avg_pooling(x)
        x = x.view(-1, self.nchannels)
        x = self.fc1(x)
        if self.droprate_fc > 0:
            x = F.dropout(x, p=self.droprate_fc, training=self.training)
        x = self.fc2(x)

        return x


def get_MicroNet(args):
    if args.model == "M0_Net":
        model = MicroNet_M0(args.droprate, args.droprate_fc, args.num_classes) \
            if args.droprate > 0 or args.droprate_fc > 0 else MicroNet_M0(0, 0.05, args.num_classes)
    elif args.model == "M1_Net":
        model = MicroNet_M1(args.droprate, args.droprate_fc, args.num_classes) \
            if args.droprate > 0 or args.droprate_fc > 0 else MicroNet_M1(0, 0.05, args.num_classes)
    elif args.model == "M2_Net":
        model = MicroNet_M2(args.droprate, args.droprate_fc, args.num_classes) \
            if args.droprate > 0 or args.droprate_fc > 0 else MicroNet_M2(0, 0.1, args.num_classes)
    else:
        model = MicroNet_M3(args.droprate, args.droprate_fc, args.num_classes) \
            if args.droprate > 0 or args.droprate_fc > 0 else MicroNet_M3(0, 0.1, args.num_classes)
    return model


if __name__ == '__main__':
    from thop import profile

    # net = MicroNet_M1(0, 0.05, 100)
    net = MicroNet_M0(0, 0.1, 100)
    inputs = torch.randn(1, 3, 224, 224)
    flops, params = profile(net, (inputs,))
    print('flops: ', flops / 1e6, 'M  ', 'params: ', params / 1e6, 'M')
