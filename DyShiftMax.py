import torch
import torch.nn as nn


class DyShiftMax(nn.Module):
    def __init__(self, channels, groups, reduction, J=2, K=2):
        super(DyShiftMax, self).__init__()
        self.channels = channels
        self.groups = groups
        self.J = J
        self.K = K

        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, J * K * channels)
        self.sigmoid = nn.Sigmoid()

        # 固定的初始参数，不会随着训练而更新，参考论文二作的文章dynamic ReLU里的做法
        self.register_buffer("lambdas", torch.Tensor([0.5] * J).float())  # J个
        self.register_buffer("alphas", torch.Tensor([1.0] + [0.0] * (K - 1)).float())  # K个

    # 获得theta(x)参数
    def get_relu_coefs(self, x):
        theta = torch.mean(x, dim=-1)
        theta = torch.mean(theta, dim=-1)
        theta = self.fc1(theta)
        theta = self.relu(theta)
        theta = self.fc2(theta)
        theta = 2 * self.sigmoid(theta) - 1
        return theta

    def forward(self, x):
        assert self.channels == x.shape[1]

        relu_coefs = self.get_relu_coefs(x).view(-1, self.channels, self.J, self.K)

        # B*C*J*K   using dynamic ReLU
        relu_coefs = relu_coefs.view(-1, self.channels, self.J, self.K).permute(0, 1, 3, 2) * self.lambdas
        relu_coefs = relu_coefs.permute(0, 1, 3, 2) + self.alphas

        # # 不使用循环完成，但是感觉反而产生更大的运算量，所以在J=2的情况下不使用，直接使用循环
        # CG = self.channels // self.groups
        # output = x.expand(self.J, -1, -1, -1, -1)  # J*B*C*H*W
        # index = ((torch.arange(self.channels).expand(self.J, -1).permute(1, 0) + torch.arange(
        #     self.J) * CG) % self.channels).permute((1, 0)).expand(
        #     x.shape[0], x.shape[2], x.shape[3], -1, -1).permute(3, 0, 4, 1, 2)
        # output = (torch.gather(output, dim=2, index=index).permute(3, 4, 1, 2, 0).unsqueeze(dim=-1) * relu_coefs).sum(
        #     dim=-2)

        # B*C*H*W->H*W*B*C*1
        output = x.permute(2, 3, 0, 1).unsqueeze(-1) * relu_coefs[:, :, 0, :]
        for j in range(1, self.J):  # J=2
            output = output + torch.cat((x[:, int(self.channels / self.groups * j):self.channels, :, :],
                                         x[:, 0:int(self.channels / self.groups * j), :, :]),
                                        dim=1).permute(2, 3, 0, 1).unsqueeze(-1) * relu_coefs[:, :, j, :]

        output = torch.max(output, dim=-1)[0].permute(2, 3, 0, 1)

        return output.contiguous()
