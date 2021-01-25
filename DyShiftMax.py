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

        # 不更新的初始参数
        self.register_buffer("lambdas", torch.Tensor([1.0] * J).float())  # J个
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

        theta = self.get_relu_coefs(x)

        relu_coefs = theta.view(-1, self.channels, self.J, self.K).permute(0, 1, 3, 2) * self.lambdas
        relu_coefs = relu_coefs.permute(0, 1, 3, 2) + self.alphas

        # B*C*H*W->H*W*B*C*1
        x_temp = x.permute(2, 3, 0, 1).unsqueeze(-1) * relu_coefs[:, :, 0, :]
        for j in range(1, self.J):
            x_temp = x_temp + torch.cat((x[:, int(self.channels / self.groups * j):self.channels, :, :],
                                         x[:, 0:int(self.channels / self.groups * j), :, :]),
                                        dim=1).permute(2, 3, 0, 1).unsqueeze(-1) * relu_coefs[:, :, j, :]
        output = torch.max(x_temp, dim=-1)[0].permute(2, 3, 0, 1)

        return output.contiguous()
