from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import os
from thop import profile
import torch
import torch.nn as nn
from torch.autograd import Variable
from functools import reduce
import operator
import numpy as np
import torch.nn.functional as F


def onehot_encoding(labels, n_classes):
    return torch.zeros(labels.size(0), n_classes).scatter_(dim=1, index=labels.view(-1, 1), value=1)


def mixup_data(input, target, alpha=0.2, n_classes=1000):
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = input.size()[0]
    index = torch.randperm(batch_size)

    mixed_input = lam * input + (1 - lam) * input[index, :]  # 自己和打乱的自己进行叠加
    target_a, target_b = onehot_encoding(target, n_classes), onehot_encoding(target[index], n_classes)
    mixed_target = lam * target_a + (1 - lam) * target_b
    return mixed_input, mixed_target


def label_Smothing(targets, epsilon=0.1):
    n_classes = targets.size(1)
    targets = targets * (1 - epsilon) + torch.ones_like(targets) * epsilon / n_classes
    return targets


class CrossEntryLoss_onehot(nn.Module):
    def __init__(self):
        super(CrossEntryLoss_onehot, self).__init__()

    def forward(self, preds, targets, reduction='mean'):
        assert reduction in ["mean", "sum"]
        logp = F.log_softmax(preds, dim=1)
        loss = torch.sum(-logp * targets, dim=1)
        if reduction == "mean":
            return loss.mean()
        else:
            return loss.sum()


def measure_model(model, IMAGE_SIZE1, IMAGE_SIZE2):
    inputs = torch.randn(1, 3, IMAGE_SIZE1, IMAGE_SIZE2)
    flops, params = profile(model, (inputs,))

    return flops, params


def make_log_dir(args):
    if not args.train_on_cloud:
        log_path = os.path.expanduser('~/lizhen_MicroNet_temper/MicroNet_log/') + str(args.dataset) \
                   + '_' + str(args.name) \
                   + '/' + 'no_' + str(args.no) + '/'
        if not args.continue_train:
            while os.path.exists(log_path):
                args.no = int(args.no) + 1
                log_path = os.path.expanduser('~/lizhen_MicroNet_temper/MicroNet_log/') + str(args.dataset) \
                           + '_' + str(args.name) \
                           + '/' + 'no_' + str(args.no) + '/'
        os.makedirs(log_path)
        return log_path
    else:
        log_path = args.train_url
        import moxing as mox
        mox.file.make_dirs(log_path)
        return log_path


# count_ops = 0
# count_params = 0
#
#
# def get_num_gen(gen):
#     return sum(1 for x in gen)
#
#
# def is_pruned(layer):
#     try:
#         layer.mask
#         return True
#     except AttributeError:
#         return False
#
#
# def is_leaf(model):
#     return get_num_gen(model.children()) == 0
#
#
# def is_mscale_conv(model):
#     return get_layer_info(model) in ['MscaleConv_v1', 'MscaleConv_v2', 'MscaleConv_v3']
#
#
# # def get_bn_value(model, args):
# #     for child in model.children():
# #         # print('Layer is', get_layer_info(child))
# #         if get_layer_info(child) == '_DenseBlock':
# #             print('###################################################DenseBlock###################################################')
# #         if get_layer_info(child) == 'Conv':
# #             # print(child.norm.weight.size())
# #             # print(child.norm.bias.size())
# #             print('****************************************************************************************************************')
# #             print('--------Weight Size--------:', child.norm.weight.size())
# #             for i in range(args.group_1x1):
# #                 print('--------Group ' + str(i) + ' Weight--------:', child.norm.weight[i::args.group_1x1])
# #                 print('Weight Mean:', torch.mean(child.norm.weight[i::args.group_1x1]))
# #                 print('Weight Std :', torch.std(child.norm.weight[i::args.group_1x1]))
# #             # print('---------Bias---------:', child.norm.bias)
# #             # print('-----Running Mean-----:', child.norm.running_mean)
# #             # print('-----Running Var------:', child.norm.running_var)
# #         if not is_leaf(child):
# #             get_bn_value(child, args)
#
#
# def get_layer_info(layer):
#     layer_str = str(layer)
#     type_name = layer_str[:layer_str.find('(')].strip()
#     return type_name
#
#
# def get_layer_param(model):
#     return sum([reduce(operator.mul, i.size(), 1) for i in model.parameters()])
#
#
# ### The input batch size should be 1 to call this function
# def measure_layer(layer, x):
#     global count_ops, count_params
#     delta_ops = 0
#     delta_params = 0
#     multi_add = 1
#     type_name = get_layer_info(layer)
#     # print(type_name)
#
#     ### ops_conv
#     if type_name in ['Conv2d']:
#         out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) /
#                     layer.stride[0] + 1)
#         out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) /
#                     layer.stride[1] + 1)
#         delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] * \
#                     layer.kernel_size[1] * out_h * out_w / layer.groups * multi_add
#         delta_params = get_layer_param(layer)
#
#     ### ops_learned_conv
#     elif type_name in ['LearnedGroupConv', 'LearnedGroupConv3x3']:
#         measure_layer(layer.relu, x)
#         measure_layer(layer.norm, x)
#         conv = layer.conv
#         out_h = int((x.size()[2] + 2 * conv.padding[0] - conv.kernel_size[0]) /
#                     conv.stride[0] + 1)
#         out_w = int((x.size()[3] + 2 * conv.padding[1] - conv.kernel_size[1]) /
#                     conv.stride[1] + 1)
#         delta_ops += conv.in_channels * conv.out_channels * conv.kernel_size[0] * \
#                      conv.kernel_size[1] * out_h * out_w / layer.condense_factor * multi_add
#         delta_params = get_layer_param(conv) / layer.condense_factor
#
#     ### ops_learned_conv
#     elif type_name in ['LearnedGroupConv_wobnrelu']:
#         conv = layer.conv
#         out_h = int((x.size()[2] + 2 * conv.padding[0] - conv.kernel_size[0]) /
#                     conv.stride[0] + 1)
#         out_w = int((x.size()[3] + 2 * conv.padding[1] - conv.kernel_size[1]) /
#                     conv.stride[1] + 1)
#         delta_ops += conv.in_channels * conv.out_channels * conv.kernel_size[0] * \
#                      conv.kernel_size[1] * out_h * out_w / layer.condense_factor * multi_add
#         delta_params = get_layer_param(conv) / layer.condense_factor
#
#     ### ops_learned_multiscale_conv_v1
#     elif type_name in ['LearnedMScaleGroupConv_v1', 'LearnedMScaleGroupConv_v3']:
#         measure_layer(layer.relu, x)
#         measure_layer(layer.norm, x)
#         conv = layer.conv
#         avgpool = layer.avgpool
#         groups = layer.groups
#         init_h, init_w = x.size()[2:]
#         for i in range(layer.scale):
#             if i == 0:
#                 out_h = int((init_h + 2 * conv.padding[0] - conv.kernel_size[0]) /
#                             conv.stride[0] + 1)
#                 out_w = out_h
#                 # print('LearnedMScaleGroupConv scale:%d, out_h:%d, out_w:%d' % (i, out_h, out_w))
#             else:
#                 out_h = int((init_h + 2 * avgpool[i - 1].padding - avgpool[i - 1].kernel_size) /
#                             avgpool[i - 1].stride + 1)
#                 out_w = out_h
#                 # print('LearnedMScaleGroupConv scale:%d, out_h:%d, out_w:%d' % (i, out_h, out_w))
#                 delta_ops += avgpool[i - 1].kernel_size * avgpool[i - 1].kernel_size * out_h * out_w * x.size()[1]
#                 # print('LearnedMScaleGroupConv scale:%d, avgpool ops:%d' % (i, avgpool[i - 1].kernel_size * avgpool[i - 1].kernel_size * out_h * out_w * x.size()[1]))
#             delta_ops += conv.in_channels * (conv.out_channels / groups) * conv.kernel_size[0] * \
#                          conv.kernel_size[1] * out_h * out_w / layer.condense_factor * multi_add
#             # print('LearnedMScaleGroupConv scale:%d, conv ops:%d' % (i, conv.in_channels * (conv.out_channels / groups) * conv.kernel_size[0] * conv.kernel_size[1] * out_h * out_w / layer.condense_factor * multi_add))
#         delta_params = get_layer_param(conv) / layer.condense_factor
#
#     ### ops_learned_multiscale_conv_v2
#     elif type_name in ['LearnedMScaleGroupConv_v2']:
#         measure_layer(layer.relu, x)
#         measure_layer(layer.norm, x)
#         conv = layer.conv
#         avgpool = layer.avgpool
#         groups = layer.groups
#         init_h, init_w = x.size()[2:]
#         for i in range(groups):
#             if i < 2:
#                 out_h = int((init_h + 2 * conv.padding[0] - conv.kernel_size[0]) /
#                             conv.stride[0] + 1)
#                 out_w = out_h
#                 # print('LearnedMScaleGroupConv scale:%d, out_h:%d, out_w:%d' % (i, out_h, out_w))
#             else:
#                 out_h = int((init_h + 2 * avgpool[i - 2].padding - avgpool[i - 2].kernel_size) /
#                             avgpool[i - 2].stride + 1)
#                 out_w = out_h
#                 # print('LearnedMScaleGroupConv scale:%d, out_h:%d, out_w:%d' % (i, out_h, out_w))
#                 delta_ops += avgpool[i - 2].kernel_size * avgpool[i - 2].kernel_size * out_h * out_w * x.size()[1]
#                 # print('LearnedMScaleGroupConv scale:%d, avgpool ops:%d' % (i, avgpool[i - 1].kernel_size * avgpool[i - 1].kernel_size * out_h * out_w * x.size()[1]))
#             delta_ops += conv.in_channels * (conv.out_channels / groups) * conv.kernel_size[0] * \
#                          conv.kernel_size[1] * out_h * out_w / layer.condense_factor * multi_add
#             # print('LearnedMScaleGroupConv scale:%d, conv ops:%d' % (i, conv.in_channels * (conv.out_channels / groups) * conv.kernel_size[0] * conv.kernel_size[1] * out_h * out_w / layer.condense_factor * multi_add))
#         delta_params = get_layer_param(conv) / layer.condense_factor
#
#     elif type_name in ['MscaleConv_v1', 'MscaleConv_v3']:
#         conv = layer.conv
#         groups = layer.groups
#         for i in range(layer.scale):
#             measure_layer(layer.relu, x[i])
#             measure_layer(layer.norm[i], x[i])
#             out_h, out_w = x[i].size()[2:]
#             # print('MscaleConv scale:%d, out_h:%d, out_w:%d' % (i, out_h, out_w))
#             if layer.groups > 1:
#                 delta_ops += (conv.in_channels / groups) * (conv.out_channels / groups) * conv.kernel_size[0] * \
#                              conv.kernel_size[1] * out_h * out_w * multi_add
#                 # print('MscaleConv scale:%d, conv ops:%d' % (i, (conv.in_channels / groups) * (conv.out_channels / groups) * conv.kernel_size[0] * \
#                 #      conv.kernel_size[1] * out_h * out_w * multi_add))
#             else:
#                 delta_ops += (conv.in_channels / layer.scale) * conv.out_channels * conv.kernel_size[0] * \
#                              conv.kernel_size[1] * out_h * out_w * multi_add
#                 # print('MscaleConv scale:%d, conv ops:%d' % (
#                 # i, (conv.in_channels / layer.scale) * conv.out_channels * conv.kernel_size[0] * \
#                 #      conv.kernel_size[1] * out_h * out_w * multi_add))
#         delta_params = get_layer_param(conv)
#
#     elif type_name in ['MscaleConv_v2']:
#         conv = layer.conv
#         groups = layer.groups
#         for i in range(groups):
#             measure_layer(layer.relu, x[i])
#             measure_layer(layer.norm[i], x[i])
#             out_h, out_w = x[i].size()[2:]
#             # print('MscaleConv scale:%d, out_h:%d, out_w:%d' % (i, out_h, out_w))
#             delta_ops += (conv.in_channels / groups) * (conv.out_channels / groups) * conv.kernel_size[0] * \
#                          conv.kernel_size[1] * out_h * out_w * multi_add
#             # print('MscaleConv scale:%d, conv ops:%d' % (i, (conv.in_channels / groups) * (conv.out_channels / groups) * conv.kernel_size[0] * \
#             #      conv.kernel_size[1] * out_h * out_w * multi_add))
#
#         delta_params = get_layer_param(conv)
#
#     ### ops_nonlinearity
#     elif type_name in ['ReLU']:
#         delta_ops = x.numel()
#         delta_params = get_layer_param(layer)
#
#     ### ops_pooling
#     elif type_name in ['AvgPool2d']:
#         in_w = x.size()[2]
#         kernel_ops = layer.kernel_size * layer.kernel_size
#         out_w = int((in_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
#         out_h = int((in_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
#         delta_ops = x.size()[0] * x.size()[1] * out_w * out_h * kernel_ops
#         delta_params = get_layer_param(layer)
#
#     elif type_name in ['AdaptiveAvgPool2d']:
#         delta_ops = x.size()[0] * x.size()[1] * x.size()[2] * x.size()[3]
#         delta_params = get_layer_param(layer)
#
#     ### ops_linear
#     elif type_name in ['Linear']:
#         weight_ops = layer.weight.numel() * multi_add
#         bias_ops = layer.bias.numel()
#         delta_ops = x.size()[0] * (weight_ops + bias_ops)
#         delta_params = get_layer_param(layer)
#
#     ### ops_nothing
#     elif type_name in ['BatchNorm2d', 'Dropout2d', 'DropChannel', 'Dropout']:
#         delta_params = get_layer_param(layer)
#
#     ### unknown layer type
#     else:
#         raise TypeError('unknown layer type: %s' % type_name)
#
#     count_ops += delta_ops
#     count_params += delta_params
#     return
#
#
# def measure_model(model, H, W):
#     global count_ops, count_params
#     count_ops = 0
#     count_params = 0
#     data = Variable(torch.zeros(1, 3, H, W))
#
#     def should_measure(x):
#         return is_leaf(x) or is_pruned(x) or is_mscale_conv(x)
#
#     def modify_forward(model):
#         for child in model.children():
#             if should_measure(child):
#                 def new_forward(m):
#                     def lambda_forward(x):
#                         measure_layer(m, x)
#                         return m.old_forward(x)
#
#                     return lambda_forward
#
#                 child.old_forward = child.forward
#                 child.forward = new_forward(child)
#             else:
#                 modify_forward(child)
#
#     def restore_forward(model):
#         for child in model.children():
#             # leaf node
#             if is_leaf(child) and hasattr(child, 'old_forward'):
#                 child.forward = child.old_forward
#                 child.old_forward = None
#             else:
#                 restore_forward(child)
#
#     modify_forward(model)
#     model.forward(data)
#     restore_forward(model)
#
#     return count_ops, count_params
