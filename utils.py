from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import os
from thop import profile
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image


class mini_imagenet(Dataset):
    def __init__(self, root_dir: str, csv_name: str, transform=None):
        images_dir = os.path.join(root_dir, "images")
        csv_dir = os.path.join(root_dir, csv_name)
        csv_data = pd.read_csv(csv_dir)
        self.image_paths = [os.path.join(images_dir, name) for name in csv_data["filename"].values]
        self.image_labels = [label for label in csv_data["label"].values]
        self.transform = transform

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, item):
        image = Image.open(self.image_paths[item])
        label = self.image_labels[item]
        if self.transform is not None:
            image = self.transform(image)

        return image, label


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
        log_path = os.path.expanduser('./MicroNet_log/') + str(args.dataset) \
                   + '_' + str(args.name) \
                   + '/' + 'no_' + str(args.no) + '/'
        if not args.continue_train:
            while os.path.exists(log_path):
                args.no = int(args.no) + 1
                log_path = os.path.expanduser('./MicroNet_log/') + str(args.dataset) \
                           + '_' + str(args.name) \
                           + '/' + 'no_' + str(args.no) + '/'
            os.makedirs(log_path)
        return log_path
