import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

import MicroNet


def main():
    # Train on GPU 并且使用 CUDNN 加速
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True

    # 参数设置
    batchsize = 512
    epochs = 600
    momentum = 0.9
    ini_LR = 0.02
    weightdecay = 3e-5  # 4e-5

    # 图片变换操作
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    traindata_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])
    valdata_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize])

    # ImageNet数据集准备
    train_dataset = torchvision.datasets.ImageFolder(root='./ImageNet/', transform=traindata_transform)
    train_dataset_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=4)

    val_dataset = torchvision.datasets.ImageFolder(root='./ImageNet/', transform=valdata_transform)
    train_dataset_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False, num_workers=4)
