from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import argparse
import os
import shutil
import time
import math
import warnings
import MicroNet

from utils import measure_model, make_log_dir

parser = argparse.ArgumentParser(description='PyTorch Micro Convolutional Networks')
parser.add_argument('--data_url', metavar='DIR', default='/home/imagenet',
                    help='path to dataset')
parser.add_argument('--dataset', metavar='DATASET', default='imagenet', choices=['cifar10', 'cifar100', 'imagenet'],
                    help='dataset')
parser.add_argument('--pad_mode', metavar='DATA', default='constant',
                    choices=['constant', 'edge', 'reflect', 'symmetric'],
                    help='dataset')
parser.add_argument('--model', default='M0_Net', type=str, metavar='M',
                    help='model to train the dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N', help='mini-batch size (default: 512)')
parser.add_argument('--lr', '--learning-rate', default=0.02, type=float,
                    metavar='LR', help='initial learning rate (default: 0.02)')
parser.add_argument('--lr_type', default='cosine', type=str, metavar='T',
                    help='learning rate strategy (default: cosine)',
                    choices=['cosine', 'multistep'])
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum (default: 0.9)')
parser.add_argument('--weight_decay', '--wd', default=3e-5, type=float,
                    metavar='W', help='weight decay (default: 3e-5)', choices=['3e-5', '4e-5'])
parser.add_argument('--print_freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model (default: false)')
parser.add_argument('--no_save_model', dest='no_save_model', action='store_true',
                    help='only save best model (default: false)')
parser.add_argument('--manual_seed', default=0, type=int, metavar='N',
                    help='manual seed (default: 0)')
parser.add_argument('--gpu', default="7", type=str,
                    help='gpu available')

parser.add_argument('--train_url', type=str, metavar='PATH', default='test',
                    help='path to save result and checkpoint (default: results/savedir)')
parser.add_argument('--name', default='basline', type=str,
                    help='name of experiment')
parser.add_argument('--no', default='1', type=str,
                    help='index of the experiment (for recording convenience)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--droprate', default=0, type=float,
                    help='drop out rate for conv (default: 0)')
parser.add_argument('--droprate_fc', default=0, type=float,
                    help='drop out rate for fc (default: 0)')

parser.add_argument('--evaluate', action='store_true',
                    help='evaluate model on validation set (default: false)')
parser.add_argument('--convert_from', default=None, type=str, metavar='PATH',
                    help='path to saved checkpoint (default: none)')
parser.add_argument('--evaluate_from', default=None, type=str, metavar='PATH',
                    help='path to saved checkpoint (default: none)')

# huawei cloud
parser.add_argument('--train_on_cloud', dest='train_on_cloud', action='store_true', default=False,
                    help='whether to run the code on huawei cloud')

# multiscale
parser.add_argument('--scale_out_ratio_1x1', type=str, metavar='CONV1X1 OUTPUT SCALE RATIO',
                    help='1x1 convolution output scale ratio')
parser.add_argument('--scale_ratio', default='0.5-0.5', type=str, metavar='CONV1X1 OUTPUT SCALE RATIO',
                    help='1x1 convolution output scale ratio')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


if args.dataset == 'cifar10':
    args.num_classes = 10
elif args.dataset == 'cifar100':
    args.num_classes = 100
else:
    args.num_classes = 1000

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)

best_acc1 = 0
args.train_url = make_log_dir(args)
args.record_file = args.train_url + 'training_process.txt'


def main():
    global args, best_acc1

    ### Calculate FLOPs & Param
    if args.model == "M0_Net":
        model = MicroNet.M0_Net(args.droprate,
                                args.droprate) if args.droprate > 0 or args.droprate_fc > 0 else MicroNet.M0_Net()
    elif args.model=="M1_Net":
        model = MicroNet.M1_Net(args.droprate,
                                args.droprate) if args.droprate > 0 or args.droprate_fc > 0 else MicroNet.M1_Net()
    elif args.model=="M2_Net":
        model = MicroNet.M2_Net(args.droprate,
                                args.droprate) if args.droprate > 0 or args.droprate_fc > 0 else MicroNet.M2_Net()
    else:
        model = MicroNet.M3_Net(args.droprate,
                                args.droprate) if args.droprate > 0 or args.droprate_fc > 0 else MicroNet.M3_Net()

    # print('Start Converting ...')
    # convert_model(model, args)
    # print('Converting End!')

    if args.dataset in ['cifar10', 'cifar100']:
        IMAGE_SIZE = 32
    else:
        IMAGE_SIZE = 224

    args.filename = 'log.txt'

    n_flops, n_params = measure_model(model, IMAGE_SIZE, IMAGE_SIZE)
    # print('FLOPs: %.2fM, Params: %.2fM' % (n_flops / 1e6, n_params / 1e6))
    fd = open(args.record_file, 'a')

    print('Args Config:', str(args))
    fd.write(str(args) + '\n')

    print('FLOPs: %.2fM, Params: %.2fM' % (n_flops / 1e6, n_params / 1e6))
    fd.write('FLOPs: %.2fM, Params: %.2fM' % (n_flops / 1e6, n_params / 1e6) + '\n')

    print('Model Struture:', str(model))
    fd.write(str(model) + '\n')

    fd.close()

    model.cuda()

    ### Define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    ### Optionally resume from a checkpoint
    if args.resume:
        checkpoint = load_checkpoint(args)
        if checkpoint is not None:
            args.start_epoch = checkpoint['epoch'] + 1
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])


    cudnn.benchmark = True

    ### Data loading 
    if args.dataset == "cifar10":
        normalize = transforms.Normalize(mean=[0.4914, 0.4824, 0.4467],
                                         std=[0.2471, 0.2435, 0.2616])
        train_set = datasets.CIFAR10(args.data_url, train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.RandomCrop(32, padding=4, padding_mode=args.pad_mode),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         normalize,
                                     ]))
        val_set = datasets.CIFAR10(args.data_url, train=False,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       normalize,
                                   ]))
    elif args.dataset == "cifar100":
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                         std=[0.2675, 0.2565, 0.2761])
        train_set = datasets.CIFAR100(args.data_url, train=True, download=True,
                                      transform=transforms.Compose([
                                          transforms.RandomCrop(32, padding=4, padding_mode=args.pad_mode),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          normalize,
                                      ]))
        val_set = datasets.CIFAR100(args.data_url, train=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        normalize,
                                    ]))
    else:
        traindir = os.path.join(args.data_url, 'train')
        valdir = os.path.join(args.data_url, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_set = datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

        val_set = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False)

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    if args.evaluate:
        validate(val_loader, model,
                 criterion)  # TODO: validate must before measure_model otherwise result be worese, but I don't know why
        n_flops, n_params = measure_model(model, IMAGE_SIZE, IMAGE_SIZE)
        fd = open(args.record_file, 'a')
        print('FLOPs: %.2fM, Params: %.2fM' % (n_flops / 1e6, n_params / 1e6))
        fd.write('FLOPs: %.2fM, Params: %.2fM' % (n_flops / 1e6, n_params / 1e6) + '\n')
        return

    epoch_time = AverageMeter()
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        ### Train for one epoch
        tr_acc1, tr_acc5, tr_loss, lr = \
            train(train_loader, model, criterion, optimizer, epoch)

        ### Evaluate on validation set
        val_acc1, val_acc5, val_loss = validate(val_loader, model, criterion)

        ### Remember best Acc@1 and save checkpoint
        is_best = val_acc1 > best_acc1
        best_acc1 = max(val_acc1, best_acc1)
        # model_filename = 'checkpoint_%03d.pth.tar' % epoch
        model_filename = 'checkpoint.pth.tar'
        save_checkpoint({
            'epoch': epoch,
            'model': args.model,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, args, is_best, model_filename, "%3d %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n" %
                                          (epoch, val_acc1, val_acc5, tr_acc1, tr_acc5, val_loss, tr_loss, lr))

        epoch_time.update(time.time() - start_time, 1)
        string = 'Duration: %4f H, Left Time: %4f H' % \
                 (epoch_time.sum / 3600, epoch_time.avg * (args.epochs - epoch - 1) / 3600)
        print(string)
        fd = open(args.record_file, 'a')
        fd.write(string + '\n')
        fd.close()
        start_time = time.time()

    ### Convert model and test
    # model = model.cpu().module
    # print('Start Converting ...')
    # convert_model(model, args)
    # print('Converting End!')
    print('Model Struture:', str(model))
    validate(val_loader, model, criterion)
    n_flops, n_params = measure_model(model, IMAGE_SIZE, IMAGE_SIZE)
    print('FLOPs: %.2fM, Params: %.2fM' % (n_flops / 1e6, n_params / 1e6))

    fd = open(args.record_file, 'a')
    fd.write(str(model) + '\n')
    fd.write('FLOPs: %.2fM, Params: %.2fM' % (n_flops / 1e6, n_params / 1e6) + '\n')
    fd.close()

    return


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # learned_module_list = []

    ### Switch to train mode
    model.train()
    running_lr = None

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        progress = float(epoch * len(train_loader) + i) / (args.epochs * len(train_loader))
        args.progress = progress
        ### Adjust learning rate
        lr = adjust_learning_rate(optimizer, epoch, args, batch=i,
                                  nBatch=len(train_loader), method=args.lr_type)
        if running_lr is None:
            running_lr = lr

        ### Measure data loading time
        data_time.update(time.time() - end)

        # target = target.cuda(non_blocking=True)
        # input_var = torch.autograd.Variable(input)
        # target_var = torch.autograd.Variable(target)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        ### Compute output
        output = model(input, progress)
        loss = criterion(output, target)

        ### Measure accuracy and record loss
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

        ### Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ### Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            fd = open(args.record_file, 'a')
            string = ('Epoch: [{0}][{1}/{2}]\t'
                      'Lr {lr:.4f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                epoch, i, len(train_loader), lr=lr, batch_time=batch_time, data_time=data_time,
                loss=losses, top1=top1, top5=top5))
            print(string)
            fd.write(string + '\n')
            fd.close()

    return top1.avg, top5.avg, losses.avg, running_lr


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    ### Switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            # target = target.cuda(non_blocking=True)
            # input_var = torch.autograd.Variable(input, volatile=True)
            # target_var = torch.autograd.Variable(target, volatile=True)
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            ### Compute output
            output = model(input)
            loss = criterion(output, target)

            ### Measure accuracy and record loss
            acc1, acc5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data.item(), input.size(0))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))

            ### Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                fd = open(args.record_file, 'a+')
                string = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                print(string)
                fd.write(string + '\n')
                fd.close()

    string = ' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5)
    print(string)
    fd = open(args.record_file, 'a+')
    fd.write(string + '\n')
    fd.close()

    return top1.avg, top5.avg, losses.avg


def load_checkpoint(args):
    # model_dir = os.path.join(args.train_url, 'save_models')
    # latest_filename = os.path.join(model_dir, 'latest.txt')
    # if os.path.exists(latest_filename):
    #     with open(latest_filename, 'r') as fin:
    #         model_filename = fin.readlines()[0]
    # else:
    #     return None
    fd = open(args.record_file, 'a')
    model_filename = args.resume
    print("=> loading checkpoint '{}'".format(model_filename))
    state = torch.load(model_filename)
    print("=> loaded checkpoint '{}'".format(model_filename))
    fd.write("=> loaded checkpoint '{}'".format(model_filename) + '\n')
    fd.close()
    return state


def save_checkpoint(state, args, is_best, filename, result):
    # print(args)
    result_filename = os.path.join(args.train_url, args.filename)
    model_dir = os.path.join(args.train_url, 'save_models')
    model_filename = os.path.join(model_dir, filename)
    latest_filename = os.path.join(model_dir, 'latest.txt')
    best_filename = os.path.join(model_dir, 'model_best.pth.tar')
    os.makedirs(args.train_url, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    print("=> saving checkpoint '{}'".format(model_filename))
    with open(result_filename, 'a') as fout:
        fout.write(result)
    torch.save(state, model_filename)
    with open(latest_filename, 'w') as fout:
        fout.write(model_filename)
    if args.no_save_model:
        shutil.move(model_filename, best_filename)
    elif is_best:
        shutil.copyfile(model_filename, best_filename)

    print("=> saved checkpoint '{}'".format(model_filename))
    return


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args, batch=None,
                         nBatch=None, method='cosine'):
    if method == 'cosine':
        T_total = args.epochs * nBatch
        T_cur = (epoch % args.epochs) * nBatch + batch
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total))
    elif method == 'multistep':
        if args.dataset in ['cifar10', 'cifar100']:
            lr, decay_rate = args.lr, 0.1
            if epoch >= args.epochs * 0.75:
                lr *= decay_rate ** 2
            elif epoch >= args.epochs * 0.5:
                lr *= decay_rate
        else:
            """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
            lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the accision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
