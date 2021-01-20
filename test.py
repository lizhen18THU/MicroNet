from ptflops import get_model_complexity_info
import torch
from torchvision import models
import MicroNet
from torchstat import stat

# with torch.cuda.device(0):
#     net=MicroNet.M3_Net()
#     FLOPs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,
#                                              print_per_layer_stat=True, verbose=True)
#     print("FLOPs: ",FLOPs)
#     print("params: ",params)

net = MicroNet.M3_Net()
stat(net, (3, 224, 224))

# import os

# os.system("md5sum ILSVRC2012_img_val.tar ILSVRC2012_img_train.tar;"
#           "29b22e2961454d5413ddabcf34fc5622 ILSVRC2012_img_val.tar;"
#           "1d675b47d978889d74fa0da5fadfb00e ILSVRC2012_img_train.tar")

# a=torch.zeros([4,5,8,1])
# a=a+1
# # a=a.unsqueeze(0)
# b=torch.zeros([1,8,3])
# b=b+1
# c=b*a
# b[:,:,2]=torch.zeros([1,8])
# c=c+b
# d=torch.zeros([64,8])
# fc=nn.Linear(8,4)
# e=fc(d)
#
# print(a)
# print(b)
# print(c)
# print(c.size())
# print(e.shape)
# a = 1
# if type(a) == int:
#     print("ok")
