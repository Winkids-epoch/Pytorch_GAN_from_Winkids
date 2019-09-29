import os
import torch
import torchvision
import torch.nn as nn
import torch.utils.data as data
import matplotlib
matplotlib.use('Qt5agg')
import matplotlib.pyplot as plt

import numpy as np


# Hyperparameters
torch.manual_seed(1)
EPOCH=3
BATCH_SIZE=64
LR =0.002
DOWNLOAD_DATA=True

# download dataset
# data:torch.size([60000,28,28]), targets:torch.size([60000,28,28])
# torchvision.transform.Compose([]) —— image preprocessing
train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,),(0.3081,))
    ]),
    download=DOWNLOAD_DATA
)

# train_dataloader
# x_size: torch.size([batch_size, 1, 28, 28]), y_size: torch.size([batch_size])
train_loader = data.DataLoader(
    dataset=train_data,
    shuffle=True,
    batch_size=BATCH_SIZE,
    num_workers=0
)

# download test dataset( used for test ) [10000,28,28]
# dont use dataloader for test_data, thus should use unsqueeze to add a dim for test_data
test_data = torchvision.datasets.MNIST(root='./mnist', train=False)                                              # [10000,1,28,28]
test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.float32)[:2000]/255                                   # About "dim" see note below
test_y = test_data.targets[:2000]


# check size
# print(train_data.data.size())       #[60000,28,28] torch_size
# print(train_data.targets.size())    #[60000] torch_size
# plt.imshow(train_data.data[0].numpy(), cmap='gray')
# plt.title('%i' % train_data.targets[0])
# plt.show()
# There is no GUI , save and check
# plt.imsave('check.png')

# check the size of dataloader
# for i, (bx, by) in enumerate (train_loader):
#     print(bx.data.shape(), by.data.shape())

# print(matplotlib.get_backend()) —— Linux(agg), local(Qt5agg)

# print(test_x.data.size())

'''
torch.unsqueeze(data, dim=):  add dim at the specified dimension
example: data is (1000,28,28)
        torch.unsqueeze(data, dim=0) ================= (1，1000，28，28)
        torch.unsqueeze(data, dim=1) =================  (1000, 1, 28, 28)
        torch.unsqueeze(data, dim=2) ================= (1000, 28, 1, 28)
'''