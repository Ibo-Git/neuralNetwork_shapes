import math
import os
import pathlib
import random
import tarfile
from multiprocessing import Process, freeze_support

import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import optimizer
import torchvision
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms as transforms
from torchvision.datasets import MNIST, ImageFolder
from torchvision.datasets.utils import download_url
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid

from NetBase import DeviceDataLoader, ImageClassificationBase, NetUtility

class ResBlock(ImageClassificationBase):
    def __init__(self, channels_in, channels_out, stride=1):
        super().__init__()
        #self.layerBN = nn.BatchNorm2d(channels_in)
        self.layerRelu = nn.ReLU()
        self.layerConv = nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1, stride = stride)
        self.layerpool = nn.MaxPool2d(2, 2)
        if channels_in != channels_out:
            self.layerRes = nn.Conv2d(channels_in, channels_out, kernel_size=1, stride = stride)
        else:
            self.layerRes = lambda x: x
    
    def forward(self, x):
        #x = self.layerBN(x)
        x = self.layerRelu(x)
        y = self.layerConv(x)
        r = self.layerRes(x)
        return y.add_(r)


class ResNet(ImageClassificationBase):
    def __init__(self, nOutput):
        super(ResNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),   # 32 x 64 x 64
            ResBlock(32, 16),                                       # 16 x 32 x 32
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),   # 8 x 32 x 32
            nn.MaxPool2d(2, 2),                                     # 8 x 16 x 16
            ResBlock(8, 8),                                         # 8 x 8 x 8
            
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, nOutput),
            nn.Relu(),
        )

    def forward(self, x):
        return self.net(x)

def main():
    transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    dataset = ImageFolder('./data_shape/train', transform=transform)
    testset = ImageFolder('./data_shape/test', transform=transform)

    train_dl, val_dl = NetUtility.load_data(dataset, [{ "shuffle": True, "percentage": 0.8 }, { "shuffle": False, "percentage": 0.2 }])

    nOutput = 5
    model =  NetUtility.to_optimal_device(ResNet(nOutput))
    num_epochs = 10
    max_lr = 0.005


    history = model.fitData(num_epochs, max_lr, train_dl, val_dl)
    model.getDatasetAccuracy(testset)

if __name__ == '__main__':
    main()

