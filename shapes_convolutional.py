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


class NetCNN(ImageClassificationBase):
    def __init__(self, nOutput):
        super(NetCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1), # 4 x 64 x 64
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 4 x 32 x 32
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1), # 8 x 32 x 32
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 8 x 16 x 16
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1), # 16 x 16 x 16
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 16 x 8 x 8
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # 32 x 16 x 16
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 32 x 8 x 8

            nn.Flatten(), 
            nn.Linear(32 * 8 * 8, 32 * 8),
            nn.ReLU(),
            nn.Linear(32 * 8, nOutput)
        )

    def forward(self, x):
        return self.net(x)

def main():
    transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    dataset = ImageFolder('./data_shape/train', transform=transform)
    testset = ImageFolder('./data_shape/test', transform=transform)

    train_dl, val_dl = NetUtility.load_data(dataset, [{ "shuffle": True, "percentage": 0.8 }, { "shuffle": False, "percentage": 0.2 }])

    nOutput = 5
    model =  NetUtility.to_optimal_device(NetCNN(nOutput))
    num_epochs = 250
    lr = 0.002

    history = model.fitData(num_epochs, lr, train_dl, val_dl)
    NetUtility.show_loss_plot(history)

    model.getDatasetAccuracy(testset)

if __name__ == '__main__':
    main()
#%%

a = torch.tensor([])
