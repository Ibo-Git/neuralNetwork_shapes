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




def main():
    transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    dataset = ImageFolder('./data_shape/train', transform=transform)
    testset = ImageFolder('./data_shape/test', transform=transform)

    train_dl, val_dl = NetUtility.load_data(dataset, [{ "shuffle": True, "percentage": 0.8 }, { "shuffle": False, "percentage": 0.2 }])

    nOutput = 5
    model =  NetUtility.to_optimal_device(ResNet(nOutput))
    num_epochs = 10
    max_lr = 0.005

    optimizer = torch.optim.SGD(model.parameters(), max_lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, steps_per_epoch=len(train_dl), epochs=num_epochs)
    history = model.fitData(num_epochs, train_dl, val_dl, optimizer, scheduler)
    model.getDatasetAccuracy(testset)

if __name__ == '__main__':
    main()

