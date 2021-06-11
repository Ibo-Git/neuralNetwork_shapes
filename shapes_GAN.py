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
from IPython.display import Image
from torch import optim
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.optim import optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms as transforms
from torchvision.datasets import MNIST, ImageFolder
from torchvision.datasets.utils import download_url
from torchvision.transforms import ToTensor
from torchvision.transforms.transforms import LinearTransformation
from torchvision.utils import make_grid, save_image

from NetBase import DeviceDataLoader, ImageClassificationBase, NetUtility


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        cOut = 4
        self.net = nn.Sequential(
            # input: 1 x 64 x 64
            nn.Conv2d(1, cOut, kernel_size=3, stride=2, padding=1), 
            nn.LeakyReLU(0.2, inplace=True),
            # 4 x 32 x 32
            nn.Conv2d(cOut, cOut*2, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(cOut*2),
            nn.LeakyReLU(0.2, inplace=True),
            # 8 x 16 x 16
            nn.Conv2d(cOut*2, cOut*4, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(cOut*4),
            nn.LeakyReLU(0.2, inplace=True),
            # 16 x 8 x 8
            nn.Conv2d(cOut*4, cOut*8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(cOut*4),
            nn.LeakyReLU(0.2, inplace=True),
            # 32 x 4 x 4
            nn.Conv2d(cOut*8, 1, kernel_size=4, stride=1, padding=0),
            # 1 x 1 x 1
            nn.Sigmoid()
        )

    def forward(self, x):
        self.net(x)
    
class Generator(nn.Module):
    def __init__(self, cIn):
        super(Generator, self).__init__()
        cOut = 64
        self.net = nn.Sequential(
            # input: cIn x 1 x 1
            nn.ConvTranspose2d(cIn, cOut*4, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(cOut*4),
            nn.ReLU(),
            # 512 x 4 x 4
            nn.ConvTranspose2d(cOut*4, cOut*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(cOut*2),
            nn.ReLU(),
            # 256 x 8 x 8
            nn.ConvTranspose2d(cOut*2, cOut, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(cOut),
            nn.ReLU(),
            # 128 x 16 x 16
            nn.ConvTranspose2d(cOut, 1, kernel_size=4, stride=4, padding=0),
            nn.Sigmoid()
            # (inp-1)*str-2*pad+(ker-1)+1
            # output: 1 x 64 x 64
        )

    def forward(self, x):
        self.net(x)

def train_discriminator(images, batch_size, latent_size, d_optimizer, g_optimizer, D, G):
    # loss for real image
    output = D(images)
    real_loss = nn.BCELoss(output, 1) # is true image
    real_score = output

    # loss for fake image
    fake_image = G(torch.randn(batch_size, latent_size))
    output = D(fake_image)
    fake_loss = nn.BCELoss(output, 0) # is fake image
    fake_score = output

    # combine
    d_loss = real_loss + fake_loss
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()
    d_loss.backward()
    d_optimizer.step()

    return d_loss, real_score, fake_score

def train_generator(batch_size, latent_size, d_optimizer, g_optimizer, D, G):

    fake_image = G(torch.randn(batch_size, latent_size))
    g_loss = nn.BCELoss(D(fake_image), 1)
    g_optimizer.zero_grad()
    d_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()

    return g_loss, fake_image

def fitData_GAN(num_epochs, data_loader, batch_size, latent_size, d_optimizer, g_optimizer, D, G):

    total_step = len(data_loader)
    d_losses, g_losses, real_scores, fake_scores = [], [], [], []

    for epoch in num_epochs:
        for i, (images, _) in enumerate(data_loader):
            d_loss, real_score, fake_score = train_discriminator(images, batch_size, latent_size, d_optimizer, g_optimizer)
            g_loss, fake_images = train_generator(batch_size, latent_size, d_optimizer, g_optimizer, D, G)
            d_losses.append(d_loss)
            g_losses.append(g_loss)
            real_scores.append(real_score)
            fake_scores.append(fake_score)

            if (i+1) % 200 == 0: # log every 200 steps from data_loader
                print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
                    .format(epoch, num_epochs, i+1, total_step, d_loss.item(), g_loss.item(), 
                        real_score.mean().item(), fake_score.mean().item()))

        save_fake_images(epoch+1, batch_size, latent_size)

def save_fake_images(index, batch_size, latent_size, sample_dir='samples'):
    fake_images = Generator(torch.randn(batch_size, latent_size))
    fake_images = fake_images.reshape(fake_images.size(0), 1, 64, 64)
    fake_fname = 'fake_images-{0:0=4d}.png'.format(index)
    print('Saving', fake_fname)
    if not os.path.exists(sample_dir): os.makedirs(sample_dir)
    save_image((fake_images), os.path.join(sample_dir, fake_fname), nrow=10)

def main():
    transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    dataset = ImageFolder('./data_shape/train', transform=transform)
    batch_size = 128
    data_loader = NetUtility.load_data(dataset, subset_configs = [{ "shuffle": True, "percentage": 1 }], batch_size = batch_size)
    
    num_epochs = 10
    latent_size = 100
    lr = 0.0002
    d_optimizer = torch.optim.Adam(Discriminator.parameters(), lr)
    g_optimizer = torch.optim.Adam(Generator.parameters(), lr)
    D = NetUtility.to_optimal_device(Discriminator())
    G = NetUtility.to_optimal_device(Generator())
    fitData_GAN(num_epochs, data_loader, batch_size, latent_size, d_optimizer, g_optimizer, D, G)


if __name__ == '__main__':
    main()

