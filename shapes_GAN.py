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

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        cOut = 4
        self.net = nn.Sequential(
            # input: 1 x 64 x 64
            nn.Conv2d(1, cOut, kernel_size=3, stride=2, padding=1), 
            nn.LeakyReLU(0.2, inplace=True),
            # 4 x 32 x 32
            nn.Conv2d(cOut, cOut*2, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(cOut*2),
            nn.LeakyReLU(0.2, inplace=True),
            # 8 x 16 x 16
            nn.Conv2d(cOut*2, cOut*4, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(cOut*4),
            nn.LeakyReLU(0.2, inplace=True),
            # 16 x 8 x 8
            nn.Conv2d(cOut*4, cOut*8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(cOut*8),
            nn.LeakyReLU(0.2, inplace=True),
            # 32 x 4 x 4
            nn.Conv2d(cOut*8, 1, kernel_size=4, stride=1, padding=0),
            # 1 x 1 x 1
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
    
class Generator(nn.Module):
    def __init__(self, cIn):
        super(Generator, self).__init__()
        cOut = 64
        # output_size = (inp-1)*str-2*pad+(ker-1)+1
        self.net2 = nn.Sequential(
            # input: cInx1x1
            nn.ConvTranspose2d(cIn, cOut*8, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(cOut*8),
            nn.ReLU(), 
            # 512x2x2
            nn.ConvTranspose2d(cOut*8, cOut * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(cOut * 4),
            nn.ReLU(), 
            # 256x4x4
            nn.ConvTranspose2d(cOut * 4, cOut * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(cOut * 2),
            nn.ReLU(), 
            # 128x8x8
            nn.ConvTranspose2d(cOut * 2, cOut, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(cOut),
            nn.ReLU(), 
            # 64x16x16
            nn.ConvTranspose2d(cOut, cOut // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(cOut // 2),
            nn.ReLU(), 
            # 32x32x32
            nn.ConvTranspose2d(cOut // 2, 1, kernel_size=4, stride=2, padding=1), 
            nn.Tanh()
            # 1x64x64
        ) 

        self.net = nn.Sequential(
            # input: cIn x 1 x 1
            nn.ConvTranspose2d(cIn, cOut*8, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(cOut*8),
            nn.ReLU(),
            # 512 x 4 x 4
            nn.ConvTranspose2d(cOut*8, cOut*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(cOut*4),
            nn.ReLU(),
            # 256 x 8 x8 
            nn.ConvTranspose2d(cOut*4, cOut*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(cOut*2),
            nn.ReLU(),
            # 128 x 16 x 16
            nn.ConvTranspose2d(cOut*2, cOut, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(cOut),
            nn.ReLU(),
            # 32 x 32 x 32
            nn.ConvTranspose2d(cOut, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
            # output: 1 x 64 x 64
        )

    def forward(self, x):

        return self.net(x)

def train_discriminator(images, latent_size, d_optimizer, g_optimizer, D, G, epoch):
    # loss for real image
    output = D(images)
    batch_size = len(images)
    loss = nn.BCELoss()
    real_loss = loss(output, (NetUtility.to_optimal_device(random.uniform(-0.3, 0.3) / (epoch + 1) + torch.ones([batch_size, 1, 1, 1])))) # is true image
    real_score = output

    # loss for fake image
    fake_image = G(NetUtility.to_optimal_device(torch.randn(batch_size, latent_size, 1, 1)))
    output = D(NetUtility.to_optimal_device(fake_image))
    fake_loss = loss(output, NetUtility.to_optimal_device(torch.zeros([batch_size, 1, 1, 1]))) # is fake image
    fake_score = output

    # combine
    d_loss = real_loss + fake_loss
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()
    d_loss.backward()
    d_optimizer.step()

    return d_loss, real_score, fake_score

def train_generator(batch_size, latent_size, d_optimizer, g_optimizer, D, G):

    fake_image = G(NetUtility.to_optimal_device(torch.randn(batch_size, latent_size, 1, 1)))
    loss = nn.BCELoss()
    g_loss = loss(D(fake_image), NetUtility.to_optimal_device((torch.ones([batch_size, 1, 1, 1]))))
    g_optimizer.zero_grad()
    d_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()

    return g_loss, fake_image

def fitData_GAN(num_epochs, data_loader, batch_size, latent_size, d_optimizer, g_optimizer, D, G, mean , std):

    total_step = len(data_loader)
    d_losses, g_losses, real_scores, fake_scores = [], [], [], []

    static_seed = NetUtility.to_optimal_device(torch.randn(batch_size, latent_size, 1, 1))

    for epoch in range(num_epochs):
        for i, (images, _) in enumerate(data_loader):
            d_loss, real_score, fake_score = train_discriminator(images, latent_size, d_optimizer, g_optimizer, D, G, epoch)
            g_loss, fake_images = train_generator(batch_size, latent_size, d_optimizer, g_optimizer, D, G)


            if (i+1) % 200 == 0 or i + 1 == total_step: # log every 200 steps from data_loader
                d_losses.append(d_loss.item())
                g_losses.append(g_loss.item())
                real_scores.append(real_score.mean().item())
                fake_scores.append(fake_score.mean().item())
                print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
                    .format(epoch + 1, num_epochs, i+1, total_step, d_loss.item(), g_loss.item(), 
                        real_score.mean().item(), fake_score.mean().item()))

        save_fake_images(epoch+1, batch_size, latent_size, G, static_seed, mean, std)

    plt.figure(figsize=(10,5))
    plt.plot(d_losses, '-')
    plt.plot(g_losses, '-')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Discriminator', 'Generator'])
    plt.title('Losses')
    plt.show()

    plt.plot(real_scores, '-')
    plt.plot(fake_scores, '-')
    plt.xlabel('epoch')
    plt.ylabel('score')
    plt.legend(['Real Score', 'Fake score'])
    plt.title('Scores')

def save_fake_images(index, batch_size, latent_size, G, static_seed, mean, std, sample_dir='data_shape/samples'):
    fake_images = G(static_seed)
    fake_images = fake_images.reshape(fake_images.size(0), 1, 64, 64)
    fake_fname = 'fake_images-{0:0=4d}.png'.format(index)
    print('Saving', fake_fname)
    if not os.path.exists(sample_dir): os.makedirs(sample_dir)
    save_image((fake_images), os.path.join(sample_dir, fake_fname), nrow=10)    
    #save_image(denorm(fake_images, mean, std), os.path.join(sample_dir, fake_fname), nrow=10)  

def denorm(x, mean, std):
    out = (x + mean/std) * std
    return out.clamp(0, 1)


def plot_multipleImages(dataset):
    images = torch.stack([image for image, _ in dataset])
    plt.figure(figsize=(10,10))
    plt.axis('off')
    nRand = random.sample(range(1, len(dataset)), 100)
    plt.imshow(make_grid(images[nRand,:,:,:], nrow=10).permute((1, 2, 0)))
    plt.show()
   
    

#%%
def main():
    transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    dataset = ImageFolder('./data_shape/train', transform=transform)
    #images = torch.stack([image for image, labels in dataset])
    #mean = torch.mean(images).item()
    #std = torch.std(images).item()
    mean = 0.5
    std = 0.5
    #transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize((mean, ), (std, ))])
    dataset = ImageFolder('./data_shape/train', transform=transform)

    batch_size = 100
    data_loader = NetUtility.load_data(dataset, subset_configs = [{ "shuffle": True, "percentage": 1 }], batch_size = batch_size)
    
    num_epochs = 100
    latent_size = 100
    lr = 0.0002
    
    d_model = NetUtility.to_optimal_device(Discriminator())
    g_model = NetUtility.to_optimal_device(Generator(latent_size))

    d_model.apply(weights_init)
    g_model.apply(weights_init)

    d_optimizer = torch.optim.Adam(d_model.parameters(), lr, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(g_model.parameters(), lr, betas=(0.5, 0.999))
    fitData_GAN(num_epochs, data_loader, batch_size, latent_size, d_optimizer, g_optimizer, d_model, g_model, mean, std)


if __name__ == '__main__':
    main()

