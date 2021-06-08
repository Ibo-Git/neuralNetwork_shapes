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
from torch.utils.data.dataset import Subset
    
class NetUtility():
    def to_optimal_device(data):
        return NetUtility.to_device(data, NetUtility.get_default_device())

    def to_device(data, device):
        """Move tensor(s) to chosen device"""
        if isinstance(data, (list,tuple)):
            return [NetUtility.to_device(x, device) for x in data]
        return data.to(device, non_blocking=True)

    def get_default_device():
        if torch.cuda.is_available(): return torch.device('cuda')
        else: return torch.device('cpu')

    def flatten(x):
        return torch.flatten(x)

    def load_data(dataset, subset_configs = [{ "shuffle": True, "percentage": 1 }], batch_size = 128):
        if len(subset_configs) == 1: return DeviceDataLoader(DataLoader(dataset, batch_size, subset_configs[0]["shuffle"], num_workers=8, pin_memory=True, persistent_workers=True), NetUtility.get_default_device())

        data_loaders = []
        subsets = random_split(dataset, list(map(lambda x: int(x["percentage"] * len(dataset)), subset_configs)))

        for i, config in enumerate(subset_configs):
            data_loaders.append(DeviceDataLoader(DataLoader(subsets[i], batch_size, config["shuffle"], num_workers=8, pin_memory=True, persistent_workers=True), NetUtility.get_default_device()))

        return tuple(data_loaders)
        
    
    def show_loss_plot(history):
        losses = [x['val_loss'] for x in history]
        trainings_loss = [x['train_loss'] for x in history]
        accuracies = [x['val_acc'] for x in history]

        fig, axs = plt.subplots(3)
        fig.suptitle('Summary plot')
        axs[0].plot(losses)
        axs[0].set_title('Loss')
        axs[1].plot(accuracies)
        axs[1].set_title('Accurracy')
        axs[2].plot(trainings_loss)
        axs[2].set_title('Trainings loss') 
        plt.subplots_adjust(left  = 0.125, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.2, hspace = 0.6)
        plt.show()
    
    def show_grid(imageset, classes, currrent_class, numImagePlot = 1000, nrow = 25):
        imageset_filtered = list(filter(lambda x: len(x) > 0, imageset))
        if len(imageset_filtered) == 0: return
    
        fig, axs = plt.subplots(len(imageset_filtered))
        fig.suptitle(currrent_class)
        ix = 0

        if len(imageset_filtered) == 1: axs = [axs] # Make sure it can be accessed via [ix]

        for i in range(len(classes)):
            if len(imageset[i]) > 0: 
                axs[ix].set_xticks([])
                axs[ix].set_yticks([])
                axs[ix].set_title(classes[i])
                axs[ix].imshow(make_grid(imageset[i][:numImagePlot], nrow=nrow).permute(1, 2, 0).clamp(0, 1))
                ix += 1

        plt.subplots_adjust(left  = 0.125, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.2, hspace = 0.6)
        plt.show()

            

    



class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield NetUtility.to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                   
        loss = F.cross_entropy(out, labels)  
        acc = self.accuracy(out, labels)   
        return { 'val_loss': loss, 'val_acc': acc }

    def accuracy(self, outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))

    def evaluate(self, val_dl):
        outputs = [self.validation_step(batch) for batch in val_dl]
        loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        return { 'val_loss': loss.item(), 'val_acc': acc.item() }
    
    def logEpoch(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))

    def fitData(self, epochs, train_dl, val_dl, optimizer, scheduler):
        train_losses = []
        lrs = []
        history = []
        for epoch in range(epochs):
            # Training
            for batch in train_dl:
                loss = self.training_step(batch)
                train_losses.append(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                lrs.append(optimizer.param_groups[0]['lr'])
                if scheduler: scheduler.step()

            # Validation
            result = self.evaluate(val_dl)
            result['train_loss'] = torch.stack(train_losses).mean().item()
            result['lrs'] = lrs
            self.logEpoch(epoch, result)
            history.append(result)
        return history

    def predictItem(self, image):
        xb = NetUtility.to_optimal_device(image.unsqueeze(0))
        yb = self(xb)
        _,pred = torch.max(yb, dim=1)
        return pred[0].item()

    def getDataloaderAccuracy(self, dataLoader):
        classes = dataLoader.dl.dataset.dataset.classes if type(dataLoader.dl.dataset) is Subset else dataLoader.dl.dataset.classes 
        numCorrect = np.zeros(len(classes))
        numFalse = np.zeros(len(classes))

        for batch in dataLoader:
            image, label = batch
            batchsize = image.shape[0]

            for nImage in range(batchsize):
                predLabel = self.predictItem(image[nImage,:])
                if predLabel == label[nImage]: numCorrect[label[nImage]] += 1
                else: 
                    numFalse[label[nImage]] += 1
        
        nClasses = len(classes)
        acc = np.zeros(nClasses)
        outputString = []
        for i in range(nClasses):
            acc[i] = numCorrect[i] / (numCorrect[i] + numFalse[i])
            outputString.append(classes[i] + ": " + str(acc[i]))

        print(", ".join(outputString))

    def getDatasetAccuracy(self, dataset):
        classes = dataset.classes
        numCorrect = np.zeros(len(classes))
        numFalse = np.zeros(len(classes))
        categorizedImage = [[[] for _ in range(len(classes))] for _ in range(len(classes))]

        for image, label in dataset:
            predLabel = self.predictItem(image)
            if predLabel == label: numCorrect[label] += 1
            else: 
                numFalse[label] += 1
                categorizedImage[label][predLabel].append(image)

        acc = np.zeros(len(classes))
        outputString = []

        for i in range(len(classes)):
            acc[i] = numCorrect[i]/(numCorrect[i]+numFalse[i])
            outputString.append(classes[i] + ": " + str(acc[i]))

        print(", ".join(outputString))

        for i in range(len(classes)):
            NetUtility.show_grid(categorizedImage[i], classes, classes[i])