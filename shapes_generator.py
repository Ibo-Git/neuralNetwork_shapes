#%%
import math
import os
import pathlib
import random
import shutil
import sys
import tarfile
from multiprocessing import Process, freeze_support

import cv2 as cv
import matplotlib
import matplotlib as plt
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

#%%


def createObject(width=10, height=10, color=255, shape="random", fillColor="random"):
    img = color*np.ones((height,width), np.uint8)
    shapeList = ["ellipse", "line", "polygon", "rectangle", "circle"]
    if shape == "random":
        idxShape = random.randint(0, 4)
        shape = shapeList[idxShape]
    
    if fillColor == "random": fillColor = random.randint(0,220)
    if shape == "ellipse": drawEllipse(img, width, height, fillColor)
    elif shape == "line": drawLine(img, width, height, fillColor)
    elif shape == "polygon": drawPolygon(img, width, height, fillColor)
    elif shape == "rectangle": drawRectangle(img, width, height, fillColor)
    elif shape == "circle" : drawCircle(img, width, height, fillColor)

    return [img, shape]

def drawEllipse(img, width, height, fillColor):
    ellipseWidth = random.randint(math.floor(0.2*width), math.ceil(0.4*width))
    ellipseHeight = random.randint(min(math.floor(0.2*height), math.floor(0.65*ellipseWidth)), min(math.floor(0.65*ellipseWidth), math.ceil(0.4*height)))
    center_pos = getRandomPos((width, height), (ellipseWidth, ellipseWidth))
    angle = random.randint(0, 360)
    cv.ellipse(img,center_pos,(ellipseWidth, ellipseHeight), angle, 0, 360, fillColor,-1)  # fill
    if random.randint(0,1):
        thickness = random.randint(1, math.floor(0.3*ellipseHeight))
        cv.ellipse(img,center_pos,(ellipseWidth, ellipseHeight), angle, 0, 360, 0, thickness)  # outline

def drawPolygon(img, width, height, fillColor):
    pos1 = getRandomPos((width, height), (0,0))
    pos2 = getRandomPos((width, height), (0,0))
    diffPos1To2 = [abs(pos1[0]-pos2[0]), abs(pos1[1]-pos2[1])]
    minDiff = math.ceil(0.2*min(width, height))
    while diffPos1To2[0] < minDiff or diffPos1To2[1] < minDiff:
        pos2 = getRandomPos((width, height), (0,0))
        diffPos1To2 = [abs(pos1[0]-pos2[0]), abs(pos1[1]-pos2[1])]

    # calculate point 3
    m = (pos1[1] - pos2[1]) / (pos1[0] - pos2[0])
    # y = m*(x-pos1[0])+pos1[1]
    start_point = [0, math.floor(m*(1-pos1[0])+pos1[1])]
    end_point = [width, math.floor(m*(width-pos1[0])+pos1[1])]

    img2 = 255*np.ones((height,width), np.uint8)
    cv.line(img2, start_point, end_point, 0, thickness=20)
    pos3 = getRandomPos((width, height), (0,0))
    while img2[pos3[0],pos3[1]] == 0: pos3 = getRandomPos((width, height), (0,0))

    pts = np.array([[pos1],[pos2],[pos3]], np.int32)
    pts = pts.reshape((-1,1,2))
    cv.fillPoly(img, [pts], fillColor)
    if random.randint(0,1):
        thickness = random.randint(1, 2)
        cv.polylines(img, [pts], True, 0, thickness)

def drawRectangle(img, width, height, fillColor):
    pos1 = getRandomPos((width, height), (0,0))
    pos2 = getRandomPos((width, height), (0,0))
    diffPos = [abs(pos1[0]-pos2[0]), abs(pos1[1]-pos2[1])]
    minDiff = math.ceil(0.2*min(width, height))
    while diffPos[0] < minDiff or diffPos[1] < minDiff:
        pos2 = getRandomPos((width, height), (0,0))
        diffPos = [abs(pos1[0]-pos2[0]), abs(pos1[1]-pos2[1])]
    cv.rectangle(img, pos1, pos2, fillColor, -1)
    if random.randint(0,1):
        rectWidth = abs(pos1[0]-pos2[0])
        rectHeight = abs(pos1[1]-pos2[1])
        thickness = random.randint(1, max(math.floor(0. *min(rectWidth, rectHeight)),1))
        cv.rectangle(img, pos1, pos2, 0, thickness)
 
def drawCircle(img, width, height, fillColor):
    radius = random.randint(math.floor(min(0.1*width, 0.1*height)), math.floor(min(0.4*width, 0.4*height)))
    posCenter = getRandomPos((width, height), (radius, radius))
    cv.circle(img, posCenter, radius, fillColor, -1)
    if random.randint(0,1): # draw border 
        thickness = random.randint(1, math.floor(0.3*radius))
        cv.circle(img, posCenter, radius, 0, thickness)

def drawLine(img, width, height, fillColor):
    thickness = random.randint(1,5)
    pos1 = getRandomPos((width, height), (0, 0))
    pos2 = getRandomPos((width, height), (0, 0))
    while pos2[0] == pos1[0] or pos2[1] == pos1[1]: pos2 = getRandomPos((width, height), (0, 0))
    cv.line(img, pos1, pos2, fillColor, thickness)
    
def getRandomPos(bounds, padding):
    return (random.randint(padding[0], bounds[0]-1 - padding[0]), random.randint(padding[1], bounds[1]-1 - padding[1]))

def generateData(force = False):
    currentPath = pathlib.Path().absolute()
    
    if os.path.exists(os.path.join(currentPath, "data_shape")) and force != True:
        shutil.rmtree(os.path.join(currentPath, "data_shape"))

    for newFolder in ["/train", "/test"]:
        for newSubFolder in ["/ellipse", "/line", "/polygon", "/rectangle", "/circle"]: 
            targetPath = (str(currentPath)+"/data_shape"+newFolder+newSubFolder)
            newPath = os.path.join(currentPath, targetPath)
            os.makedirs(targetPath, exist_ok=True)

    dataset = [["train", 50000], ["test", 25000]]
    shapes = ["ellipse", "line", "polygon", "rectangle", "circle"]
    for set, number in dataset:
        for j in shapes:
            for i in range(0, int(number/len(shapes))):
                [img, shape] = createObject(width=64, height=64, color=255, shape=j, fillColor="random")
                filename = str("%0"+str(len(str(number)))+"d") % (i,)
                path = os.path.join(currentPath, "data_shape", set, shape)
            
                cv.imwrite(os.path.join(path ,"{}.png".format(filename)), img)

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


def get_default_device():
    if torch.cuda.is_available(): return torch.device('cuda')
    else: return torch.device('cpu')


def main():
    generateData()

    device = get_default_device()

    # Load data and testset
    data_dir = './data_shape'
    dataset = ImageFolder(data_dir+'/train', transform=ToTensor())
    batch_size = 128
    train_dl = DataLoader(dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    train_dl = DeviceDataLoader(train_dl, device)

if __name__ == '__main__':
    main()

# %%
