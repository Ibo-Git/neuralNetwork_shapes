import copy
import itertools
import math
import os
import pathlib
import random
import re
import shutil
import statistics
import string
import tarfile
from multiprocessing import Process, freeze_support
from os import listdir
from os.path import isfile, join
from typing import Sequence

import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt
import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from IPython.display import Image
from IPython.lib.display import ScribdDocument
from torch import are_deterministic_algorithms_enabled, optim
from torch._C import device
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.optim import optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, dataloader, random_split
from torchvision import transforms as transforms
from torchvision.datasets import MNIST, ImageFolder
from torchvision.datasets.utils import download_url
from torchvision.transforms import ToTensor
from torchvision.transforms.transforms import LinearTransformation
from torchvision.utils import make_grid, save_image

from NetBase import DeviceDataLoader, ImageClassificationBase, NetUtility
from collections import OrderedDict


class NetMultihead(ImageClassificationBase):
    def __init__(self):
        super(NetMultihead, self).__init__()
        self.masked_attention = nn.MultiheadAttention(embed_dim=10, num_heads=1, dropout=0, batch_first=True)
        self.norm = nn.LayerNorm(10)
        self.linear = nn.Linear(10, 10)
        self.softmax = nn.Softmax(dim=2)
        self.tgt_mask = self.generate_square_subsequent_mask(2, 2)

    def generate_square_subsequent_mask(self, tgt_seq_len, src_seq_len):
        mask = (torch.triu(torch.ones((tgt_seq_len, src_seq_len))) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):
        x, _ = self.masked_attention(x, x, x, attn_mask=self.tgt_mask)
        x = self.norm(x)
        x = self.linear(x)
        x = self.softmax(x)
        return x


loss = nn.CrossEntropyLoss()
model = NetMultihead()
optime = torch.optim.Adam(model.parameters(), lr=0.0002)

inputs = torch.Tensor([ # Batch
    [ # Sequence
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # Word
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # Word
    ],
    [ # Sequence
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # Word
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # Word
    ],
    [ # Sequence
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], # Word
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], # Word
    ],
    [ # Sequence
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], # Word
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], # Word
    ]
])

inputs.requires_grad = True
targets = torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7])

for epoch in range(100):
    lossVals = []

    for batch in range(1000):
        if epoch == 99:
            print(torch.argmax(output, 2).reshape(-1))

        output = model(inputs)
        optime.zero_grad()
        lossVal = loss(output.reshape(-1, output.shape[-1]), targets)
        lossVal.backward()
        optime.step()
        lossVals.append(lossVal.detach())

    avgLoss = np.average(lossVals)
    print(avgLoss)
    print(torch.argmax(output, 2).reshape(-1))