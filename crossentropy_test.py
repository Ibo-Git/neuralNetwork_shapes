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
torch.set_printoptions(sci_mode=False)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class NetMultihead(ImageClassificationBase):
    def __init__(self):
        super(NetMultihead, self).__init__()
        self.masked_attention = nn.MultiheadAttention(embed_dim=10, num_heads=1, dropout=0, batch_first=True)
        self.norm = nn.LayerNorm(10)
        self.linear = nn.Linear(10, 10)
        self.softmax = nn.Softmax(dim=2)
        self.tgt_mask = self.generate_square_subsequent_mask(10, 10)

        self.embedding = nn.Embedding(11, 10)
        self.positional_encoding = PositionalEncoding(10, 0)


    def generate_square_subsequent_mask(self, tgt_seq_len, src_seq_len):        
        mask = (torch.triu(torch.ones((tgt_seq_len, src_seq_len))) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):
        padding_mask = (x == 9)

        x = self.embedding(x)
        x = self.positional_encoding(x)
        x, _ = self.masked_attention(x, x, x, key_padding_mask=padding_mask, attn_mask=self.tgt_mask)
        x = self.norm(x)
        x = self.linear(x)
        #x = self.softmax(x)
        return x

vocabOutput = {
    0: '<EOS>',
    1: 'du',
    2: 'spast',
    3: 'bist',
    4: 'so',
    5: 'dumm',
    6: 'xd',
    7: '!',
    8: '<SOS>',
    9: '<PAD>',
}

wordInput = torch.LongTensor([ # Batch
    [8, 1, 2, 3, 4, 5, 6, 7, 9, 9], # Sequence
    [8, 7, 6, 5, 4, 3, 2, 1, 9, 9], # Sequence
    [8, 2, 3, 4, 5, 6, 7, 1, 9, 9], # Sequence
    [8, 3, 4, 5, 6, 7, 1, 2, 9, 9],  # Sequence
])

wordTargets = torch.LongTensor([
    1, 2, 3, 4, 5, 6, 7, 0, 9, 9, # Sequence
    7, 6, 5, 4, 3, 2, 1, 0, 9, 9, # Sequence
    2, 3, 4, 5, 6, 7, 1, 0, 9, 9, # Sequence
    3, 4, 5, 6, 7, 1, 2, 0, 9, 9,  # Sequence
])

loss = nn.CrossEntropyLoss(ignore_index=9)
model = NetMultihead()
optime = torch.optim.Adam(model.parameters(), lr=0.0002)

inputs = torch.Tensor([ # Batch
    [ # Sequence
        [1.0, 0.0, 0.0, 0.0, 5.0, 0.0, 3.0, 0.0, 0.0, 0.0], # Word
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 16.0, 0.0, 0.0, 0.0], # Word
    ],
    [ # Sequence
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0], # Word
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # Word
    ],
    [ # Sequence
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 3.0, 0.0, 0.0, 0.0], # Word
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], # Word
    ],
    [ # Sequence
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], # Word
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], # Word
    ]
])

inputs = torch.randn(4, 2, 10, requires_grad=True)

inputs.requires_grad = True
targets = torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7])
targets = torch.empty(8, dtype=torch.long).random_(10)

for epoch in range(100):
    lossVals = []

    for batch in range(1000):
        if epoch == 99:
            print(torch.argmax(output, 2).reshape(-1))

        #output = model(inputs)
        output = model(wordInput)
        optime.zero_grad()
        #lossVal = loss(output.reshape(-1, output.shape[-1]), targets)
        lossVal = loss(output.reshape(-1, output.shape[-1]), wordTargets)
        lossVal.backward()
        optime.step()
        lossVals.append(lossVal.detach())

        #print(' '.join([vocabOutput[x] for x in torch.argmax(output, 2).reshape(-1).cpu().detach().numpy()]))

    avgLoss = np.average(lossVals)
    print(avgLoss)
    outputFormatted = [vocabOutput[x] for x in torch.argmax(output, 2).reshape(-1).cpu().detach().numpy()]
    outputFormattedSplit = np.array_split(outputFormatted, wordInput.shape[0])
    wordTargetsFormatted = [vocabOutput[x] for x in wordTargets.cpu().detach().numpy()]
    wordTargetsFormattedSplit = np.array_split(wordTargetsFormatted, wordInput.shape[0])
    
    for i in range(len(outputFormattedSplit)):
        print(' '.join(outputFormattedSplit[i])  + ' - Expected: \n' + ' '.join(wordTargetsFormattedSplit[i]))
