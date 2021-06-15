#%%
import math
import os
import pathlib
import random
import tarfile
from multiprocessing import Process, freeze_support
from IPython.lib.display import ScribdDocument

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


class modelTransformer(nn.Module):
    def __init__(self, src_vocab_size, embedding_size, trg_vocab_size):
        super(modelTransformer, self).__init__()
        self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size),
        self.transformer = nn.Transformer(d_model = embedding_size),
        self.fc_out = nn.Linear(embedding_size, trg_vocab_size),

    def forward(self, x):
        x = self.src_word_embedding(x)
        # ...
        return x

class UtilityRNN():
    def getUniqueWords(text):
        uniqueWords = []
        for word in text.split():
            if word not in uniqueWords: uniqueWords.append(word)
        return uniqueWords

    def assignIndex(uniqueWords):
        word2index = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        for word in uniqueWords: word2index[word] = len(word2index)
        return word2index
    
    def encodeText(text, word2index, batch_size):
        batches = np.array(text.split(" ")).reshape(-1, batch_size)
        if math.floor(len(text.split())/batch_size) < len(batches): batches = batches[0:-1]
        return [[word2index[word] for word in batch] for batch in batches]


def main():

    with open ("textfiles/data.txt", "r") as text:
        text=text.read().replace('\n', '')
    uniqueWords = UtilityRNN.getUniqueWords(text)
    word2index = UtilityRNN.assignIndex(uniqueWords)

    embedding_size = 512
    src_vocab_size = len(word2index)
    trg_vocab_size = 10 # ?

    model = modelTransformer(src_vocab_size, embedding_size, trg_vocab_size)

if __name__ == '__main__':
    main()


# %%
