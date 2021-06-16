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
        self.positional_encoding = PositionalEncoding(d_model=embedding_size),
        self.transformer = nn.Transformer(d_model = embedding_size),
        self.fc_out = nn.Linear(embedding_size, trg_vocab_size),

    def forward(self, x):
        x = self.src_word_embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer(x)
        x = self.fc_out(x)
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p = dropout)

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

def TrainingLoop(num_epochs, model, optimizer, text, batches, trg_vocab_size):
    for epoch in range(num_epochs):
        for numBatch in range(len(batches)):
            input = batches[numBatch]
            output = model(input)
            loss = training_step(input, output, text, trg_vocab_size) 
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print('Epoch:{}, Batch number: {}'.format(epoch, numBatch))

def training_step(input, output, text, trg_vocab_size):
    index = text.find(input)
    exp_output = np.zeros(len(trg_vocab_size))
    exp_output[index+1] = 1
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, exp_output) 
    return loss

def main():

    #with open ("textfiles/data.txt", "r") as text:
    #    text=text.read().replace('\n', '')
    
    text = "a b c d e f g h i j k l m n o p q r s t u v w x y z"
    uniqueWords = UtilityRNN.getUniqueWords(text)
    word2index = UtilityRNN.assignIndex(uniqueWords)
    batches = UtilityRNN.encodeText(text, word2index, 2)

    embedding_size = 20
    src_vocab_size = len(word2index)
    trg_vocab_size = len(word2index)
    num_epochs = 20 
    model = modelTransformer(src_vocab_size, embedding_size, trg_vocab_size)
    optimizer = torch.optim.adam(model.parameters(), lr = 0.005)
    TrainingLoop(num_epochs, model, optimizer, text, batches, trg_vocab_size)

if __name__ == '__main__':
    main()


# %%
