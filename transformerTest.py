#%%
import math
import os
import pathlib
import random
import string
import tarfile
from multiprocessing import Process, freeze_support
from IPython.lib.display import ScribdDocument
import copy

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
    def __init__(self, src_vocab_size, embedding_size, tgt_vocab_size, device):
        self.embedding_size = embedding_size
        super(modelTransformer, self).__init__()
        self.embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.positional_encoding = PositionalEncoding(embedding_size)
        self.tgt_mask = nn.Transformer.generate_square_subsequent_mask(self, sz = 6).to(device)
        self.transformer = nn.Transformer(embedding_size, nhead=2, num_encoder_layers=2, num_decoder_layers=2, dropout=0)
        self.fc_out = nn.Linear(embedding_size, tgt_vocab_size)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, src, tgt):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        src = src.reshape(-1, 1, self.embedding_size)
        tgt = tgt.reshape(-1, 1, self.embedding_size)
        #src = self.positional_encoding(src)
        #tgt = self.positional_encoding(tgt)
        src = self.transformer(src, tgt, tgt_mask = self.tgt_mask)
        src = self.fc_out(src)
        src = self.softmax(src)
        return src


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.2, max_len=5000):
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
        vocab = {'<SOS>': 0, '<EOS>': 1} #{'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        for word in uniqueWords: vocab[word] = len(vocab)
        return vocab
    
    def get_batch(text, batch_size):
        numBatch = len(text.split()) // batch_size
        text = np.array(text.split(" "))
        batches = text[0:numBatch*batch_size].reshape(-1, batch_size)
        return batches
    
    def encodeText(batches, vocab, lookUpTable):
        idx_targets = [[lookUpTable.index(i) for i in batch] for batch in batches]
        targets = [[lookUpTable[x+1 if x != len(lookUpTable)-1 else 0] for x in target ] for target in idx_targets]
        batches = [[vocab[word] for word in batch] for batch in batches]
        targets = [[vocab[word] for word in target] for target in targets]
        exp_outputs = copy.deepcopy(targets)
        for i in range(len(exp_outputs)): 
            exp_outputs[i].append(vocab['<EOS>'])
            targets[i].insert(0,vocab['<SOS>'])
        return torch.tensor(batches), torch.tensor(targets), torch.tensor(exp_outputs)

    def encodeTarget(vector, vocab):
        encodedVec = torch.zeros(len(vector), len(vocab))
        for i in range(len(vector)): encodedVec[i][vector[i]] = 1
        return encodedVec

    def decodeChar(vector, vocab):
        indices = torch.argmax(vector, 1)
        key_list = list(vocab)
        return [key_list[index] for index in indices]

def TrainingLoop(num_epochs, model, optimizer, batches, targets, exp_outputs, vocab, device):
    model.train()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    for epoch in range(num_epochs):
        for numBatch in range(len(batches)):
            input = batches[numBatch]
            target = targets[numBatch]
            exp_output = exp_outputs[numBatch]
            output = model(input, target)
            output = output.reshape(-1, len(vocab))
            exp_output = UtilityRNN.encodeTarget(exp_output, vocab).to(device)
            loss = training_loss(output, exp_output)          
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.zero_grad()
            scheduler.step()

            if numBatch % 25 == 0:
                expOutputChar = UtilityRNN.decodeChar(exp_output, vocab)
                outputChar = UtilityRNN.decodeChar(output, vocab)
                print('Epoch:{}, Batch number: {}, Expected Output: {}, Output: {}, Loss: {}'.format(epoch, numBatch, expOutputChar, outputChar, loss))

def training_loss(output, target):
    criterion = nn.BCELoss()
    loss = criterion(output, target)
    return loss         


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #with open ("textfiles/data.txt", "r") as text:
    #    text=text.read().replace('\n', '')
    text = [random. choice(string.ascii_letters) for i in range(50000)]
    text = ' '.join(text).lower()
    lookUpTable = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",  "u", "v", "w", "x", "y", "z"]
    uniqueWords = UtilityRNN.getUniqueWords(text)
    vocab = UtilityRNN.assignIndex(uniqueWords)
    batches = UtilityRNN.get_batch(text, 5)
    batches, targets, exp_outputs = UtilityRNN.encodeText(batches, vocab, lookUpTable)

    embedding_size = 30
    src_vocab_size = len(vocab)
    tgt_vocab_size = len(vocab)
    num_epochs = 100 
    model = modelTransformer(src_vocab_size, embedding_size, tgt_vocab_size, device).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr = 5)
    batches = batches.to(device)
    targets = targets.to(device)
    exp_outputs = exp_outputs.to(device)
    TrainingLoop(num_epochs, model, optimizer, batches, targets, exp_outputs, vocab, device)

if __name__ == '__main__':
    main()


# %%
