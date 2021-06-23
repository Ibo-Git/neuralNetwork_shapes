import copy
import itertools
import math
import os
import pathlib
import random
import re
import shutil
import string
import tarfile
from multiprocessing import Process, freeze_support
from os import listdir
from os.path import isfile, join
from typing import Sequence

import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from IPython.display import Image
from IPython.lib.display import ScribdDocument
from torch import optim
from torch._C import device
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


class modelTransformer(nn.Module):
    def __init__(self, 
                 src_vocab_size, 
                 embedding_size, 
                 tgt_vocab_size, 
                 tgt_seq_len, device,
                 n_heads, #8
                 num_encoder_layers, #3
                 num_decoder_layers, #3
                 dropout): #0.1

        self.embedding_size = embedding_size
        super(modelTransformer, self).__init__()
        self.embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.positional_encoding = PositionalEncoding(embedding_size)
        self.tgt_mask = nn.Transformer.generate_square_subsequent_mask(self, sz = tgt_seq_len).to(device)
        self.transformer = nn.Transformer(embedding_size=embedding_size, 
                                          nhead=n_heads, 
                                          num_encoder_layers=num_encoder_layers, 
                                          num_decoder_layers=num_decoder_layers, 
                                          dropout=dropout)
        self.fc_out = nn.Linear(embedding_size, tgt_vocab_size)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, src, tgt):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)
        out = self.transformer(src, tgt, tgt_mask = self.tgt_mask)
        out = self.fc_out(out)
        out = self.softmax(out)
        return out


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
    def read_dataset():
        current_path = pathlib.Path().absolute()
        files_names = os.listdir(os.path.join(current_path, 'trump\\originals'))
        file_all = []
        for file_name in files_names:
            with open(os.path.join(current_path, 'trump\\originals', file_name), 'r', encoding="UTF-8") as file:
                file = file.read().replace('\n', '')
            file_all.append(file)
        
        file_all = ''.join(file_all)
        return file_all

    def getUniqueWords(text):
        uniqueWords = []
        text = re.sub('([.,!?"()])', r' \1 ', text)
        text = re.sub('\s{2,}', ' ', text)
        text = re.sub(r'([0-9]{1}) . ([0-9]{1})', r'\1.\2', text)
        for word in text.split():
            if word not in uniqueWords: uniqueWords.append(word)
        return uniqueWords

    def assignIndex(uniqueWords):
        vocab = {'<SOS>': 0, '<EOS>': 1} #{'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        for word in uniqueWords: vocab[word] = len(vocab)
        return vocab

    def splitText(text):
        text = re.sub('([.,!?"()])', r' \1 ', text)
        text = re.sub('\s{2,}', ' ', text)
        text = re.sub(r'([0-9]{1}) . ([0-9]{1})', r'\1.\2', text)
        text = list(text.split(" "))
        return text

    def text2index(text, encIn_seq_len, decIn_seq_len, vocab, device):
        # text -> sequence
        numSequence_max = len(text) // (encIn_seq_len + decIn_seq_len)
        seqText = [[[] for j in range(3)] for i in range(numSequence_max)]
        for numSequence in range(numSequence_max):
            startIdx = numSequence * (encIn_seq_len + decIn_seq_len)
            endIdx_enc = startIdx + encIn_seq_len
            endIdx_dec = endIdx_enc + decIn_seq_len
            seqText[numSequence][0] = text[startIdx:endIdx_enc]
            seqText[numSequence][1] = text[endIdx_enc:endIdx_dec]
            seqText[numSequence][2] = text[endIdx_enc:endIdx_dec]   
        # sequence -> index
        seqIndex = [[[vocab[word] for word in seqType] for seqType in seqText[seq]] for seq in range(len(seqText))]
        for i in range(len(seqIndex)): 
            seqIndex[i][1].append(vocab['<EOS>'])
            seqIndex[i][2].insert(0,vocab['<SOS>'])
        return seqIndex     

    def dataloader(seqIndex, percent_val, percent_test, batch_size, device, shuffle=True):
        # shuffle
        if shuffle: random.shuffle(seqIndex)
        # extract encoder input, decoder input, expected output
        encoder_input = torch.Tensor([seqIndex[i][0] for i in range(len(seqIndex))]).to(device)
        decoder_intput = torch.Tensor([seqIndex[i][1] for i in range(len(seqIndex))]).to(device)
        expected_output = torch.Tensor([seqIndex[i][2] for i in range(len(seqIndex))]).to(device)
        # get index for splitting into train, val, test
        idx_split_1 = math.ceil(math.ceil(len(seqIndex)*(1-percent_val-percent_test))/batch_size)*batch_size
        idx_split_2 = math.ceil(len(seqIndex)*(1-percent_test))
        # split into batches for training dataset
        numBatches = idx_split_1//batch_size
        enc_train = encoder_input[:idx_split_1].reshape(numBatches, batch_size, -1)
        dec_train = decoder_intput[:idx_split_1].reshape(numBatches, batch_size, -1)
        expOut_train = expected_output[:idx_split_1].reshape(numBatches, batch_size, -1)
        # get train, val and test
        train_ds = [[enc_train[i], dec_train[i], expOut_train[i], expOut_train[i].reshape(-1)] for i in range(numBatches)]
        val_ds = [encoder_input[idx_split_1:idx_split_2], decoder_intput[idx_split_1:idx_split_2], expected_output[idx_split_1:idx_split_2], expected_output[idx_split_1:idx_split_2].reshape(-1)]
        test_ds = [encoder_input[idx_split_2:-1], decoder_intput[idx_split_2:-1], expected_output[idx_split_2:-1], expected_output[idx_split_1:idx_split_2].reshape(-1)]
        
        return train_ds, val_ds, test_ds




        

    def encodeTarget(vector, vocab):
        # check dimensions of vector
        if len(vector.shape) == 2: encodedVec = torch.zeros(vector.shape[1], vector.shape[0], len(vocab))
        elif len(vector.shape) == 1: encodedVec = torch.zeros(vector.shape[1], 1, len(vocab))

        vector = vector.permute(1, 0)
        for batch in range(vector.shape[1]):
            for entry in range(vector.shape[0]): 
                encodedVec[entry][batch][vector[entry][batch]] = 1
        
        return encodedVec

    def decodeChar(vector, vocab):
        indices = torch.argmax(vector, 2)
        key_list = list(vocab)
        decodedVec = [[key_list[index] for index in batch] for batch in indices]
        decodedVec = np.reshape(decodedVec, (indices.shape[0], indices.shape[1])).T
        return decodedVec



def TrainingLoop(num_epochs, model, optimizer, text, vocab, lookUpTable, encIn_seq_len, decIn_seq_len, batch_size, device):
    model.train()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    for epoch in range(num_epochs):
        numBatch_max = (len(text) // ((encIn_seq_len+decIn_seq_len)*batch_size))-1
        accuracies = []
        for numBatch in range(0, numBatch_max):
            input, target = UtilityRNN.get_batch(text, encIn_seq_len, decIn_seq_len, batch_size, numBatch)
            input, target, exp_output = UtilityRNN.encodeBatch(input, target, vocab, device)

            output = model(input, target)
            #output = output.reshape(-1, len(vocab))

            exp_output = UtilityRNN.encodeTarget(exp_output, vocab).to(device)
            loss = training_loss(output, exp_output)          
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.zero_grad()
            #scheduler.step()
            if epoch % 1 == 0:
                output_max = torch.argmax(output, 2)
                exp_output_max = torch.argmax(exp_output, 2)
                accuracies.append(np.average(np.average([[1 if exp_output_max[i][j].tolist() == y.tolist() else 0 for j, y in enumerate(x)] for i, x in enumerate(output_max)], 0)))

            if numBatch % 100 == 0:
                expOutputChar = UtilityRNN.decodeChar(exp_output, vocab)
                outputChar = UtilityRNN.decodeChar(output, vocab)
                print('Epoch:{}, Batch number: {}, Expected Output: {}, Output: {}, Loss: {}, Accuracy: {}'.format(epoch, numBatch, expOutputChar[:][0], outputChar[:][0], loss, np.average(accuracies)))

def training_loss(output, exp_output):
    criterion = nn.BCELoss()
    loss = criterion(output, exp_output)
    return loss         















def main(): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text = read_dataset()
    uniqueWords = UtilityRNN.getUniqueWords(text)
    vocab = UtilityRNN.assignIndex(uniqueWords)
    text = UtilityRNN.splitText(text)

    encIn_seq_len = 10
    decIn_seq_len = 25
    seqIndex = UtilityRNN.text2index(text, encIn_seq_len, decIn_seq_len, vocab, device)
    train_ds, val_ds, test_ds = UtilityRNN.dataloader(seqIndex, 0.2, 0.2, 128, device, shuffle=True)

    embedding_size = 512
    src_vocab_size = len(vocab)
    tgt_vocab_size = len(vocab)
    num_epochs = 100

    batch_size = 128
    model = modelTransformer(src_vocab_size, embedding_size, tgt_vocab_size, decIn_seq_len+1, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0005)
 
    TrainingLoop(num_epochs, model, optimizer, text, vocab, lookUpTable, encIn_seq_len, decIn_seq_len, batch_size, device)

if __name__ == '__main__':
    main()

