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
from torch import are_deterministic_algorithms_enabled, optim
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
                 n_heads,
                 num_encoder_layers,
                 num_decoder_layers,
                 dropout):

        self.embedding_size = embedding_size
        super(modelTransformer, self).__init__()
        self.embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.positional_encoding = PositionalEncoding(embedding_size)
        self.transformer = nn.Transformer(embedding_size, 
                                          nhead=n_heads, 
                                          num_encoder_layers=num_encoder_layers, 
                                          num_decoder_layers=num_decoder_layers, 
                                          dropout=dropout)
        self.fc_out = nn.Linear(embedding_size, tgt_vocab_size)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, src, tgt, src_mask, tgt_mask):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)
        out = self.transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
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
        encoder_input = torch.LongTensor([seqIndex[i][0] for i in range(len(seqIndex))]).to(device)
        decoder_intput = torch.LongTensor([seqIndex[i][1] for i in range(len(seqIndex))]).to(device)
        expected_output = torch.LongTensor([seqIndex[i][2] for i in range(len(seqIndex))]).to(device)
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

    def decodeChar(vector, vocab):
        if len(vector.shape) == 3:
            vector = vector.permute(1, 0, 2)
            vector = torch.argmax(vector, 2)
        key_list = list(vocab)
        decodedVec = [[key_list[index] for index in batch] for batch in vector]
        #decodedVec = np.reshape(decodedVec, (vector.shape[0], vector.shape[1])).T
        return decodedVec



class BaseTransformer():
    def generate_square_subsequent_mask(sz, device):
        mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
        #mask.to(device)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_mask(src, tgt, device):
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]
        tgt_mask = BaseTransformer.generate_square_subsequent_mask(tgt_seq_len, device)
        src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool)
        #src_padding_mask = (src == PAD_IDX).transpose(0, 1)
        #tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
        return src_mask, tgt_mask #, src_padding_mask, tgt_padding_mask

    def train(model, train_batch, optimizer, scheduler, criterion, device):
        encoder_input = train_batch[0]
        decoder_input = train_batch[1]
        expected_output_reshaped = train_batch[3]

        src_mask, tgt_mask = BaseTransformer.create_mask(encoder_input, decoder_input, device)
        output = model(encoder_input, decoder_input, src_mask, tgt_mask)
        loss = criterion(output.reshape(-1, output.shape[-1]), expected_output_reshaped)
        optimizer.zero_grad()     
        loss.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        # scheduler.step()
        return loss, output

    def evaluate(model, val_ds, criterion):
        encoder_input = val_ds[0]
        decoder_input = val_ds[1]
        expected_output_reshaped = val_ds[3]
        output = model(encoder_input, decoder_input)
        loss = criterion(output.reshape(-1, output.shape[-1]), expected_output_reshaped)
        acc = BaseTransformer.get_accuracy()
        return loss, acc

    def get_accuracy():
        acc = None
        return acc



def TrainingLoop(num_epochs, model, optimizer, train_ds, val_ds, vocab):
    model.train()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        for numBatch in range(len(train_ds)):
            train_loss, output = BaseTransformer.train(model, train_ds[numBatch], optimizer, scheduler, criterion, device)
            if numBatch % 1 == 0:
                #val_loss, val_acc = BaseTransformer.evaluate(model, val_ds, criterion)
                expOutputChar = UtilityRNN.decodeChar(train_ds[numBatch][2], vocab)
                outputChar = UtilityRNN.decodeChar(output, vocab)
                print('Epoch:{}, Batch number: {}, Expected Output: {}, Output: {}, train_loss: {}, val_loss: {}, accuracy: {}'
                .format(epoch, numBatch, expOutputChar[0], outputChar[0], train_loss, 0, 0))

            #if epoch % 1 == 0:
            #    output_max = torch.argmax(output, 2)
            #    exp_output_max = torch.argmax(exp_output, 2)
            #    accuracies.append(np.average(np.average([[1 if exp_output_max[i][j].tolist() == y.tolist() else 0 for j, y in enumerate(x)] for i, x in enumerate(output_max)], 0)))




def main():
    # define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load and prepare data
    text = UtilityRNN.read_dataset()
    uniqueWords = UtilityRNN.getUniqueWords(text)
    vocab = UtilityRNN.assignIndex(uniqueWords)
    text = UtilityRNN.splitText(text)

    # parameters dataloader
    encIn_seq_len = 10
    decIn_seq_len = 25
    percent_val = 0.05
    percent_test = 0.05
    batch_size = 128
    seqIndex = UtilityRNN.text2index(text, encIn_seq_len, decIn_seq_len, vocab, device)
    train_ds, val_ds, test_ds = UtilityRNN.dataloader(seqIndex, percent_val, percent_test, batch_size, device, shuffle=True)

    # paramters transformer
    embedding_size = 512
    src_vocab_size = len(vocab)
    tgt_vocab_size = len(vocab)
    n_heads = 8
    num_encoder_layers = 3
    num_decoder_layers = 3
    dropout = 0.1

    # parameters training
    num_epochs = 100

    # define model and optimizer
    model = modelTransformer(src_vocab_size, embedding_size, tgt_vocab_size, n_heads, num_encoder_layers, num_decoder_layers, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    
    # train the model
    TrainingLoop(num_epochs, model, optimizer, train_ds, val_ds, vocab)

if __name__ == '__main__':
    main()

