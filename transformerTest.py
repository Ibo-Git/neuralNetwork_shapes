#%%
import copy
import itertools
import math
import os
import pathlib
import random
import re
import string
import tarfile
from multiprocessing import Process, freeze_support
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
    def __init__(self, src_vocab_size, embedding_size, tgt_vocab_size, tgt_seq_len, device):
        self.embedding_size = embedding_size
        super(modelTransformer, self).__init__()
        self.embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.positional_encoding = PositionalEncoding(embedding_size)
        self.tgt_mask = nn.Transformer.generate_square_subsequent_mask(self, sz = tgt_seq_len).to(device)
        self.transformer = nn.Transformer(embedding_size, nhead=4, num_encoder_layers=1, num_decoder_layers=1, dropout=0)
        self.fc_out = nn.Linear(embedding_size, tgt_vocab_size)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, src, tgt):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        #src = src.reshape(-1, 1, self.embedding_size)
        #tgt = tgt.reshape(-1, 1, self.embedding_size)
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)
        #src = self.positional_encoding(src)
        #tgt = self.positional_encoding(tgt)
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
    
    #def get_batch(text, batch_size):
    #    numBatch = len(text.split()) // batch_size
    #    text = np.array(text.split(" "))
    #    batches = text[0:numBatch*batch_size].reshape(-1, batch_size)
    #    return batches
    
    #def get_batch(text, batch_size, sequence_length, numBatch):
    #    text = re.sub('([.,!?"()])', r' \1 ', text)
    #    text = re.sub('\s{2,}', ' ', text)
    #    text = re.sub(r'([0-9]{1}) . ([0-9]{1})', r'\1.\2', text)
    #    text = list(text.split(" "))
    #    text_to_split = text[numBatch*batch_size*sequence_length:(numBatch+1)*batch_size*sequence_length]
    #    batch = np.reshape(text_to_split, (batch_size, sequence_length))
    #    return batch

    def splitText(text):
        text = re.sub('([.,!?"()])', r' \1 ', text)
        text = re.sub('\s{2,}', ' ', text)
        text = re.sub(r'([0-9]{1}) . ([0-9]{1})', r'\1.\2', text)
        text = list(text.split())
        return text

    def get_batch(text, encIn_seq_len, decIn_seq_len, batch_size, numBatch):
        text_to_split = text[numBatch*batch_size*(encIn_seq_len+decIn_seq_len):(numBatch+1)*batch_size*(encIn_seq_len+decIn_seq_len)]
        encIn = []
        decIn = []
        for i in range(batch_size):
            startIdx_enc = i * (encIn_seq_len + decIn_seq_len)
            endIdx_enc = startIdx_enc + encIn_seq_len
            endIdx_dec = endIdx_enc + decIn_seq_len

            temp_encIn = text_to_split[startIdx_enc:endIdx_enc]
            temp_decIn = text_to_split[endIdx_enc:endIdx_dec]
            encIn.append(temp_encIn)
            decIn.append(temp_decIn)

        return encIn, decIn

    def encodeBatch(encIn, decIn, vocab, device):
        enc_input = [[vocab[word] for word in seq] for seq in encIn]
        dec_input = [[vocab[word] for word in seq] for seq in decIn]
        exp_outputs = copy.deepcopy(dec_input)
        for i in range(len(exp_outputs)): 
            exp_outputs[i].append(vocab['<EOS>'])
            dec_input[i].insert(0,vocab['<SOS>'])
        return torch.tensor(enc_input).to(device), torch.tensor(dec_input).to(device), torch.tensor(exp_outputs).to(device)

        

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



def TrainingLoop(num_epochs, model, optimizer, text, vocab, encIn_seq_len, decIn_seq_len, batch_size, device):
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

            loss = training_loss(output, exp_output)          
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.zero_grad()
            #scheduler.step()
            exp_output = UtilityRNN.encodeTarget(exp_output, vocab).to(device)
            if epoch % 9 == 0:
                output_max = torch.argmax(output, 2)
                exp_output_max = torch.argmax(exp_output, 2)
                accuracies.append(np.average(np.average([[1 if exp_output_max[i][j].tolist() == y.tolist() else 0 for j, y in enumerate(x)] for i, x in enumerate(output_max)], 0)))

            if numBatch % 100 == 0:
                expOutputChar = UtilityRNN.decodeChar(exp_output, vocab)
                outputChar = UtilityRNN.decodeChar(output, vocab)
                print('Epoch:{}, Batch number: {}, Expected Output: {}, Output: {}, Loss: {}, Accuracy: {}'.format(epoch, numBatch, expOutputChar[:][0], outputChar[:][0], loss, np.average(accuracies)))

def training_loss(output, exp_output):
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output.view(-1, output.shape[2]), exp_output.reshape(-1))
    return loss         





import re
import shutil
from os import listdir
from os.path import isfile, join


def prep_dataset():    
    current_path = pathlib.Path().absolute()
    files_names = os.listdir(os.path.join(current_path, 'trump\\originals'))

    if os.path.exists(os.path.join(current_path, 'trump\\prepared')):
        shutil.rmtree(os.path.join(current_path, 'trump\\prepared'))
    
    os.mkdir(os.path.join(current_path, 'trump\\prepared'))

    for i, file_name in enumerate(files_names):
        with open(os.path.join(current_path, 'trump\\originals', file_name), 'r', encoding="UTF-8") as file:
            data = file.read()
            data_segments = [x.strip() for x in re.findall(r'.*?(?=[\.\?\!])."?', data)]
            for j, data_segment in enumerate(data_segments):
                with open(os.path.join(current_path, 'trump\\prepared', str(i) + '_' + str(j) + '_' + file_name), "wt") as segment_file:
                    segment_file.write(data_segment)
            

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








def main():
    #prep_dataset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text = read_dataset()
    #text = [random. choice(string.ascii_letters) for i in range(50000)]
    #text = ' '.join(text).lower()
    #lookUpTable = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",  "u", "v", "w", "x", "y", "z"]
    uniqueWords = UtilityRNN.getUniqueWords(text)
    vocab = UtilityRNN.assignIndex(uniqueWords)
    text = UtilityRNN.splitText(text)


    embedding_size = 128
    src_vocab_size = len(vocab)
    tgt_vocab_size = len(vocab)
    num_epochs = 100
    encIn_seq_len = 5
    decIn_seq_len = 5
    batch_size = 128
    model = modelTransformer(src_vocab_size, embedding_size, tgt_vocab_size, decIn_seq_len+1, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0002)
 
    TrainingLoop(num_epochs, model, optimizer, text, vocab, encIn_seq_len, decIn_seq_len, batch_size, device)

if __name__ == '__main__':
    main()


# %%
