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








class ModelTransformer(nn.Module):
    def __init__(self, src_vocab_size, embedding_size, tgt_vocab_size, n_heads, num_encoder_layers, num_decoder_layers, dropout):

        self.embedding_size = embedding_size
        super(ModelTransformer, self).__init__()
        self.embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.transformer = nn.Transformer(embedding_size, nhead=n_heads, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(embedding_size, tgt_vocab_size)
        self.softmax = nn.Softmax(dim=2)

        self.max_len_pos_enc = 5000
        self.dropout_pos_enc = nn.Dropout(p = 0.2)
        pe = torch.zeros(self.max_len_pos_enc, self.embedding_size)
        position = torch.arange(0, self.max_len_pos_enc, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embedding_size, 2).float() * (-math.log(10000.0) / self.embedding_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, src, tgt):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        src = self.dropout_pos_enc(src + self.pe[:src.size(0), :])
        tgt = self.dropout_pos_enc(tgt + self.pe[:tgt.size(0), :])
        out = self.transformer(src, tgt, src_mask=self.src_mask, tgt_mask=self.tgt_mask)
        out = self.fc_out(out)
        out = self.softmax(out)
        return out

    def init_data(self, train_ds, val_ds, vocab, device):
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.vocab = vocab
        self.device = device
        #self.criterion = nn.CrossEntropyLoss() # CrossEntropy
        self.criterion = nn.BCELoss()
        self.src_seq_len = train_ds[0]['enc_train'].shape[1]
        self.tgt_seq_len = train_ds[0]['dec_train'].shape[1]
        self.src_mask = torch.zeros((self.src_seq_len, self.src_seq_len)).type(torch.bool).to(device)
        self.tgt_mask = self.generate_square_subsequent_mask(self.tgt_seq_len).to(device)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(self.device)

    def train_batch(self, train_batch, optimizer, scheduler):
        output = self(train_batch['enc_train'], train_batch['dec_train'])
        loss = self.criterion(output, train_batch['exp_out_train_encoded'])
        #loss = self.criterion(output.reshape(-1, output.shape[-1]), train_batch[3]) # CrossEntropy
        optimizer.zero_grad()     
        loss.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        # scheduler.step()
        return loss, output

    def evaluate(self):
        output = self(self.val_ds['encoder_input'], self.val_ds['decoder_input'])
        loss = self.criterion(output, self.val_ds['exp_out_train_encoded'])
        #loss = self.criterion(output.reshape(-1, output.shape[-1]), self.val_ds['expected_output_flat'])  # CrossEntropy
        acc = self.get_accuracy(output)
        return loss, acc

    def get_accuracy(self, output):
        output = torch.argmax(output, 2).reshape(-1)
        self.val_ds['expected_output_flat']
        acc = sum(self.val_ds['expected_output_flat'] == output).item() / len(self.val_ds['expected_output_flat'])
        return acc

    def start_training(self, num_epochs, optimizer, scheduler):
        self.train()

        for epoch in range(num_epochs):
            for num_batch in range(len(self.train_ds)):
                train_loss, output = self.train_batch(self.train_ds[num_batch], optimizer, scheduler)

                if num_batch % 1 == 0:
                    val_loss, val_acc = self.evaluate()
                    exp_output_char = UtilityRNN.decode_char(self.train_ds[num_batch]['exp_out_train'], self.vocab)
                    output_char = UtilityRNN.decode_char(output, self.vocab)

                    print('Epoch:{}, Batch number: {}, Expected Output: {}, Output: {}, train_loss: {}, val_loss: {}, accuracy: {}'
                        .format(epoch, num_batch, exp_output_char[0], output_char[0], train_loss, val_loss, val_acc))


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

    def get_unique_words(text):
        unique_words = []
        text = re.sub('([.,!?"()])', r' \1 ', text)
        text = re.sub('\s{2,}', ' ', text)
        text = re.sub(r'([0-9]{1}) . ([0-9]{1})', r'\1.\2', text)

        for word in text.split():
            if word not in unique_words: unique_words.append(word)

        return unique_words

    def assign_index(unique_words):
        vocab = {'<SOS>': 0, '<EOS>': 1} #{'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        for word in unique_words: vocab[word] = len(vocab)
        return vocab

    def split_text(text):
        text = re.sub('([.,!?"()])', r' \1 ', text)
        text = re.sub('\s{2,}', ' ', text)
        text = re.sub(r'([0-9]{1}) . ([0-9]{1})', r'\1.\2', text)
        text = list(text.split(" "))
        return text

    def text2index(text, enc_in_seq_len, dec_in_seq_len, vocab, device):
        # text -> sequence
        num_sequence_max = len(text) // (enc_in_seq_len + dec_in_seq_len)
        seq_text = [[[] for j in range(3)] for i in range(num_sequence_max)]

        for num_sequence in range(num_sequence_max):
            start_idx = num_sequence * (enc_in_seq_len + dec_in_seq_len)
            end_idx_enc = start_idx + enc_in_seq_len
            end_idx_dec = end_idx_enc + dec_in_seq_len
            seq_text[num_sequence][0] = text[start_idx:end_idx_enc]
            seq_text[num_sequence][1] = text[end_idx_enc:end_idx_dec]
            seq_text[num_sequence][2] = text[end_idx_enc:end_idx_dec]

        # sequence -> index
        seq_index = [[[vocab[word] for word in seqType] for seqType in seq_text[seq]] for seq in range(len(seq_text))]

        for i in range(len(seq_index)): 
            seq_index[i][1].append(vocab['<EOS>'])
            seq_index[i][2].insert(0,vocab['<SOS>'])

        return seq_index     

    def dataloader(seq_index, percent_val, percent_test, batch_size, device, vocab, shuffle=True):
        # shuffle
        if shuffle: random.shuffle(seq_index)

        # extract encoder input, decoder input, expected output
        encoder_input = torch.LongTensor([seq_index[i][0] for i in range(len(seq_index))]).to(device)
        decoder_input = torch.LongTensor([seq_index[i][1] for i in range(len(seq_index))]).to(device)
        expected_output = torch.LongTensor([seq_index[i][2] for i in range(len(seq_index))]).to(device)

        # get index for splitting into train, val, test
        idx_split_1 = math.ceil(math.ceil(len(seq_index)*(1-percent_val-percent_test))/batch_size)*batch_size
        idx_split_2 = math.ceil(len(seq_index)*(1-percent_test))

        # split into batches for training dataset
        num_batches = idx_split_1 // batch_size
        enc_train = encoder_input[:idx_split_1].reshape(num_batches, batch_size, -1)
        dec_train = decoder_input[:idx_split_1].reshape(num_batches, batch_size, -1)
        exp_out_train = expected_output[:idx_split_1].reshape(num_batches, batch_size, -1)

        # get train, val and test
        train_ds = [{ 
                'enc_train': enc_train[i], 
                'dec_train': dec_train[i], 
                'exp_out_train': exp_out_train[i], 
                'exp_out_train_flat': exp_out_train[i].reshape(-1),
                'exp_out_train_encoded': UtilityRNN.encode_target(exp_out_train[i], vocab).to(device)
            } for i in range(num_batches)
        ]
            
        val_ds = { 
            'encoder_input': encoder_input[idx_split_1:idx_split_2], 
            'decoder_input': decoder_input[idx_split_1:idx_split_2], 
            'expected_output': expected_output[idx_split_1:idx_split_2], 
            'expected_output_flat': expected_output[idx_split_1:idx_split_2].reshape(-1),
            'exp_out_train_encoded': UtilityRNN.encode_target(expected_output[idx_split_1:idx_split_2], vocab).to(device) 
        }

        test_ds = { 
            'encoder_input': encoder_input[idx_split_2:-1], 
            'decoder_input': decoder_input[idx_split_2:-1], 
            'expected_output':  expected_output[idx_split_2:-1],
            'expected_output_flat':  expected_output[idx_split_1:idx_split_2].reshape(-1) 
        }

        return train_ds, val_ds, test_ds

    def decode_char(vector, vocab):
        if len(vector.shape) == 3:
            vector = torch.argmax(vector, 2)

        key_list = list(vocab)
        decoded_vec = [[key_list[index] for index in batch] for batch in vector]
        #decoded_vec = np.reshape(decoded_vec, (vector.shape[0], vector.shape[1])).T
        return decoded_vec

    def process_text(enc_in_seq_len, dec_in_seq_len, percent_val, percent_test, batch_size, device):
        text = UtilityRNN.read_dataset()
        unique_words = UtilityRNN.get_unique_words(text)
        vocab = UtilityRNN.assign_index(unique_words)
        text = UtilityRNN.split_text(text)
        seq_index = UtilityRNN.text2index(text, enc_in_seq_len, dec_in_seq_len, vocab, device)
        train_ds, val_ds, test_ds = UtilityRNN.dataloader(seq_index, percent_val, percent_test, batch_size, device, vocab, shuffle=True)
        return train_ds, val_ds, test_ds, vocab

    def encode_target(vector, vocab):
        # check dimensions of vector
        encodedVec = torch.zeros(vector.shape[0], vector.shape[1], len(vocab))
        for batch in range(vector.shape[0]):
            for entry in range(vector.shape[1]): 
                encodedVec[batch][entry][vector[batch][entry]] = 1

        return encodedVec


from enum import Enum

class ManagedTensorMemoryStorageMode(Enum):
    DEFAULT_DEVICE = 0
    CPU = 1
    GPU = 2
    


class ManagedTensor:
    def init(device):
        ManagedTensor.instances = []
        ManagedTensor.device_cpu = 'cpu'
        ManagedTensor.device_gpu = 'cuda'
        ManagedTensor.device_default = device
        ManagedTensor.device_map = [ManagedTensor.device_default, ManagedTensor.device_cpu, ManagedTensor.device_gpu]
    
    def __init__(self, tensor=None, storage_mode:ManagedTensorMemoryStorageMode=ManagedTensorMemoryStorageMode.DEFAULT_DEVICE):
        self.storage_mode = storage_mode
        self.current_device = 'none'
        self.allow_autoconvert = False
        if (tensor != None): self.tensor = tensor
        ManagedTensor.instances.append(self)
     
    def get_storage_device(self):
        return ManagedTensor.device_map[self.storage_mode.value]

    @property
    def tensor(self):
        if (self.allow_autoconvert and self._tensor != None and self.current_device != ManagedTensor.device_default): 
            self.current_device = ManagedTensor.device_default
            self._tensor = self._tensor.to(self.current_device)

        return self._tensor

    @tensor.setter
    def tensor(self, value):
        self._tensor = value
        self.move_to_storage() # Auto-convert tensor to device

    def move_to_storage(self):
        if (self._tensor != None and self.current_device != self.get_storage_device()): 
            self.current_device = self.get_storage_device()
            self._tensor = self._tensor.to(self.current_device)

    def __enter__(self):
        self.allow_autoconvert = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.allow_autoconvert = False
        self.move_to_storage()



def main():
    # define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # parameters dataloader
    enc_in_seq_len = 10
    dec_in_seq_len = 25
    percent_val = 0.02
    percent_test = 0.01
    batch_size = 128

    # Set default device
    ManagedTensor.init(device)

    test_tensor2 = ManagedTensor(torch.Tensor([1, 2, 3, 4]))
    tester = test_tensor2.tensor
    a = 2

    cpu_stored_tensor = ManagedTensor(torch.Tensor([1, 2, 3, 4]), ManagedTensorMemoryStorageMode.CPU)

    print(cpu_stored_tensor.tensor.device)

    with cpu_stored_tensor as cpu_stored_tensor:
        tester = cpu_stored_tensor.tensor
        print(cpu_stored_tensor.tensor.device)
        test = 3

    print(cpu_stored_tensor.tensor.device)
    

    train_ds, val_ds, test_ds, vocab = UtilityRNN.process_text(enc_in_seq_len, dec_in_seq_len, percent_val, percent_test, batch_size, device)
    
    
    

    # define model
    embedding_size = 512
    src_vocab_size = len(vocab)
    tgt_vocab_size = len(vocab)
    n_heads = 8
    num_encoder_layers = 3
    num_decoder_layers = 3
    dropout = 0.1

    model = ModelTransformer(src_vocab_size, embedding_size, tgt_vocab_size, n_heads, num_encoder_layers, num_decoder_layers, dropout).to(device)
    model.init_data(train_ds, val_ds, vocab, device)

    # train the model
    num_epochs = 100
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    model.start_training(num_epochs, optimizer, scheduler)

if __name__ == '__main__':
    main()


