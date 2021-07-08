import copy
import itertools
import json
import math
import os
import pathlib
import pickle
import random
import re
import shutil
import statistics
import string
import tarfile
from collections import OrderedDict
from enum import Enum
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
from TrumpDecoder import ManagedTensorMemoryStorageMode

import gc


class TransformerBlock(nn.Module):

    def __init__(self, embedding_size, n_heads, dropout):
        super(TransformerBlock, self).__init__()
        self.masked_attention = nn.MultiheadAttention(embed_dim=embedding_size, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embedding_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, embedding_size),
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, tgt_padding_mask, tgt_attention_mask):
        out_attn, _ = self.masked_attention(query, key, value, key_padding_mask=tgt_padding_mask, attn_mask=tgt_attention_mask)
        out_norm_1 = self.norm(out_attn + query)
        out_dp_1 =  self.dropout(out_norm_1)
        out_ff = self.feed_forward(out_dp_1)
        out_norm_2 = self.norm(out_dp_1 + out_ff)
        out = self.dropout(out_norm_2)
        return out


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


class ModelTransformer(nn.Module):

    def __init__(self, tgt_vocab_size, embedding_size, n_heads, num_decoder_layers, dropout, device):
        self.embedding_size = embedding_size
        self.num_decoder_layers = num_decoder_layers
        super(ModelTransformer, self).__init__()

        self.embedding = nn.Embedding(tgt_vocab_size, embedding_size)
        self.pos_encoder = PositionalEncoding(embedding_size, dropout)
        
        self.transformer_decoder = [[] for i in range(self.num_decoder_layers)]
        for i in range(self.num_decoder_layers):
            self.transformer_decoder[i] = TransformerBlock(embedding_size, n_heads, dropout).to(device)

        self.fc_out = nn.Linear(embedding_size, tgt_vocab_size)
        self.softmax = nn.Softmax(dim=2)
        self.flatten = nn.Flatten(start_dim=0, end_dim=1) 


    def forward(self, tgt):
        out = self.embedding(tgt)
        out = self.pos_encoder(out)

        tgt_padding_mask = (tgt == self.vocab['<PAD>'])

        for i in range(self.num_decoder_layers):
            out = self.transformer_decoder[i](out, out, out, tgt_padding_mask, self.tgt_mask)

        out = self.fc_out(out)
        #out = self.softmax(out)

        out = self.flatten(out)
        return out


    def init_data(self, train_ds, val_ds, vocab, batch_size_train, minibatch_size, device):
        # datasets
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.vocab = vocab
        self.batch_size_train = batch_size_train
        self.minibatch_size = minibatch_size
        self.device = device
        self.criterion = nn.CrossEntropyLoss(ignore_index = self.vocab['<PAD>'])
        # model inputs
        self.tgt_seq_len = train_ds[0]['decoder_input'][0].tensor.shape[-1]
        self.tgt_mask = self.generate_square_subsequent_mask(self.tgt_seq_len, self.tgt_seq_len)

    def generate_square_subsequent_mask(self, tgt_seq_len, src_seq_len):
        mask = (torch.triu(torch.ones((tgt_seq_len, src_seq_len))) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(self.device)


    def train_batch(self, train_batch, optimizer, scheduler):
        optimizer.zero_grad()

        total_loss = 0

        for i in range(self.minibatch_size):
            with train_batch['decoder_input'][i] as train_batch_decoder_input, train_batch['expected_output_flat'][i] as train_batch_expected_output_flat:
                output = self(train_batch_decoder_input.tensor)
                loss = self.criterion(output, train_batch_expected_output_flat.tensor) # CrossEntropy
                total_loss += loss.detach().item()
                loss.backward()
                output = output.detach()
                loss = loss.detach()
        
        del loss
        torch.cuda.empty_cache()

        optimizer.step()
        return total_loss / self.minibatch_size, output
        


    def evaluate(self):
        val_loss = []
        val_acc = []

        for i in range(len(self.val_ds)):
            total_loss = 0
            total_acc = 0
                
            for j in range(self.minibatch_size):
                with self.val_ds[i]['decoder_input'][j] as val_ds_decoder_input, self.val_ds[i]['expected_output_flat'][j] as val_ds_expected_output_flat:
                    output = self(val_ds_decoder_input.tensor)
                    loss = self.criterion(output, val_ds_expected_output_flat.tensor) # Crossentropy
                    total_loss += loss.item()
                    total_acc += self.get_accuracy(output, val_ds_expected_output_flat.tensor)
                    del output, loss
                    torch.cuda.empty_cache()

            val_loss.append(total_loss / self.minibatch_size)
            val_acc.append(total_acc / self.minibatch_size)

        return np.average(val_loss), np.average(val_acc)


    def get_accuracy(self, output, expected_output):
        output = torch.argmax(output, 1).reshape(-1)
        no_pad_expected_output = expected_output[expected_output != self.vocab['<PAD>']]
        no_pad_output = output[expected_output != self.vocab['<PAD>']]
        acc = sum(no_pad_output == no_pad_expected_output).item() / len(no_pad_expected_output)
        return acc


    def start_training(self, num_epochs, optimizer, scheduler):

        for epoch in range(num_epochs):
            for num_batch in range(len(self.train_ds)):
                self.train()
                train_loss, output = self.train_batch(self.train_ds[num_batch], optimizer, scheduler)

                if num_batch == len(self.train_ds) - 1:
                    self.eval()
                    val_loss, val_acc = self.evaluate()
                    exp_output_char = UtilityTextProcessing.decode_char(self.train_ds[num_batch]['expected_output'][-1].tensor, self.vocab)
                    output_reshaped = UtilityTextProcessing.reshape_output(output, self.batch_size_train, self.minibatch_size, self.tgt_seq_len)
                    output_char = UtilityTextProcessing.decode_char(output_reshaped, self.vocab)

                    print('Epoch:{}, Batch number: {}\n\nExpected Output: {}\n\nOutput: {}\n\ntrain_loss: {}, val_loss: {}, accuracy: {}\n\n\n'
                        .format(epoch, num_batch, exp_output_char[0], output_char[0], train_loss, val_loss, val_acc))
                
                del output
                torch.cuda.empty_cache()
                gc.collect()


class UtilityTextProcessing():

    def read_dataset():
        data_type = 'hotel_data'
        current_path = pathlib.Path().absolute()
        files_names = os.listdir(os.path.join(current_path, 'opin_dataset\\', data_type))
        file_all = []

        for file_name in files_names:
            with open(os.path.join(current_path, 'opin_dataset\\', data_type, file_name), 'r') as file:
                if data_type == 'car_data':
                    file = file.read().replace('\n', '')
                    file = file.lower()
                    file = re.sub(r'(.*?)/(.*?)', r'\1 / \2', file)
                    file = re.sub(r'(.*?)-(.*?)', r'\1 - \2', file)
                    matches = re.findall(r'<TEXT>.+?</TEXT>', file)
                    textfile = []
                    for sublist in matches:
                        textfile.append(re.sub(r'<TEXT>|</TEXT>', '', sublist))
                        
                elif data_type == 'hotel_data':
                    file = file.read().replace('\t', '')
                    file = file.lower()
                    file = re.sub(r'(.*?)/(.*?)', r'\1 / \2', file)
                    file = re.sub(r'(.*?)-(.*?)', r'\1 - \2', file)
                    textfile = file.split('\n')
                
                

            file_all = file_all + textfile
        return file_all


    def get_unique_words(text):
        unique_words = []
        words = [nltk.word_tokenize(sub_text) for sub_text in text]

        for sub_text in words:
            for word in sub_text:
                if word not in unique_words: unique_words.append(word)

        return words, unique_words


    def assign_index(words, unique_words):
        vocab = {'<SOS>': 0, '<EOS>': 1, '<PAD>': 2} #{'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        for word in unique_words: vocab[word] = len(vocab)
        words_index = [[vocab[word] for word in sub_text] for sub_text in words]
        return words_index, vocab


    def dataloader(words_index, percent_val, batch_size_train, batch_size_val, minibatch_size, device, vocab, shuffle=True):

        # shuffle
        if shuffle: random.shuffle(words_index)

        # extract decoder input, expected output
        decoder_input = copy.deepcopy(words_index)
        expected_output = copy.deepcopy(words_index)

        for i in range(len(words_index)): 
            decoder_input[i].insert(0, vocab['<SOS>'])
            expected_output[i].append(vocab['<EOS>'])

         # find max len sentence and fill rest of the sentences with padding token
        max_len = len(max(decoder_input, key=len))
        
        for i in range(len(words_index)):
            num_pad = max_len - len(decoder_input[i])
            decoder_input[i] = decoder_input[i]+([vocab['<PAD>']]*num_pad)
            expected_output[i] = expected_output[i]+([vocab['<PAD>']]*num_pad)


        decoder_input = torch.LongTensor(decoder_input).to(device)
        expected_output = torch.LongTensor(expected_output).to(device)

        # get index for splitting into train, val
        idx_split_1 = math.ceil(math.ceil(len(words_index)*(1-percent_val))/batch_size_train)*batch_size_train
        idx_split_2 = idx_split_1 + math.floor(len(words_index[idx_split_1:-1])/batch_size_val)*batch_size_val

        # split into batches for training dataset
        num_batches_train = idx_split_1 // batch_size_train
        dec_in_train = decoder_input[:idx_split_1].reshape(num_batches_train, minibatch_size, batch_size_train//minibatch_size, -1)
        exp_out_train = expected_output[:idx_split_1].reshape(num_batches_train, minibatch_size, batch_size_train//minibatch_size, -1)

        # split into batches for validation dataset
        num_batches_val = (idx_split_2 - idx_split_1) // batch_size_train
        dec_in_val = decoder_input[idx_split_1:idx_split_2].reshape(num_batches_val, minibatch_size, batch_size_val//minibatch_size, -1)
        exp_out_val = expected_output[idx_split_1:idx_split_2].reshape(num_batches_val, minibatch_size, batch_size_val//minibatch_size, -1)

        # get train, val and test
        train_ds = [{ 
                'decoder_input': [ManagedTensor(dec_in_train[i][j], ManagedTensorMemoryStorageMode.CPU) for j in range(minibatch_size)], 
                'expected_output': [ManagedTensor(exp_out_train[i][j], ManagedTensorMemoryStorageMode.CPU) for j in range(minibatch_size)], 
                'expected_output_flat': [ManagedTensor(exp_out_train[i][j].reshape(-1), ManagedTensorMemoryStorageMode.CPU) for j in range(minibatch_size)], 
            } for i in range(num_batches_train)
        ]
            
        val_ds = [{ 
                'decoder_input': [ManagedTensor(dec_in_val[i][j], ManagedTensorMemoryStorageMode.CPU) for j in range(minibatch_size)], 
                'expected_output': [ManagedTensor(exp_out_val[i][j], ManagedTensorMemoryStorageMode.CPU) for j in range(minibatch_size)], 
                'expected_output_flat': [ManagedTensor(exp_out_val[i][j].reshape(-1), ManagedTensorMemoryStorageMode.CPU) for j in range(minibatch_size)], 
            } for i in range(num_batches_val)
        ]

        return train_ds, val_ds


    def process_text(percent_val, batch_size_train, batch_size_val, minibatch_size, device):
        if os.path.isfile('processed_text.pkl') and os.path.isfile('vocab.pkl'):
            file_1 = open('vocab.pkl', 'rb')
            vocab = pickle.load(file_1)

            file_2 = open('processed_text.pkl', 'rb')
            words_index = pickle.load(file_2)

        else:
            text = UtilityTextProcessing.read_dataset()
            words, unique_words = UtilityTextProcessing.get_unique_words(text)
            words_index, vocab = UtilityTextProcessing.assign_index(words, unique_words)

            with open("vocab.pkl", "wb") as file_1:
                pickle.dump(vocab, file_1)

            with open("processed_text.pkl", "wb") as file_2:
                pickle.dump(words_index, file_2)


        train_ds, val_ds = UtilityTextProcessing.dataloader(words_index, percent_val, batch_size_train, batch_size_val, minibatch_size, device, vocab, shuffle=True)
        return train_ds, val_ds, vocab


    def reshape_output(output, batch_size, minibatch_size, tgt_seq_len):
        # reshape from  [batch_size x sequence_length, vocab_size]
        # to            [batch_size, sequence_length]
        output = torch.argmax(output, 1)
        output = output.reshape(batch_size//minibatch_size, tgt_seq_len)
        return output

    def decode_char(vector, vocab):
        # input vector shape: [batch_size, sequence_length]
        key_list = list(vocab)
        decoded_vec = [[key_list[index] for index in batch] for batch in vector]
        return decoded_vec


    def encode_target(vector, vocab):
        # check dimensions of vector
        encodedVec = torch.zeros(vector.shape[0], vector.shape[1], len(vocab))
        for batch in range(vector.shape[0]):
            for entry in range(vector.shape[1]): 
                encodedVec[batch][entry][vector[batch][entry]] = 1

        return encodedVec


#    def plot_attention_weights(model_input_decoded, model_output_decoded, attention_head_weights):
#        in_tokens = tf.convert_to_tensor([sentence])
#        in_tokens = tokenizers.pt.tokenize(in_tokens).to_tensor()
#        in_tokens = tokenizers.pt.lookup(in_tokens)[0]
#        in_tokens
#
#        fig = plt.figure(figsize=(16, 8))
#        for h, head in enumerate(attention_head_weights):
#            ax = fig.add_subplot(2, 4, h+1)
#
#            plot_attention_head(in_tokens, translated_tokens, head)
#
#            ax.set_xlabel(f'Head {h+1}')
#
#       plt.tight_layout()
#        plt.show()


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
        self.allow_autoconvert = False
        if (tensor != None): self.tensor = tensor
        ManagedTensor.instances.append(self)
     
    def __get_storage_device(self):
        return ManagedTensor.device_map[self.storage_mode.value]

    @property
    def tensor(self):
        if (self.allow_autoconvert and self._tensor != None and self._tensor.device.type != ManagedTensor.device_default): 
            self._tensor = self._tensor.to(ManagedTensor.device_default)

        return self._tensor

    @tensor.setter
    def tensor(self, value):
        self._tensor = value
        self.__move_to_storage() # Auto-convert tensor to device

    def __move_to_storage(self):
        if (self._tensor != None and self._tensor.device.type != self.__get_storage_device()): 
            self._tensor = self._tensor.to(self.__get_storage_device())

    def __enter__(self):
        self.allow_autoconvert = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.allow_autoconvert = False
        self.__move_to_storage()


def main():
    # define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ManagedTensor.init(device)  

    # parameters dataloader
    #dec_in_seq_len = 25    # not needed since sequence length is now length of longest sentence
    percent_val = 0.2
    batch_size_train = 128
    batch_size_val = 128
    minibatch_size = 16
    train_ds, val_ds, vocab = UtilityTextProcessing.process_text(percent_val, batch_size_train, batch_size_val, minibatch_size, device)
    
    # define model
    embedding_size = 1024
    tgt_vocab_size = len(vocab)
    n_heads = 8
    num_decoder_layers = 12
    dropout = 0.002
    model = ModelTransformer(tgt_vocab_size, embedding_size, n_heads, num_decoder_layers, dropout, device).to(device)
    model.init_data(train_ds, val_ds, vocab, batch_size_train, minibatch_size, device)

    # train the model
    num_epochs = 100
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    model.start_training(num_epochs, optimizer, scheduler)


if __name__ == '__main__':
    main()


