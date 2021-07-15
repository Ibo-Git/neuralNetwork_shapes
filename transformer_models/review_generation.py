import copy
import gc
import itertools
import json
from json import decoder
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
import time
from collections import OrderedDict
from enum import Enum, unique
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
from langdetect import detect
from spellchecker import SpellChecker
from torch import are_deterministic_algorithms_enabled, optim
from torch._C import device
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.utils.rnn import pad_sequence
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


class TransformerBlock(nn.Module):

    def __init__(self, embedding_size, n_heads, dropout):
        super(TransformerBlock, self).__init__()
        self.masked_attention = nn.MultiheadAttention(embed_dim=embedding_size, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm_1 = nn.LayerNorm(embedding_size)
        self.norm_2 = nn.LayerNorm(embedding_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, embedding_size),
        )
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    
    def forward(self, query, key, value, tgt_padding_mask, tgt_attention_mask):
        out_attn, attn_head_weights = self.masked_attention(query, key, value, key_padding_mask=tgt_padding_mask, attn_mask=tgt_attention_mask)
        out_norm_1 = self.norm_1(out_attn + query)
        out_dp_1 =  self.dropout_1(out_norm_1)
        out_ff = self.feed_forward(out_dp_1)
        out_norm_2 = self.norm_2(out_dp_1 + out_ff)
        out = self.dropout_2(out_norm_2)
        return out, attn_head_weights


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
        
        #self.transformer_decoder = [[] for i in range(self.num_decoder_layers)]
        #for i in range(self.num_decoder_layers):
        #    self.transformer_decoder[i] = TransformerBlock(embedding_size, n_heads, dropout).to(device)
        decoder_layer = nn.TransformerEncoderLayer(embedding_size, n_heads, dim_feedforward=2048, dropout=dropout, batch_first=True, device=device)
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_decoder_layers)

        self.fc_out = nn.Linear(embedding_size, tgt_vocab_size)
        self.softmax = nn.Softmax(dim=2)
        self.flatten = nn.Flatten(start_dim=0, end_dim=1) 


    def forward(self, tgt):
        out = self.embedding(tgt)
        out = self.pos_encoder(out)
        # model inputs
        tgt_seq_len = tgt.shape[1]
        tgt_attn_mask = self.generate_square_subsequent_mask(tgt_seq_len, tgt_seq_len)
        tgt_padding_mask = (tgt == self.vocab['<PAD>'])

        out = self.transformer_decoder(out, tgt_attn_mask, tgt_padding_mask)

        #attn_head_weights_all = {}
        #for i in range(self.num_decoder_layers):
        #    out, attn_head_weights = self.transformer_decoder[i](out, out, out, tgt_padding_mask, tgt_attn_mask)
        #    attn_head_weights_all['decoder_layer_'+str(i+1)] = attn_head_weights
        #    #UtilityTextProcessing.plot_attention_head(tgt, attn_head_weights_all['decoder_layer_1'][1].detach(), vocab=self.vocab)
       
        out = self.fc_out(out)

        out = self.flatten(out)
        return out#, attn_head_weights_all


    def init_data(self, train_dl, val_dl, vocab, batch_size_train, batch_size_val, minibatch_size, device):
        # datasets
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.vocab = vocab
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.minibatch_size = minibatch_size
        self.device = device
        self.criterion = nn.CrossEntropyLoss(ignore_index = self.vocab['<PAD>'])


    def generate_square_subsequent_mask(self, tgt_seq_len, src_seq_len):
        mask = (torch.triu(torch.ones((tgt_seq_len, src_seq_len))) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(self.device)


    def train_batch(self, decoder_input, expected_output_flat, optimizer, scheduler):
        optimizer.zero_grad()

        total_loss = 0

        for i in range(self.batch_size_train//self.minibatch_size):
            with decoder_input[i] as decoder_input_mb, expected_output_flat[i] as expected_output_flat_mb:
                output = self(decoder_input_mb.tensor)
                loss = self.criterion(output, expected_output_flat_mb.tensor) # CrossEntropy
                total_loss += loss.detach().item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)

                output = output.detach()
                loss = loss.detach()
        
        del loss
        del output
        torch.cuda.empty_cache()

        optimizer.step()

        return total_loss / (self.batch_size_train/self.minibatch_size)
        


    def evaluate(self):
        val_loss = []
        val_acc = []
        num_minibatches = self.batch_size_val//self.minibatch_size

        for num_batch, (decoder_input, expected_output, expected_output_flat) in enumerate(self.val_dl):
            total_loss = 0
            total_acc = 0
                
            for num_minibatch in range(num_minibatches):
                with decoder_input[num_minibatch] as decoder_input_mb,  expected_output_flat[num_minibatch] as expected_output_flat_mb:
                    output = self(decoder_input_mb.tensor)
                    loss = self.criterion(output, expected_output_flat_mb.tensor) # Crossentropy
                    total_loss += loss.item()
                    total_acc += self.get_accuracy(output, expected_output_flat_mb.tensor)

                    if num_batch == len(self.val_dl)-1 and num_minibatch == num_minibatches-1:
                        del loss
                    else:
                        del output, loss

                    torch.cuda.empty_cache()

            val_loss.append(total_loss / num_minibatches)
            val_acc.append(total_acc / num_minibatches)

        self.log_val(
            torch.argmax(output, 1)[decoder_input_mb.tensor.shape[1]:], 
            expected_output[-1].tensor[-1],
            np.average(val_loss), np.average(val_acc)
        )

        del output


    def get_accuracy(self, output, expected_output):
        output = torch.argmax(output, 1).reshape(-1)
        no_pad_expected_output = expected_output[expected_output != self.vocab['<PAD>']]
        no_pad_output = output[expected_output != self.vocab['<PAD>']]
        acc = sum(no_pad_output == no_pad_expected_output).item() / len(no_pad_expected_output)
        return acc

    def log_val(self, output, expected_output, val_loss, val_acc):
        
        key_list = list(self.vocab)
        exp_output_char = [key_list[index] for index in expected_output if key_list[index] != '<PAD>']
        output_char = [key_list[index] for index in output]

        output_char = output_char[:len(exp_output_char)]

        print('Expected Output: {}\n\nOutput: {}\n\n, val_loss: {}, accuracy: {}\n\n\n'
            .format(exp_output_char, output_char, val_loss, val_acc))


    def start_training(self, num_epochs, optimizer, scheduler):

        for epoch in range(num_epochs):
            for num_batch, (decoder_input, _, expected_output_flat) in enumerate(self.train_dl):
                self.train()
                train_loss = self.train_batch(decoder_input, expected_output_flat, optimizer, scheduler)

                if num_batch == len(self.train_dl) - 1:
                    print('Epoch:{}, Batch number: {}, train_loss: {}\n\n'
                        .format(epoch, num_batch, train_loss))
                    self.eval()
                    self.evaluate()

                torch.cuda.empty_cache()
                gc.collect()



class UtilityTextProcessing():

    def read_review():
        data_type = 'hotel_data'
        current_path = pathlib.Path().absolute()
        files_names = os.listdir(os.path.join(current_path, 'opin_dataset\\', data_type))
        file_all = []

        for file_name in files_names:
            with open(os.path.join(current_path, 'opin_dataset\\', data_type, file_name), 'r') as file:
                file = file.read()
                file = file.lower()
                file = re.sub(r'(.*?)/(.*?)', r'\1 / \2', file)
                file = re.sub(r'(.*?)-(.*?)', r'\1 - \2', file)

                if data_type == 'car_data':
                    file = file.replace('\n', '')
                    matches = re.findall(r'<TEXT>.+?</TEXT>', file)
                    textfile = []
                    for sublist in matches:
                        textfile.append(re.sub(r'<TEXT>|</TEXT>', '', sublist))
                        
                elif data_type == 'hotel_data':
                    file = file.read().replace('\t', '')
                    file = file.lower()
                    #file = re.sub(r'([\w\d]+)\/([\w\d]+)\g', r'\1 / \2', file)
                    #file = re.sub(r'([\w\d]+)\-([\w\d]+)\g', r'\1 - \2', file)
                    #file = re.sub(r'([\w]+)([\-\/\./]){1}([\w]+)', r'\1 \2 \3', file)
                    file = re.sub(r'([\-\/\.—]){1}([\w]+)', r'\1 \2', file)
                    file = re.sub(r'([\w]+)([\-\/\.—]){1}', r'\1 \2', file)
                    file = re.sub(r'([\d]+)([a-zA-Z]+)', r'\1 \2', file)
                    file = re.sub(r'([a-zA-Z]+)([\d]+)', r'\1 \2', file)
                    textfile = list(filter(lambda x: len(x.strip()) > 0 and len(x) < 600, file.split('\n')))
                    textfile = list(filter(lambda x: detect(x) == 'en', textfile))

                    if file_name == 'china_beijing_oakwood_residence_beijing.txt':
                        break

            file_all = file_all + textfile
        return file_all

    def read_trump():
        current_path = pathlib.Path().absolute()
        files_names = os.listdir(os.path.join(current_path, 'trump\\originals'))
        file_all = []
        sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

        for file_name in files_names:
            with open(os.path.join(current_path, 'trump\\originals', file_name), 'r', encoding="UTF-8") as file:
                file = file.read().replace('\n', '')
                file = file.lower()
                file = re.sub('…', '.', file)
                file = file.lower()
                sentences = sent_detector.tokenize(file.strip())

            file_all = file_all + sentences
        
        return file_all



    def get_unique_words(text):
        unique_words = set()
        unique_words_indices = {}
        words = [nltk.word_tokenize(sub_text) for sub_text in text]
        
        for (i, sub_text) in enumerate(words):
            for (j, word) in enumerate(sub_text):
                if word not in unique_words: unique_words.add(word)

                if word not in unique_words_indices: unique_words_indices[word] = [(i, j)]
                else: unique_words_indices[word].append((i, j))

        
        spell = SpellChecker()
        spell.known('n\'t')
        misspelled = spell.unknown(unique_words)

        for word in misspelled:
            correctedWord = word #spell.correction(word)

            if correctedWord == word: 
                correctedWord = '<UNK>'

            # Update unique_words
            unique_words.remove(word)
            if correctedWord not in unique_words: unique_words.add(correctedWord)

            # Replace all occurances of word in words
            for occurance in unique_words_indices[word]:
                words[occurance[0]][occurance[1]] = correctedWord


        return words, list(unique_words)


    def assign_index(words, unique_words):
        vocab = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2} #{'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        for word in unique_words: vocab[word] = len(vocab)
        words_index = [[vocab[word] for word in sub_text] for sub_text in words]
        return words_index, vocab



    def process_text(data_type, percent_val, batch_size_train, batch_size_val, minibatch_size):
        if data_type == 'trump':
            textfile_name = 'trump_text'
            vocabfile_name = 'vocab_trump'
        elif data_type == 'review':
            textfile_name = 'review_text'
            vocabfile_name = 'vocab_review'

        if os.path.isfile((textfile_name + '.pkl')) and os.path.isfile((vocabfile_name + '.pkl')):
                file_1 = open((vocabfile_name + '.pkl'), 'rb')
                vocab = pickle.load(file_1)

                file_2 = open((textfile_name + '.pkl'), 'rb')
                words_index = pickle.load(file_2)
        else:
            if data_type == 'trump':
                text = UtilityTextProcessing.read_trump()
            elif data_type == 'review':
                text = UtilityTextProcessing.read_review()

            words, unique_words = UtilityTextProcessing.get_unique_words(text)
            words_index, vocab = UtilityTextProcessing.assign_index(words, unique_words)
            with open((vocabfile_name + '.pkl'), "wb") as file_1:
                pickle.dump(vocab, file_1)

            with open((textfile_name + '.pkl'), "wb") as file_2:
                pickle.dump(words_index, file_2)

        idx_split_1 = math.floor(math.floor(len(words_index)*(1-percent_val))/batch_size_train)*batch_size_train
        idx_split_2 = idx_split_1 + math.floor(len(words_index[idx_split_1:-1])/batch_size_val)*batch_size_val

        train_ds = words_index[:idx_split_1]
        val_ds = words_index[idx_split_1:idx_split_2]


        train_ds = CustomDataset(train_ds, vocab, minibatch_size)
        val_ds = CustomDataset(val_ds, vocab, minibatch_size)

        train_dl = DataLoader(dataset=train_ds, batch_size=batch_size_train, shuffle=True, collate_fn=collate_fn, pin_memory=True, num_workers=4)
        val_dl = DataLoader(dataset=val_ds, batch_size=batch_size_val, shuffle=True, collate_fn=collate_fn, pin_memory=True, num_workers=4)

        return train_dl, val_dl, vocab


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
#        plt.tight_layout()
#        plt.show()

    def plot_attention_head(in_tokens, attention, vocab):

        sequence_length = 10
        ax = plt.gca()
        attention = attention[0:sequence_length, 0:sequence_length]
        ax.matshow(attention)
        ax.set_xticks(range(sequence_length))
        ax.set_yticks(range(sequence_length))

        labels = UtilityTextProcessing.decode_char(in_tokens, vocab)
        ax.set_xticklabels(labels[1][0:sequence_length], rotation=90)
        ax.set_yticklabels(labels[1][0:sequence_length],)



class CustomDataset(Dataset):

    def __init__(self, dataset, vocab, minibatch_size):
        self.dataset = dataset
        self.decoder_input = [(torch.cat((torch.tensor([vocab['<SOS>']]), torch.tensor(dataset[i])))) for i in range(len(dataset))]
        self.expected_output = [torch.cat((torch.tensor(dataset[i]), torch.tensor([vocab['<EOS>']]))) for i in range(len(dataset))]
        self.vocab = vocab
        self.minibatch_size = minibatch_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        decoder_input = self.decoder_input[idx]
        expected_output = self.expected_output[idx]
        vocab = self.vocab
        minibatch_size = self.minibatch_size

        return decoder_input, expected_output, vocab, minibatch_size


def collate_fn(batch):
    ManagedTensor.init(device)  
    decoder_input, expected_output = [], []

    for temp_decoder_input, temp_expected_output, vocab, minibatch_size in batch:
        decoder_input.append(temp_decoder_input)
        expected_output.append(temp_expected_output)

    decoder_input = pad_sequence(decoder_input, batch_first=True, padding_value = vocab['<PAD>'])
    expected_output = pad_sequence(expected_output, batch_first=True, padding_value = vocab['<PAD>'])
    expected_output_flat = expected_output.reshape(-1)
    
    batch_size = decoder_input.shape[0]
    reshape_size = (batch_size//minibatch_size, minibatch_size, -1)
    decoder_input = [ManagedTensor(decoder_input.reshape(reshape_size)[j], ManagedTensorMemoryStorageMode.CPU) for j in range(batch_size//minibatch_size)]
    expected_output_flat = [ManagedTensor(expected_output.reshape(reshape_size)[j].reshape(-1), ManagedTensorMemoryStorageMode.CPU) for j in range(batch_size//minibatch_size)]
    expected_output = [ManagedTensor(expected_output.reshape(reshape_size)[j], ManagedTensorMemoryStorageMode.CPU) for j in range(batch_size//minibatch_size)]

    return  decoder_input, expected_output, expected_output_flat



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
        self._tensor.pin_memory()
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
    percent_val = 0.1 #0.05
    batch_size_train = 128
    batch_size_val = 128
    minibatch_size = 8
    data_type = 'trump' # 'trump' / 'review'
    train_dl, val_dl, vocab = UtilityTextProcessing.process_text(data_type, percent_val, batch_size_train, batch_size_val, minibatch_size)
    
    #with open('train_ds_128_1.pkl', 'rb') as file_1:
    #    train_ds = pickle.load(file_1)

    #with open('val_ds_128_1.pkl', 'rb') as file_2:
    #    val_ds = pickle.load(file_2)

    # define model
    embedding_size = 512
    tgt_vocab_size = len(vocab)
    n_heads = 8
    num_decoder_layers = 12
    dropout = 0
    model = ModelTransformer(tgt_vocab_size, embedding_size, n_heads, num_decoder_layers, dropout, device).to(device)
    model.init_data(train_dl, val_dl, vocab, batch_size_train, batch_size_val, minibatch_size, device)

    # train the model
    num_epochs = 100
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    model.start_training(num_epochs, optimizer, scheduler)


if __name__ == '__main__':
    main()


