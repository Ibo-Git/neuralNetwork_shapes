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
from collections import OrderedDict
from enum import Enum
from multiprocessing import Process, freeze_support
from os import listdir
from os.path import isfile, join
from typing import Sequence
import pickle
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

import csv
from phoneme_model import ModelTransformer, Training
from RNN import EncoderRNN, DecoderRNN, AttnDecoderRNN, train, evaluate


class UtilityRNN():
    def read_dataset(filename):
        current_path = pathlib.Path().absolute()

        text = []
        with open(filename, encoding="utf8") as file:
                file = csv.reader(file, delimiter="\t")

                for line in file:
                    text.append(line)  
        
        return text



    def sort_input(text_1, text_2, max_len):
        phoneme_sorted = []
        list_phoneme_input, list_words_input = zip(*text_1)
        list_phoneme_target, list_words_target = zip(*text_2)

        dict_words_input = list(list_words_input)
        list_words_target = list(list_words_target)
        list_phoneme_input = list(list_phoneme_input)
        list_phoneme_target = list(list_phoneme_target)
        
        n = 0
        while len(phoneme_sorted) < max_len:
            word = list_words_target[n]
            if word in list_words_input:
                phoneme_input = list_phoneme_input[list_words_input.index(word)]
                # if difference in phonemes length too large, skip 
                if len(phoneme_input) + 5 > len(list_phoneme_target[n]):
                    n += 1
                    phoneme_sorted.append(list_phoneme_input[list_words_input.index(word)])
                else:
                    del list_words_target[n]
                    del list_phoneme_target[n]
            else:
                del list_words_target[n]
                del list_phoneme_target[n]

        return phoneme_sorted, list_phoneme_target[0:max_len]

    def split_list(phoneme_list, words_list):
        splitted_list = []
        for i, word in enumerate(words_list):
            m = re.split(rf"({'|'.join(phoneme_list)})", word)
            m = [n for n in m if n]
            splitted_list.append(m)
        return splitted_list

    def assign_index(input, target, unique_words):
        vocab = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<TRL>':3}
        for word in unique_words: vocab[word] = len(vocab)

        index_input = [[vocab[char] for char in word] for word in input]
        index_target = [[vocab[char] for char in word] for word in target]


        return index_input, index_target, vocab

    def decode_char(batch, vocab):
        # input vector shape: [batch_size, sequence_length]
        key_list = list(vocab)
        decoded_vec = [[key_list[index] for index in sequence] for sequence in batch]
        return decoded_vec

def transform_to_tensor(index_input, index_target, batch_size, vocab, device):

    if len(max(index_target, key=len)) > len(max(index_input, key=len)):
        max_len = len(max(index_target, key=len))
    else:
        max_len = len(max(index_input, key=len))

    for i in range(len(index_input)):
        num_pad = max_len - len(index_input[i])
        index_input[i] = torch.tensor(index_input[i]+[vocab['<EOS>']]+[vocab['<PAD>']]*num_pad)
        num_pad = max_len - len(index_target[i])
        index_target[i] = torch.tensor(index_target[i]+([vocab['<PAD>']]*num_pad))

    input = index_input
    target = index_target

    return input, target

    

def main():
    batch_size = 32
    num_batches = 100
    len_string = batch_size*num_batches

    textfile_name = 'list_phoneme_input'
    vocabfile_name = 'list_phoneme_target'

    if os.path.isfile((textfile_name + '.pkl')) and os.path.isfile((vocabfile_name + '.pkl')):
        file_1 = open((vocabfile_name + '.pkl'), 'rb')
        phoneme_input = pickle.load(file_1)

        file_2 = open((textfile_name + '.pkl'), 'rb')
        phoneme_target = pickle.load(file_2)
    else:
        target = UtilityRNN.read_dataset('g2pPhonemes_PROBEARBEIT.tsv')
        input = UtilityRNN.read_dataset('modelPhonemes_PROBEARBEIT.tsv')
        phoneme_input, phoneme_target = UtilityRNN.sort_input(input, target, len_string)

        with open((vocabfile_name + '.pkl'), "wb") as file_1:
            pickle.dump(phoneme_input, file_1)

        with open((textfile_name + '.pkl'), "wb") as file_2:
            pickle.dump(phoneme_target, file_2)
    

    phoneme_list = ['T', 'i', 'I', 'e', 'E', 'y', '2', '9', '@', '6', '3', 'a', 'u', 'U', 'o', 'O', 'p', 'b', 't', 'd', 'tS', 'dZ', 'c', 'g', 'q', 'p', 'B', 'f', 'v', 's', 'z', 'S', 'Z', 'C', 'x', 'h', 'm', 'n', 'N', 'l', 'R', 'j', ':', '~', 'k', 'r', 'Y']
    phoneme_input_splitted = UtilityRNN.split_list(phoneme_list, phoneme_input)
    phoneme_target_splitted = UtilityRNN.split_list(phoneme_list, phoneme_target)
    index_input, index_target, vocab = UtilityRNN.assign_index(phoneme_input_splitted, phoneme_target_splitted, phoneme_list)    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input, target = transform_to_tensor(index_input, index_target, batch_size, vocab, device)

    split_index = int(len_string * 0.8)
    input_train = input[0:split_index]
    input_val = input[split_index:]
    target_train = target[0:split_index]
    target_val = target[split_index:]
    max_length = input[0].shape[0]
    encoder = EncoderRNN(len(input), 256, device).to(device)
    decoder = AttnDecoderRNN(256, len(vocab), device, 0.1, max_length).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<PAD>'])
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr = 0.0001)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr = 0.0001)


    for epoch in range(50):
        total_loss = 0
        for i in range(len(input_train)):
            _, _, _ = train(input_train[i], target_train[i], encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, device, vocab, max_length=max_length)
        for i in range(len(input_val)):
            loss, decoded_output_seq, decoded_expected_seq = evaluate(input_val[i], target_val[i], encoder, decoder, device, vocab, criterion, max_length=max_length) 
            total_loss += loss

        print('Epoch: {}, Loss: {}\n'.format(epoch, total_loss / i))
        print('Expected: {} \nOutput: {}'.format(decoded_expected_seq, decoded_output_seq))







    

if __name__ == '__main__':
    main()
