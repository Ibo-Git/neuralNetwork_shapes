import gc
import math
import random
import string
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from langdetect import detect
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as transforms

from TrumpDecoder import ManagedTensorMemoryStorageMode


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

class ModelTransformer(nn.Module):

    def __init__(self, src_vocab_size, embedding_size, n_heads, num_encoder_layers, dropout, device):
        self.embedding_size = embedding_size
        self.num_encoder_layers = num_encoder_layers
        super(ModelTransformer, self).__init__()

        self.embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.pos_encoder = PositionalEncoding(embedding_size, dropout)
        
        self.transformer_decoder = [[] for i in range(self.num_encoder_layers)]
        for i in range(self.num_encoder_layers):
            self.transformer_decoder[i] = TransformerBlock(embedding_size, n_heads, dropout).to(device)
        encoder_layer = nn.TransformerEncoderLayer(embedding_size, n_heads, dim_feedforward=2048, dropout=dropout, batch_first=True, device=device)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        self.fc_out = nn.Linear(embedding_size, src_vocab_size)
        self.flatten = nn.Flatten(start_dim=0, end_dim=1) 


    def forward(self, src):
        out = self.embedding(src)
        out = self.pos_encoder(out)

        # model inputs
        src_seq_len = src.shape[1]
        src_attn_mask = self.generate_square_subsequent_mask(src_seq_len, src_seq_len)
        src_padding_mask = None
        out = self.transformer_encoder(out, src_attn_mask)
        #for i in range(self.num_encoder_layers):
        #    out, _ = self.transformer_decoder[i](out, out, out, src_padding_mask, src_attn_mask)

        out = self.fc_out(out)
        out = self.flatten(out)

        return out


    def init_data(self, train_dl, val_dl, vocab, batch_size_train, batch_size_val, sequence_length, device):
        # datasets
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.vocab = vocab
        # batch sizes and sequence length
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.sequence_length = sequence_length
        # others
        self.device = device
        self.criterion = nn.CrossEntropyLoss(ignore_index = self.vocab['<PAD>'])


    def generate_square_subsequent_mask(self, src_seq_len, tgt_seq_len):
        mask = (torch.triu(torch.ones((src_seq_len, tgt_seq_len))) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(self.device)


    def train_batch(self, encoder_input, expected_output, optimizer, scheduler):
        optimizer.zero_grad()
        total_loss = 0

        with encoder_input as encoder_input_mb, expected_output as expected_output_mb:
            output = self(encoder_input_mb.tensor)
            loss = self.criterion(output, expected_output_mb.tensor) # CrossEntropy
            total_loss += loss.detach().item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)

            output = output.detach()
            loss = loss.detach()
        
        del loss
        del output
        torch.cuda.empty_cache()

        optimizer.step()

        return total_loss
        

    def evaluate(self):
        val_loss = []
        val_acc = []

        for num_batch, (encoder_input, expected_output, expected_output_flat) in enumerate(self.val_dl):
            total_loss = 0
            total_acc = 0
                
            #for num_minibatch in range(num_minibatches):
            with encoder_input as encoder_input_mb,  expected_output_flat as expected_output_flat_mb:
                output = self(encoder_input_mb.tensor)
                loss = self.criterion(output, expected_output_flat_mb.tensor) # Crossentropy
                total_loss += loss.item()
                total_acc += self.get_accuracy(output, expected_output_flat_mb.tensor)

                if num_batch == len(self.val_dl)-1:
                    del loss
                else:
                    del output, loss

                    torch.cuda.empty_cache()

            val_loss.append(total_loss)
            val_acc.append(total_acc)

        self.log_val(
            torch.argmax(output, 1).reshape(encoder_input_mb.tensor.shape)[-1], 
            expected_output.tensor[-1],
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

        print('Expected Output: {}\n\nOutput: {}\n\nval_loss: {}, accuracy: {}\n\n\n'
            .format(exp_output_char, output_char, val_loss, val_acc))


    def start_training(self, num_epochs, optimizer, scheduler):

        for epoch in range(num_epochs):
            for num_batch, (encoder_input, _, expected_output_flat) in enumerate(self.train_dl):
                self.train()
                train_loss = self.train_batch(encoder_input, expected_output_flat, optimizer, scheduler)

                if num_batch % 4 == 0:
                    print('Epoch:{}, Batch number: {}, train_loss: {}\n'
                        .format(epoch, num_batch, train_loss))
                    self.eval()
                    self.evaluate()

                torch.cuda.empty_cache()
                gc.collect()

        input = [["h", "a", "l", "l", "o", "x", "d"]]
        self.test_model(input)


    def test_model(self, input_letter_sequence):
        input = torch.tensor([[self.vocab[letter] for letter in batch] for batch in input_letter_sequence])
        output = self(input)
        output = torch.argmax(output, 1).reshape(-1)
        key_list = list(self.vocab)
        output_char = [key_list[index] for index in output]
        print(input_letter_sequence, output_char)


class UtilityTextProcessing():

    def get_unique_words(text):
        unique_words = set()
        words = text.split()
        
        for word in words:
                if word not in unique_words: unique_words.add(word)

        return words, list(unique_words)


    def assign_index(words, unique_words):
        vocab = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2}
        for word in unique_words: vocab[word] = len(vocab)
        words_index = [vocab[word] for word in words]
        return words_index, vocab


    def reshape_output(output, batch_size, sequence_length, src_seq_len):
        # reshape from  [batch_size x sequence_length, vocab_size]
        # to            [batch_size, sequence_length]
        output = torch.argmax(output, 1)
        output = output.reshape(batch_size//sequence_length, src_seq_len)
        return output


    def decode_char(batch, vocab):
        # input vector shape: [batch_size, sequence_length]
        key_list = list(vocab)
        decoded_vec = [[key_list[index] for index in sequence] for sequence in batch]
        return decoded_vec


    def decode_letter(index, vocab):
        key_list = list(vocab)
        letter = key_list[index]
        return letter


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


    def process_text(text, percent_val, batch_size_train, batch_size_val, sequence_length):
        
        words, unique_words = UtilityTextProcessing.get_unique_words(text)
        words_index, vocab = UtilityTextProcessing.assign_index(words, unique_words)

        # get datasets
        idx_split = math.floor(math.floor(len(words_index)*(1-percent_val))/batch_size_train)*batch_size_train
        train_ds = words_index[:idx_split]
        val_ds = words_index[idx_split:]
        train_ds = CustomDataset(train_ds, unique_words, vocab, sequence_length)
        val_ds = CustomDataset(val_ds, unique_words, vocab, sequence_length)

        # get datalaoder
        train_dl = DataLoader(dataset=train_ds, batch_size=batch_size_train, shuffle=True, collate_fn=UtilityCustomDataloader.collate_fn, pin_memory=True)
        val_dl = DataLoader(dataset=val_ds, batch_size=batch_size_val, shuffle=True, collate_fn=UtilityCustomDataloader.collate_fn, pin_memory=True)

        return train_dl, val_dl, vocab


class UtilityCustomDataloader():
    def text_transform(dec_input, exp_output):
        encoder_input = torch.tensor(dec_input)
        expected_output = torch.tensor(exp_output)
        return encoder_input, expected_output


    def collate_fn(batch):
        encoder_input, expected_output = [], []
        batch_1 = [batch[idx][0] for idx in range(len(batch))]
        batch_2 = [batch[idx][1] for idx in range(len(batch))]

        for _, _, vocab, sequence_length in batch:
            sequence_length = sequence_length
            vocab = vocab
            break

        temp_encoder_input, temp_expected_output = UtilityCustomDataloader.text_transform(batch_1, batch_2)
        encoder_input.append(temp_encoder_input)
        expected_output.append(temp_expected_output)

        encoder_input = pad_sequence(encoder_input, batch_first=True, padding_value = vocab['<PAD>'])
        expected_output = pad_sequence(expected_output, batch_first=True, padding_value = vocab['<PAD>'])
        
        batch_size = len(batch)
        reshape_size = (batch_size//sequence_length, sequence_length)

        encoder_input = ManagedTensor(encoder_input.reshape(reshape_size), ManagedTensorMemoryStorageMode.CPU)
        expected_output_flat = ManagedTensor(expected_output.reshape(reshape_size).reshape(-1), ManagedTensorMemoryStorageMode.CPU)
        expected_output = ManagedTensor(expected_output.reshape(reshape_size), ManagedTensorMemoryStorageMode.CPU)

        return  encoder_input, expected_output, expected_output_flat


class CustomDataset(Dataset):

    def __init__(self, dataset, unique_words, vocab, sequence_length):

        self.lookUpTable = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",  "u", "v", "w", "x", "y", "z"]
        
        self.encoder_input = dataset
        self.expected_output = [[] for i in range(len(dataset))]
        for i, number in enumerate(dataset):
            letter = UtilityTextProcessing.decode_letter(number, vocab)
            idxLetter = self.lookUpTable.index(letter)
            if idxLetter == len(self.lookUpTable)-1: idxLetter = -1
            self.expected_output[i] = self.lookUpTable[idxLetter+1]
        self.expected_output, _ = UtilityTextProcessing.assign_index(self.expected_output, unique_words)
        self.vocab = vocab
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.encoder_input)

    def __getitem__(self, idx):
        dec_input = self.encoder_input[idx]
        exp_output = self.expected_output[idx]
        vocab = self.vocab
        sequence_length = self.sequence_length
        return dec_input, exp_output, vocab, sequence_length



def main():
    # define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ManagedTensor.init(device)

    text = [random. choice(string.ascii_letters) for i in range(3200*10)]
    text = ' '.join(text).lower()

    # parameters dataloader
    percent_val = 0.2
    sequence_length = 8
    batch_size_train = 4
    batch_size_train = batch_size_train*sequence_length
    batch_size_val = 4
    batch_size_val = batch_size_val*sequence_length
    train_dl, val_dl, vocab = UtilityTextProcessing.process_text(text, percent_val, batch_size_train, batch_size_val, sequence_length)
    
    # define model
    embedding_size = 64
    src_vocab_size = len(vocab)
    n_heads = 1
    num_encoder_layers = 1
    dropout = 0.5
    model = ModelTransformer(src_vocab_size, embedding_size, n_heads, num_encoder_layers, dropout, device).to(device)
    model.init_data(train_dl, val_dl, vocab, batch_size_train, batch_size_val, sequence_length, device)

    # train the model
    num_epochs = 10
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    model.start_training(num_epochs, optimizer, scheduler)


if __name__ == '__main__':
    main()


