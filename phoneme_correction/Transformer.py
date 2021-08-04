import math
import os

import torch
import torch.nn as nn


class modelTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embedding_size, heads, encoder_layer, decoder_layer, vocab, device):
        self.embedding_size = embedding_size
        self.heads = heads
        self.encoder_layer = encoder_layer
        self.decoder_layer = decoder_layer
        self.device = device
        self.vocab = vocab
        super(modelTransformer, self).__init__()

        self.embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.positional_encoding = PositionalEncoding(embedding_size)
        self.transformer = nn.Transformer(embedding_size, nhead=self.heads, num_encoder_layers=self.encoder_layer, num_decoder_layers=self.decoder_layer, dropout=0.1)
        self.fc_out = nn.Linear(embedding_size, tgt_vocab_size)

    def forward(self, src, tgt):
        
        # get masks for transformer
        tgt_seq_len = tgt.shape[1]
        src_padding_mask = []
        tgt_padding_mask = []
        for i in range(src.shape[0]):
            src_padding_mask.append((src[i] == self.vocab['<PAD>']))
            tgt_padding_mask.append((tgt[i] == self.vocab['<PAD>']))
        src_padding_mask = torch.stack(src_padding_mask)
        tgt_padding_mask = torch.stack(tgt_padding_mask)
        tgt_mask = self.transformer.generate_square_subsequent_mask(sz = tgt_seq_len).to(self.device)

        src = self.embedding(src)
        tgt = self.embedding(tgt)
        #src = self.positional_encoding(src)
        #tgt = self.positional_encoding(tgt)
        tgt = torch.permute(tgt, [1, 0, 2])
        src = torch.permute(src, [1, 0, 2])
        out = self.transformer(src, tgt, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask, src_key_padding_mask=src_padding_mask)
        out = torch.permute(out, [1, 0, 2])
        out = self.fc_out(out)

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




def train_transformer(input_tensor, decoder_input, target_tensor, model, optimizer, criterion, device, vocab):
    model.train()

    encoder_input = input_tensor.to(device)
    target_tensor = target_tensor.to(device)
    decoder_input = decoder_input.to(device)

    optimizer.zero_grad()
    
    output = model(encoder_input, decoder_input)
    loss = criterion(output.reshape(-1, len(vocab)), target_tensor.reshape(-1))
    loss.backward()
    optimizer.step()
   
    return loss.item()


def evaluate_transformer(input_tensor, decoder_input_tensor, target_tensor, model, device, vocab, criterion):
    model.eval()

    encoder_input = input_tensor.to(device)
    target_tensor = target_tensor.to(device)
    decoder_input = decoder_input_tensor.to(device)
  
    output = model(encoder_input, decoder_input)
    loss = criterion(output.reshape(-1, len(vocab)), target_tensor.reshape(-1))
    acc = get_accuracy(torch.argmax(output, 2), target_tensor, vocab)
    #acc_2 = get_accuracy(input_tensor, target_tensor, vocab)
    decoded_output_seq, decoded_expected_seq = decode_char(torch.argmax(output, 2)[4], target_tensor[4], vocab)

    return loss.item(), acc, decoded_output_seq, decoded_expected_seq


def get_accuracy(output, target, vocab):
    # [batch, sequence]
    total_acc = 0
    for i, target_seq in enumerate(target):
        index_pad = (target_seq == vocab['<PAD>']).nonzero(as_tuple=True)[0]
        if len(index_pad) == 0:
            index_pad = len(target_seq)
        else:
            index_pad = index_pad[0]
        
        acc = torch.sum(output[i][0:index_pad] == target[i][0:index_pad]) / len(target[i][0:index_pad])
        total_acc += acc

    return total_acc / target.shape[0]


def decode_char(output_sequence, target_sequence, vocab):
    key_list = list(vocab)
    index_pad = (target_sequence == vocab['<PAD>']).nonzero(as_tuple=True)[0]
    if len(index_pad) == 0:
        index_pad = len(target_sequence)
    else:
        index_pad = index_pad[0]

    decoded_vec_output = []
    decoded_vec_expected = []
    for i, index in enumerate(output_sequence):
        if i < index_pad:
            decoded_vec_output.append(key_list[index])
        else:
            break

    for i, index in enumerate(target_sequence):
        if i < index_pad:
            decoded_vec_expected.append(key_list[index])
        else:
            break

    return decoded_vec_output, decoded_vec_expected


def save(model, file_name):
    model_folder_path = '.\saved_files'
    if not os.path.exists(model_folder_path):
        os.makedirs(model_folder_path)

    file_name = os.path.join(model_folder_path, file_name)
    torch.save(model.state_dict(), file_name)


def load(filename, model, device):
    model.load_state_dict(torch.load(os.path.join('.\saved_files', filename + '.pth'), map_location=torch.device(device)))
    model.eval()
