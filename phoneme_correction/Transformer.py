import math
import os

import torch
import torch.nn as nn
import nltk


class modelTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embedding_size, heads, encoder_layer, decoder_layer, vocab, device):
        super(modelTransformer, self).__init__()
        self.embedding_size = embedding_size
        self.heads = heads
        self.encoder_layer = encoder_layer
        self.decoder_layer = decoder_layer
        self.device = device
        self.vocab = vocab

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


class Trainer():
    def __init__(self, model, optimizer, criterion, vocab, device):
        super(Trainer, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.vocab = vocab
        self.device = device
        self.key_list = list(self.vocab)


    def train_transformer(self, input_tensor, decoder_input, target_tensor):
        self.model.train()

        encoder_input = input_tensor.to(self.device)
        target_tensor = target_tensor.to(self.device)
        decoder_input = decoder_input.to(self.device)

        self.optimizer.zero_grad()
        
        output = self.model(encoder_input, decoder_input)
        loss = self.criterion(output.reshape(-1, len(self.vocab)), target_tensor.reshape(-1))
        loss.backward()
        self.optimizer.step()
    
        return loss.item()


    def evaluate_transformer(self, input_tensor, decoder_input_tensor, target_tensor):
        self.model.eval()

        encoder_input = input_tensor.to(self.device)
        target_tensor = target_tensor.to(self.device)
        decoder_input = decoder_input_tensor.to(self.device)
    
        output = self.model(encoder_input, decoder_input)
        loss = self.criterion(output.reshape(-1, len(self.vocab)), target_tensor.reshape(-1))

        num_sequence = 4
        acc = self.get_accuracy(torch.argmax(output, 2), target_tensor, num_sequence)
        decoded_input_seq = self.decode_seq(encoder_input[num_sequence])
        decoded_output_seq = self.decode_seq(torch.argmax(output, 2)[num_sequence], self.idx_EOS_target)
        decoded_expected_seq = self.decode_seq(target_tensor[num_sequence])

        return loss.item(), acc, decoded_input_seq, decoded_output_seq, decoded_expected_seq


    def get_accuracy(self, output, target, num_sequence):
        # [batch, sequence]
        total_acc = 0

        for output_seq, target_seq in zip(output, target):
            idx_EOS_target = self.get_end_of_sentence(target_seq)
            ls_dist = nltk.edit_distance(output_seq[0:idx_EOS_target], target_seq[0:idx_EOS_target], substitution_cost=1, transpositions=True)
            acc = 1 - ls_dist/(idx_EOS_target + 1)
            total_acc += acc

        self.idx_EOS_target = self.get_end_of_sentence(target[num_sequence])

        return total_acc / target.shape[0]


    def get_end_of_sentence(self, sequence):
        idx_EOS = (sequence == self.vocab['<EOS>']).nonzero(as_tuple=True)[0]

        if len(idx_EOS) == 0:
            idx_EOS = len(sequence)-1

        return idx_EOS


    def decode_seq(self, sequence, idx_EOS=None):

        if idx_EOS == None:
            idx_EOS = self.get_end_of_sentence(sequence)

        decoded_sequence = [self.key_list[index] for index in sequence[0:idx_EOS]]

        return decoded_sequence


    def save(self, file_name):
        model_folder_path = '.\saved_files\phonemes'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.model.state_dict(), file_name)


    def load(self, filename):
        self.model.load_state_dict(torch.load(os.path.join('.\saved_files\phonemes', filename + '.pth'), map_location=torch.device(self.device)))
        self.model.eval()
