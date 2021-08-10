import math

import torch
import torch.nn as nn

from TransformerDataset import TokenIDX


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


class TransformerDecoderModel(nn.Module):

    def __init__(self, tgt_vocab_size, embedding_size, n_heads, num_encoder_layers, dropout, device):
        super(TransformerDecoderModel, self).__init__()
        self.device = device

        self.embedding = nn.Embedding(tgt_vocab_size, embedding_size)
        self.pos_encoder = PositionalEncoding(embedding_size, dropout)
        encoder_layer = nn.TransformerEncoderLayer(embedding_size, nhead=n_heads, dim_feedforward=2048, dropout=dropout, batch_first=True, device=device)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.fc_out = nn.Linear(embedding_size, tgt_vocab_size)

    def forward(self, tgt):
        # masks
        tgt_seq_len = tgt.shape[1]
        tgt_attn_mask = nn.Transformer.generate_square_subsequent_mask(self, tgt_seq_len)
        tgt_padding_mask = (tgt == TokenIDX.PAD_IDX)

        out = self.embedding(tgt)
        out = self.pos_encoder(out)
        out = self.transformer_encoder(out, tgt_attn_mask, tgt_padding_mask)
        out = self.fc_out(out)

        return out


class Trainer():
    def __init__(self, model, optimizer, scheduler, num_minibatches, device):
        super(Trainer, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = nn.CrossEntropyLoss(ignore_index=TokenIDX.PAD_IDX)
        self.device = device
        self.num_minibatches = num_minibatches
        self.minibatch_counter = 0
        self.total_batch_loss = 0
        self.optimizer.zero_grad()

    def train(self, dec_in, exp_out, exp_out_flat):
        output = self.model(dec_in)
        loss = self.criterion(output.reshape(exp_out_flat.shape[0], -1), exp_out_flat)
        loss.backward()
        self.total_batch_loss += loss.item() 
        self.minibatch_counter += 1

        # update every batch
        if self.minibatch_counter == self.num_minibatches:
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.scheduler is not None: self.scheduler.step()
            self.minibatch_counter = 0
            total_batch_loss = self.total_batch_loss
            self.total_batch_loss = 0
        
        return total_batch_loss / self.num_minibatches

    def evaluate(self, dec_in, exp_out, exp_out_flat):
        output = self.model(dec_in)
        loss = self.criterion(output.reshape(exp_out_flat.shape[0], -1), exp_out_flat)
        output = torch.argmax(output, 2)
        accuracy = torch.sum(output.reshape(-1) == exp_out_flat) / len(exp_out_flat)
        
        return loss, accuracy, output