import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random
import os

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(EncoderRNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1) 
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, device):
        super(DecoderRNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output[0])
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, device, dropout, max_length):
        super(AttnDecoderRNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = self.out(output[0])
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)



def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, device, vocab, max_length):
    encoder.train()
    decoder.train()

    teacher_forcing_ratio = 0.5

    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.shape[0]
    target_length = target_tensor.shape[0]
    
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([vocab['<SOS>']], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    output_sequence = []
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, torch.unsqueeze(target_tensor[di], 0))
            decoder_input = target_tensor[di]  # Teacher forcing
            output_sequence.append(torch.argmax(decoder_output))
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, torch.unsqueeze(target_tensor[di], 0))
            if decoder_input.item() == vocab['<EOS>']:
                break
            
            output_sequence.append(torch.argmax(decoder_output))

    
    decoded_output_seq, decoded_expected_seq = decode_char(output_sequence, target_tensor, vocab)
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length, decoded_expected_seq, decoded_output_seq



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


def evaluate(input_tensor, target_tensor, encoder, decoder, device, vocab, criterion, max_length):
    teacher_forcing_ratio = 0.5
    encoder.eval()
    decoder.eval()

    encoder_hidden = encoder.initHidden()
    input_length = input_tensor.shape[0]
    target_length = target_tensor.shape[0]
    
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([vocab['<SOS>']], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    output_sequence = []
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, torch.unsqueeze(target_tensor[di], 0))
            decoder_input = target_tensor[di]  # Teacher forcing
            output_sequence.append(torch.argmax(decoder_output))
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, torch.unsqueeze(target_tensor[di], 0))
            if decoder_input.item() == vocab['<EOS>']:
                break
            
            output_sequence.append(torch.argmax(decoder_output))

    
    decoded_output_seq, decoded_expected_seq = decode_char(output_sequence, target_tensor, vocab)

    return loss.item() / target_length, decoded_output_seq, decoded_expected_seq


def save(model, file_name = 'snake_model.pth'):
    model_folder_path = '.\saved_files'
    if not os.path.exists(model_folder_path):
        os.makedirs(model_folder_path)

    file_name = os.path.join(model_folder_path, file_name)
    torch.save(model.state_dict(), file_name)


def load(filename, model, device):
    model.load_state_dict(torch.load(os.path.join('.\saved_files', filename + '.pth'), map_location=torch.device(device)))
    model.eval()
