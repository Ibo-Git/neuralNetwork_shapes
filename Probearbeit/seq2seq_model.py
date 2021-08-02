
import os
import pathlib
import re



import pickle
import torch
import torch.nn as nn

from torchvision import transforms as transforms

import csv
from RNN import EncoderRNN, DecoderRNN, AttnDecoderRNN, train, evaluate
from Transformer import modelTransformer, train_transformer, evaluate_transformer

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

    index_dec_in = index_target.copy()
    for i in range(len(index_input)):
        num_pad = max_len - len(index_input[i])
        index_input[i] = torch.tensor(index_input[i]+[vocab['<EOS>']]+[vocab['<PAD>']]*num_pad)

        num_pad = max_len - len(index_target[i])
        index_dec_in[i] = torch.tensor([vocab['<SOS>']]+index_target[i]+([vocab['<PAD>']]*num_pad))
        index_target[i] = torch.tensor(index_target[i]+[vocab['<EOS>']]+([vocab['<PAD>']]*num_pad))


    enc_in = index_input
    target = index_target
    dec_in = index_dec_in

    return enc_in, dec_in, target

def batch_tensors(input, dec_in, target, len_string, percent, batch_size):

    input = [torch.stack(input[i*batch_size:(i+1)*batch_size]) for i in range(len_string//batch_size)]
    target = [torch.stack(target[i*batch_size:(i+1)*batch_size]) for i in range(len_string//batch_size)]
    dec_in = [torch.stack(dec_in[i*batch_size:(i+1)*batch_size]) for i in range(len_string//batch_size)]

    split_index = int(len_string * percent) // batch_size

    input_train = input[0:split_index]
    input_val = input[split_index:]

    target_train = target[0:split_index]
    target_val = target[split_index:]

    dec_train = dec_in[0:split_index]
    dec_val = dec_in[split_index:]
    
    return input_train, input_val, target_train, target_val, dec_train, dec_val

def main():
    batch_size = 32
    num_batches = 100
    len_string = batch_size*num_batches

    textfile_name = 'list_phoneme_input'
    vocabfile_name = 'list_phoneme_target'

    if os.path.isfile(os.path.join('Probearbeit', textfile_name + '.pkl')) and os.path.isfile(os.path.join('Probearbeit', vocabfile_name + '.pkl')):
        file_1 = open(os.path.join('Probearbeit', vocabfile_name + '.pkl'), 'rb')
        phoneme_input = pickle.load(file_1)

        file_2 = open(os.path.join('Probearbeit', textfile_name + '.pkl'), 'rb')
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
    enc_in, dec_in, target = transform_to_tensor(index_input, index_target, batch_size, vocab, device)
    input_train, input_val, target_train, target_val, dec_train, dec_val = batch_tensors(enc_in, dec_in, target, len_string, 0.8, batch_size)
 
    max_length = enc_in[0].shape[0]
    #encoder = EncoderRNN(len(input), 256, device).to(device)
    #decoder = AttnDecoderRNN(256, len(vocab), device, 0.1, max_length).to(device)

    model = modelTransformer(len(vocab), 256, len(vocab), device, vocab).to(device)
    transformer_optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<PAD>'])
    #encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr = 0.0001)
    #decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr = 0.0001)


    for epoch in range(50):
        total_loss = 0
        for num_batch in range(len(input_train)):
            _ = train_transformer(input_train[num_batch], dec_train[num_batch], target_train[num_batch], model, transformer_optimizer, criterion, device, vocab)
        for num_batch in range(len(input_val)):
            loss, decoded_output_seq, decoded_expected_seq = evaluate_transformer(input_val[num_batch], dec_val[num_batch], target_val[num_batch], model, device, vocab, criterion) 
            total_loss += loss

        print('Epoch: {}, Loss: {}\n'.format(epoch, total_loss / num_batch))
        print('Expected: {} \nOutput: {}'.format(decoded_expected_seq, decoded_output_seq))







    

if __name__ == '__main__':
    main()
