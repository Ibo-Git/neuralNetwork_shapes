
import csv
import math
import os
import pickle
import re

import nltk
import torch
import torch.nn as nn
from torchvision import transforms as transforms
from tqdm import tqdm

from GRU import AttnDecoderRNN, DecoderRNN, EncoderRNN, evaluate, train
from Transformer import Trainer, modelTransformer


class UtilityRNN():
    # load files
    def read_dataset(filename):
        text = []
        with open(filename, encoding="utf8") as file:
            file = csv.reader(file, delimiter="\t")

            for line in file:
                text.append(line)  
        
        return text

    # sort phonemes 
    def sort_input(text_1, text_2, len_data):
        phoneme_sorted = []
        list_phoneme_input, list_words_input = zip(*text_1)
        list_phoneme_target, list_words_target = zip(*text_2)

        list_words_input = {key: value for value, key in enumerate(list(list_words_input))}
        list_words_target = list(list_words_target)
        list_phoneme_input = list(list_phoneme_input)
        list_phoneme_target = list(list_phoneme_target)

        for n in tqdm(range(len_data-1, -1, -1)):
            word = list_words_target[n]

            if word in list_words_input:
                phoneme_sorted.insert(0, list_phoneme_input[list_words_input[word]])
            else:
                del list_phoneme_target[n]            
            
        return phoneme_sorted, list_phoneme_target[0:len(phoneme_sorted)]

    # split at phonemes
    def split_list(phoneme_list, words_list):
        splitted_list = []
        for word in tqdm(words_list):
            m = re.split(rf"({'|'.join(phoneme_list)})", word)
            m = [n for n in m if n]
            splitted_list.append(m)
        return splitted_list

    # sort unmatching phonemes out using BLEU score 
    def sort_out_phonemes(phoneme_input_splitted, phoneme_target_splitted):

        for i in tqdm(range(len(phoneme_input_splitted)-1, -1, -1)):
            reference = phoneme_target_splitted[i]
            candidate = phoneme_input_splitted[i]
            ls_dist = nltk.edit_distance(reference, candidate, substitution_cost=1, transpositions=True)

            if ls_dist > math.ceil(0.4*len(reference)) or ls_dist > math.ceil(0.4*len(candidate)):
                del phoneme_input_splitted[i]
                del phoneme_target_splitted[i]

        
        return phoneme_input_splitted, phoneme_target_splitted
 
    # assign index according to vocab 
    def assign_index(input, target, unique_words):
        vocab = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<TRL>':3}
        for word in unique_words: vocab[word] = len(vocab)

        index_input = [[vocab[char] for char in word] for word in input]
        index_target = [[vocab[char] for char in word] for word in target]

        return index_input, index_target, vocab

    # append SOS and EOS tokens
    def transform_to_tensor(index_input, index_target, vocab):
        index_dec_in = index_target.copy()

        for i in tqdm(range(len(index_input))):
            index_input[i] = torch.tensor(index_input[i]+[vocab['<EOS>']])
            index_dec_in[i] = torch.tensor([vocab['<SOS>']]+index_target[i])
            index_target[i] = torch.tensor(index_target[i]+[vocab['<EOS>']])

        enc_in = index_input
        target = index_target
        dec_in = index_dec_in

        return enc_in, dec_in, target

    # transform to tensors and insert or append EOS/SOS/PAD tokens
    def transform_to_tensor_2(index_input, index_target, vocab):

        if len(max(index_target, key=len)) > len(max(index_input, key=len)):
            max_len = len(max(index_target, key=len))
        else:
            max_len = len(max(index_input, key=len))

        index_dec_in = index_target.copy()
        for i in tqdm(range(len(index_input))):
            num_pad = max_len - len(index_input[i])
            index_input[i] = torch.tensor(index_input[i]+[vocab['<EOS>']]+[vocab['<PAD>']]*num_pad)

            num_pad = max_len - len(index_target[i])
            index_dec_in[i] = torch.tensor([vocab['<SOS>']]+index_target[i]+([vocab['<PAD>']]*num_pad))
            index_target[i] = torch.tensor(index_target[i]+[vocab['<EOS>']]+([vocab['<PAD>']]*num_pad))

        enc_in = index_input
        target = index_target
        dec_in = index_dec_in

        return enc_in, dec_in, target

    # batch input and split to training and val datasets
    def batch_tensors(input, dec_in, target, len_data, percent, batch_size):
        input = [torch.stack(input[i*batch_size:(i+1)*batch_size]) for i in range(len_data//batch_size)]
        target = [torch.stack(target[i*batch_size:(i+1)*batch_size]) for i in range(len_data//batch_size)]
        dec_in = [torch.stack(dec_in[i*batch_size:(i+1)*batch_size]) for i in range(len_data//batch_size)]

        split_index = int(len_data * percent) // batch_size

        input_train = input[0:split_index]
        input_val = input[split_index:]

        target_train = target[0:split_index]
        target_val = target[split_index:]

        dec_train = dec_in[0:split_index]
        dec_val = dec_in[split_index:]
        
        return input_train, input_val, target_train, target_val, dec_train, dec_val

    # load if files exist, else perform text processing and save
    def save_or_load(phoneme_list, len_data, input_filename, target_filename, vocab_filename):
        if (os.path.isfile(os.path.join('phoneme_correction', input_filename + '.pkl')) and
            os.path.isfile(os.path.join('phoneme_correction', target_filename + '.pkl')) and
            os.path.isfile(os.path.join('phoneme_correction', vocab_filename + '.pkl'))):
            
            print('load data...')
            file_1 = open(os.path.join('phoneme_correction', input_filename + '.pkl'), 'rb')
            index_input = pickle.load(file_1)

            file_2 = open(os.path.join('phoneme_correction', target_filename + '.pkl'), 'rb')
            index_target = pickle.load(file_2)

            file_3 = open(os.path.join('phoneme_correction', vocab_filename + '.pkl'), 'rb')
            vocab = pickle.load(file_3)
        else:
            # read dataset
            print('read data...')
            target = UtilityRNN.read_dataset(os.path.join('datasets', 'phonemes', 'g2pPhonemes.tsv'))
            input = UtilityRNN.read_dataset(os.path.join('datasets', 'phonemes', 'modelPhonemes.tsv'))
            # prepare dataset
            print('sort data...')
            phoneme_input, phoneme_target = UtilityRNN.sort_input(input, target, len_data)
            print('split text...')
            phoneme_input_splitted = UtilityRNN.split_list(phoneme_list, phoneme_input)
            phoneme_target_splitted = UtilityRNN.split_list(phoneme_list, phoneme_target)
            print('sort out phonemes...')
            phoneme_input_splitted, phoneme_target_splitted = UtilityRNN.sort_out_phonemes(phoneme_input_splitted, phoneme_target_splitted)
            print('assign index...')
            index_input, index_target, vocab = UtilityRNN.assign_index(phoneme_input_splitted, phoneme_target_splitted, phoneme_list)
        

            # save data
            print('save data...')
            with open(os.path.join('phoneme_correction', input_filename + '.pkl'), "wb") as file_1:
                pickle.dump(index_input, file_1)

            with open(os.path.join('phoneme_correction', target_filename + '.pkl'), "wb") as file_2:
                pickle.dump(index_target, file_2)

            with open(os.path.join('phoneme_correction', vocab_filename + '.pkl'), "wb") as file_3:
                pickle.dump(vocab, file_3)

        return index_input, index_target, vocab




def main():
    # define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # max length: 853592
    len_data = 853592
    model_type = 'Transformer'
    num_epochs = 50
    learning_rate = 0.0001
    batch_size = 512

    phoneme_list = ['T', 'i', 'I', 'e', 'E', 'y', '2', '9', '@', '6', '3', 'a', 'u', 'U', 'o', 'O', 'p', 'b', 't', 'd', 'tS', 'dZ', 'c', 'g', 'q', 'p', 'B', 'f', 'v', 's', 'z', 'S', 'Z', 'C', 'x', 'h', 'm', 'n', 'N', 'l', 'R', 'j', ':', '~', 'k', 'r', 'Y']
    # load if files exist, else perform text processing and save
    index_input, index_target, vocab = UtilityRNN.save_or_load(phoneme_list, len_data, input_filename = 'idx_input_phoneme', target_filename = 'idx_target_phoneme', vocab_filename = 'phoneme_vocab')

    # process text
    print('transform to tensors...')
    enc_in, dec_in, target = UtilityRNN.transform_to_tensor_2(index_input, index_target, vocab)
    print('batch data...')
    input_train, input_val, target_train, target_val, dec_train, dec_val = UtilityRNN.batch_tensors(input=enc_in, dec_in=dec_in, target=target, len_data=len(enc_in), percent=0.8, batch_size=batch_size)

    



    # define loss criterion
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<PAD>'])

    if model_type == 'Transformer':
        model = modelTransformer(src_vocab_size=len(vocab), tgt_vocab_size=len(vocab), embedding_size=256, heads=4, encoder_layer=6, decoder_layer=6, vocab=vocab, device=device).to(device)
        transformer_optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
        trainer = Trainer(model, transformer_optimizer, criterion, vocab, device)

        for epoch in range(num_epochs):
            total_loss = 0
            total_acc = 0

            print('start training...')
            for num_batch in tqdm(range(len(input_train))):
                train_loss = trainer.train_transformer(input_train[num_batch], dec_train[num_batch], target_train[num_batch])

            print('start validation...')
            for num_batch in tqdm(range(len(input_val))):
                loss, acc, decoded_input_seq, decoded_output_seq, decoded_expected_seq = trainer.evaluate_transformer(input_val[num_batch], dec_val[num_batch], target_val[num_batch]) 
                total_loss += loss
                total_acc += acc

            print('Epoch: {}, val_loss: {}, val_acc: {}'.format(epoch, total_loss / (num_batch+1), total_acc / (num_batch+1)))
            print('Input: {}\nOutput: {} \nExpected: {}\n'.format(decoded_input_seq, decoded_output_seq, decoded_expected_seq))




    elif model_type == 'GRU':
        max_length = enc_in[0].shape[0]
        encoder = EncoderRNN(len(enc_in), 256, device).to(device)
        decoder = AttnDecoderRNN(256, len(vocab), device, 0.1, max_length).to(device)
        encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr = learning_rate)
        decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr = learning_rate)

        for epoch in range(num_epochs):
            total_loss = 0
            total_acc = 0
            for i in range(len(input_train)):
                for n in range(len(input_train[i])):
                    _, _, _ = train(input_train[i][n], target_train[i][n], encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, device, vocab, max_length=max_length)
            for i in range(len(input_val)):
                total_batch_loss = 0
                total_batch_acc = 0
                for n in range(len(input_val[i])):
                    loss, acc, decoded_output_seq, decoded_expected_seq = evaluate(input_val[i][n], target_val[i][n], encoder, decoder, device, vocab, criterion, max_length=max_length) 
                    total_batch_loss += loss
                    total_batch_acc += acc
                
                total_loss += total_batch_loss
                total_acc += total_batch_acc

            print('Epoch: {}, val_loss: {}, val_acc: {}'.format(epoch, total_loss / (i+1) / (n+1), total_acc / (i+1) / (n+1)))
            print('Expected: {} \nOutput: {}\n'.format(decoded_expected_seq, decoded_output_seq))




if __name__ == '__main__':
    main()
