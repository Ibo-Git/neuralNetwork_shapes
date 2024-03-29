import math
import os
import pickle
import random

import torch
from tqdm import tqdm
import youtokentome as yttm
from torch.cuda import is_available
from torch.functional import split
from torch.utils.data import DataLoader

from TransformerDataset import TokenIDX, TransformerDataset
from TransformerDecoderModel import Trainer, TransformerDecoderModel


class DataProcessing():
    def __init__(self, filepath, savepath, txt_filename, modelname, load_filename, state='save'):
        super(DataProcessing, self).__init__()
        self.filepath = filepath
        self.state = state

        self.txt_filepath = os.path.join(filepath, txt_filename) # file path to read file from
        self.txt_filepath_lc = os.path.join(filepath, 'lowercase.txt') # file path to save file with lower case
        self.modelpath = os.path.join(savepath, modelname) # file path to load model 
        self.save_filepath = os.path.join(savepath, load_filename) # file path to save bpe-encoded file

        self.read_data()


    def read_data(self):
        with open(self.txt_filepath, 'r', encoding="UTF-8") as file:
            file = file.read()
            self.file = file.lower()
        
        with open(self.txt_filepath_lc, 'w', encoding='UTF-8') as txt_file:
            txt_file.write(self.file)

    
    def data_preprocessing(self, vocab_size):
        # process and save
        if self.state == 'save':
            yttm.BPE.train(
                data = self.txt_filepath_lc, 
                vocab_size = vocab_size, 
                model = self.modelpath, 
                pad_id = TokenIDX.PAD_IDX,
                unk_id = TokenIDX.UNK_IDX,
                bos_id = TokenIDX.SOS_IDX,
                eos_id = TokenIDX.EOS_IDX
            )

            self.bpe = yttm.BPE(model=self.modelpath)
            encoded_file = self.bpe.encode([self.file], output_type=yttm.OutputType.SUBWORD)

            with open(self.save_filepath, 'wb') as file:
                pickle.dump(encoded_file, file)

        # load
        elif self.state == 'load':
            self.bpe = yttm.BPE(model=self.modelpath)

            with open(self.save_filepath, 'rb') as file:
                encoded_file = pickle.load(file)

        return self.bpe.vocab(), encoded_file[0]


    def data_splitting(self, encoded_file, sequence_length, split_val_percent):

        sequences = [[] for i in range(len(encoded_file)//sequence_length+1)]
        num_token = 0
        num_sequence = 0

        print('Split data...')
        for i in tqdm(range(len(encoded_file))):

            if num_token >= sequence_length:
                if encoded_file[i+1][0] == '▁':
                    if num_token > sequence_length*1.25:
                        del sequences[num_sequence]
                        num_token = 0
                    else:
                        num_token = 0
                        sequences[num_sequence] = ''.join(sequences[num_sequence])
                        sequences[num_sequence].replace('▁',' ')
                        sequences[num_sequence] = self.bpe.encode(sequences[num_sequence], output_type=yttm.OutputType.ID)
                        num_sequence += 1
                else:
                    sequences[num_sequence].append(encoded_file[i])
                    num_token += 1
            else:
                sequences[num_sequence].append(encoded_file[i])
                num_token += 1

        del sequences[num_sequence:]


        max_len = len(max(sequences, key=len))

        decoder_sequence =  [[] for i in range(len(sequences))]
        target_sequence = [[] for i in range(len(sequences))]

        print('Transform to tensors: Insert SOS, EOS and PAD...')
        for i, sequence in enumerate(tqdm(sequences)):
            decoder_sequence[i] =  torch.tensor([TokenIDX.SOS_IDX] + sequence + (max_len-len(sequence)+1)*[TokenIDX.PAD_IDX])
            target_sequence[i] = torch.tensor(sequence + [TokenIDX.EOS_IDX] + (max_len-len(sequence)+1)*[TokenIDX.PAD_IDX])

        random.shuffle(sequences)

        train_dec_sequences =  decoder_sequence[:math.floor(len(decoder_sequence)*(1-split_val_percent))]
        train_target_sequences = target_sequence[:math.floor(len(target_sequence)*(1-split_val_percent))]
        val_dec_sequences =  decoder_sequence[math.floor(len(decoder_sequence)*(1-split_val_percent)):]
        val_target_sequences =  target_sequence[math.floor(len(target_sequence)*(1-split_val_percent)):]

        return train_dec_sequences, train_target_sequences, val_dec_sequences, val_target_sequences


    def decode_sequence(self, sequence):
        decoded_sequence = self.bpe.decode(ids=sequence)
        return decoded_sequence


def main():
    # define hyperparameters
    # ...for data processing
    vocab_size = 200
    batch_size = 128
    num_minibatches = 2 # set to 1 if no minibatch is required
    minibatch_size = int(batch_size / num_minibatches)
    sequence_length = 100
    split_val_percent = 0.2
    state = 'save'
    # ...for model 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_size = 128; n_heads = 4; num_encoder_layers = 6; dropout = 0.1
    # ...for optimizer
    lr = 0.00002
    # ...for training
    num_epochs = 100


    # specify all path and filenames
    
    filepath = os.path.join('datasets', 'law')
    savepath = os.path.join('saved_files', 'law')
    if not os.path.isdir(filepath):
        os.makedirs(filepath)
    if not os.path.isdir(savepath):
        os.makedirs(savepath)

    txt_filename = 'law.txt'
    load_filename = 'bpe_' + str(vocab_size) + '.pkl'
    modelname = 'bpe_' + str(vocab_size) + '.model'

    # process data
    processor = DataProcessing(filepath, savepath, txt_filename, modelname, load_filename, state)
    vocab, encoded_file = processor.data_preprocessing(vocab_size=vocab_size)
    train_dec_sequences, train_target_sequences, val_dec_sequences, val_target_sequences = processor.data_splitting(encoded_file, sequence_length, split_val_percent)
    # dataloader
    train_ds = TransformerDataset(train_dec_sequences, train_target_sequences, batch_size=batch_size)
    val_ds = TransformerDataset(val_dec_sequences, val_target_sequences, batch_size=batch_size)
    train_dl = DataLoader(train_ds, batch_size=minibatch_size, shuffle=True, num_workers=0, collate_fn=TransformerDataset.collate_fn, pin_memory=True, persistent_workers=False)
    val_dl = DataLoader(val_ds, batch_size=minibatch_size, shuffle=True, num_workers=0, collate_fn=TransformerDataset.collate_fn, pin_memory=True, persistent_workers=False)


    # model & training
    model = TransformerDecoderModel(tgt_vocab_size=len(vocab), embedding_size=embedding_size, n_heads=n_heads, num_encoder_layers=num_encoder_layers, dropout=dropout, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    trainer = Trainer(model, optimizer, scheduler, device)

    # trainer.test_model(model, os.path.join(savepath, modelname), 'Hallo, mein Freund :) .', 10)

    # Training loop
    for epoch in range(num_epochs):
        print('Epoch:', epoch)

        # training
        print('Start training...')
        model.train()
        total_train_loss = 0
        for num_minibatch, (decoder_input, expected_output, expected_output_flat) in enumerate(tqdm(train_dl)):
            # to device
            decoder_input = decoder_input.to(device)
            expected_output_flat = expected_output_flat.to(device)
            # train
            train_minibatch_loss = trainer.train_batch(decoder_input, expected_output_flat)
            total_train_loss += train_minibatch_loss

            if (num_minibatch + 1) % num_minibatches == 0:
                trainer.train_update()
                trainer.train_reset()

        print('train_loss: {}\n'.format(total_train_loss / (num_minibatch + 1)))

        # validation
        print('Start validation...')
        model.eval()
        total_val_loss = 0
        total_val_acc = 0
        for num_batch, (decoder_input, expected_output, expected_output_flat) in enumerate(tqdm(val_dl)):
            # to device
            decoder_input = decoder_input.to(device)
            expected_output_flat = expected_output_flat.to(device)
            # evaluate
            val_batch_loss, val_batch_acc, output_batch = trainer.evaluate(decoder_input, expected_output, expected_output_flat)
            total_val_loss += val_batch_loss
            total_val_acc += val_batch_acc

        # decode sequences: [0] -> first entry of batch. [0:N] also possible to display more outputs
        decoded_output = processor.decode_sequence(output_batch[0].tolist())
        decoded_exp_output = processor.decode_sequence(expected_output[0].tolist())
        # show results 
        print('val_loss: {}, val_acc: {}\n'.format(total_val_loss / (num_batch + 1), total_val_acc / (num_batch + 1)))
        print('Output: {}\nExpected: {}\n'.format(decoded_output, decoded_exp_output))
        

if __name__ == '__main__':
    main()

