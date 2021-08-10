import math
import os
import pickle
import random

import torch
import tqdm
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
        self.txt_filepath_lc = os.path.join(savepath, 'lowercase.txt') # file path to save file with lower case
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
            encoded_file = self.bpe.encode([self.file], output_type=yttm.OutputType.ID)

            with open(self.save_filepath, 'wb') as file:
                pickle.dump(encoded_file, file)

        # load
        elif self.state == 'load':
            self.bpe = yttm.BPE(model=self.modelpath)

            with open(self.save_filepath, 'rb') as file:
                encoded_file = pickle.load(file)

        return self.bpe.vocab(), encoded_file


    def data_splitting(self, encoded_file, sequence_length, split_val_percent):
        sequences = [encoded_file[0][x:x + sequence_length] for x in range(0, len(encoded_file[0]), sequence_length)]
        random.shuffle(sequences)
        train_sequences =  sequences[:math.floor(len(sequences)*(1-split_val_percent))]
        val_sequences =  sequences[math.floor(len(sequences)*(1-split_val_percent)):]
        return train_sequences, val_sequences


    def decode_sequence(self, sequence):
        decoded_sequence = self.bpe.decode(ids=sequence)
        return decoded_sequence


def main():
    # define hyperparameters
    # ...for data processing
    vocab_size = 200
    batch_size = 128
    num_minibatches = 1 # set to 1 if no minibatch is required
    minibatch_size = int(batch_size / num_minibatches)
    sequence_length = 100
    split_val_percent = 0.2
    state = 'load'
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
    txt_filename = 'law.txt'
    load_filename = 'bpe_' + str(vocab_size) + '.pkl'
    modelname = 'bpe_' + str(vocab_size) + '.model'

    # process data
    processor = DataProcessing(filepath, savepath, txt_filename, modelname, load_filename, state)
    vocab, encoded_file = processor.data_preprocessing(vocab_size=vocab_size)
    train_sequences, val_sequences = processor.data_splitting(encoded_file, sequence_length, split_val_percent)

    # dataloader
    train_ds = TransformerDataset(train_sequences, train_sequences, batch_size=batch_size)
    val_ds = TransformerDataset(val_sequences, val_sequences, batch_size=batch_size)
    train_dl = DataLoader(train_ds, batch_size=minibatch_size, shuffle=True, num_workers=4, collate_fn=TransformerDataset.collate_fn, pin_memory=True, persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=minibatch_size, shuffle=True, num_workers=4, collate_fn=TransformerDataset.collate_fn, pin_memory=True, persistent_workers=True)

    # model & training
    model = TransformerDecoderModel(tgt_vocab_size=len(vocab), embedding_size=embedding_size, n_heads=n_heads, num_encoder_layers=num_encoder_layers, dropout=dropout, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    trainer = Trainer(model, optimizer, scheduler, num_minibatches, device)

    # Training loop
    for epoch in range(num_epochs):
        # training
        print('Start training...')
        model.train()
        total_train_loss = 0
        for num_batch, (decoder_input, expected_output, expected_output_flat) in enumerate(train_dl):
            print(num_batch)
            train_batch_loss = trainer.train(decoder_input.to(device), expected_output, expected_output_flat.to(device))
            total_train_loss += train_batch_loss

        print('Epoch: {}, train_loss: {}'.format(epoch, total_train_loss / (num_batch + 1)))

        # validation
        print('Start validation...')
        model.eval()
        total_val_loss = 0
        total_val_acc = 0
        for num_batch, (decoder_input, expected_output, expected_output_flat) in enumerate(val_dl):
            val_batch_loss, val_batch_acc, output_sequence = trainer.evaluate(decoder_input.to(device), expected_output, expected_output_flat.to(device))
            total_val_loss += val_batch_loss
            total_val_acc += val_batch_acc

        decoded_output_sequences = processor.decode_sequence(output_sequence[0:2].tolist())
        print('val_loss: {}, val_acc: {}'.format(total_val_loss / (num_batch + 1), val_batch_acc / (num_batch + 1)))
        print('Output: {}'.format(decoded_output_sequences))
        
if __name__ == '__main__':
    main()

