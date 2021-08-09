import os
import pathlib
import pickle
from enum import Enum
from torch.utils.data import DataLoader
import youtokentome as yttm

from TransformerDataset import TransformerDataset, TokenIDX


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


    def decode_sequence(self, sequence):
        decoded_sequence = self.bpe.decode(ids=sequence, ignore_ids=None)
        return decoded_sequence


def main():
    # define hyperparameters
    vocab_size = 200
    batch_size = 128

    # specify all path and filenames
    filepath = os.path.join('datasets', 'law')
    savepath = os.path.join('saved_files', 'law')
    txt_filename = 'law.txt'
    load_filename = 'bpe_' + str(vocab_size) + '.pkl'
    modelname = 'bpe_' + str(vocab_size) + '.model'

    # process data
    processor = DataProcessing(filepath, savepath, txt_filename, modelname, load_filename, 'save')
    vocab, encoded_file = processor.data_preprocessing(vocab_size=vocab_size)
    # split data into given sequence length

    # dataloader
    train_ds = TransformerDataset(train_dec_in, train_exp_out, batch_size)
    val_ds = TransformerDataset(val_dec_in, val_exp_out, batch_size)
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, collate_fn=TransformerDataset.collate_fn, pin_memory=True, persistent_workers=True)
    train_dl = DataLoader(val_ds, batch_size, shuffle=True, num_workers=4, collate_fn=TransformerDataset.collate_fn, pin_memory=True, persistent_workers=True)

    # model

    # training

if __name__ == '__main__':
    main()

