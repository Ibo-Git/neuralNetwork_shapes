import os
import pathlib
import pickle
from enum import Enum

import youtokentome as yttm

from TransformerDataset import TransformerDataset, TokenIDX


class DataProcessing():
    def __init__(self, file_path, model_path, save_path, load_filename, state='save'):
        super(DataProcessing, self).__init__()
        # path to read and save txt files
        self.file_path = file_path
        self.model_path = model_path
        self.save_path = save_path
        self.load_filename = load_filename
        self.state = state

        self.txt_file_path = os.path.join(save_path, 'law_file.txt')
        self.read_data()


    def read_data(self):
        with open(self.file_path, 'r', encoding="UTF-8") as file:
            file = file.read()
            self.file = file.lower()
        
        with open(self.txt_file_path, 'w', encoding='UTF-8') as txt_file:
            txt_file.write(self.file)

    
    def data_preprocessing(self, vocab_size):
        # process and save
        if self.state == 'save':
            yttm.BPE.train(
                data = self.txt_file_path, 
                vocab_size = vocab_size, 
                model = self.model_path, 
                pad_id = TokenIDX.PAD_IDX,
                unk_id = TokenIDX.UNK_IDX,
                bos_id = TokenIDX.SOS_IDX,
                eos_id = TokenIDX.EOS_IDX
            )

            self.bpe = yttm.BPE(model=self.model_path)
            vocab = self.bpe.vocab()
            encoded_file = self.bpe.encode([self.file], output_type=yttm.OutputType.ID)

            savename = os.path.join(self.save_path, 'vocab_enc_' + str(vocab_size) + '.pkl')
            with open(savename, 'wb') as file:
                pickle.dump([vocab, encoded_file], file)

        # load
        elif self.state == 'load':
            loadname = os.path.join(self.save_path, self.load_filename)
            with open(loadname, 'rb') as file:
                vocab, encoded_file = pickle.load(file)

        return vocab, encoded_file


    def decode_sequence(self, sequence):
        decoded_sequence = self.bpe.decode(ids=sequence, ignore_ids=None)
        return decoded_sequence


def main():
    # specify all path
    filepath = os.path.join('datasets', 'law', 'law.txt')
    modelpath = os.path.join('saved_files', 'law', 'law.model')
    savepath = os.path.join('saved_files', 'law')
    load_filename = 'vocab_enc_100.pkl'

    # process data
    processor = DataProcessing(filepath, modelpath, savepath, load_filename, 'load')
    vocab, encoded_file = processor.data_preprocessing(vocab_size = 100)

    # dataloader

    print()

if __name__ == '__main__':
    main()

