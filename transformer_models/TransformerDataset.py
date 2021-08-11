import torch
from torch.nn.utils.rnn import pad_packed_sequence, pad_sequence
from torch.utils.data import Dataset


class TokenIDX():
    PAD_IDX = 0
    UNK_IDX = 1
    SOS_IDX = 2
    EOS_IDX = 3


class TransformerDataset(Dataset):
    def __init__(self, decoder_input, expected_output, batch_size):
        self.decoder_input = decoder_input[0:len(decoder_input) // batch_size * batch_size]
        self.expected_output = expected_output[0:len(expected_output) // batch_size * batch_size]


    def __len__(self):
        return len(self.expected_output)


    def __getitem__(self, idx):
        decoder_input = self.decoder_input[idx]
        expected_output = self.expected_output[idx]
        return decoder_input, expected_output


    @staticmethod
    def collate_fn(batch):
        decoder_input, expected_output = zip(*batch)

        decoder_input = torch.stack(decoder_input)
        expected_output = torch.stack(expected_output)
        expected_output_flat = expected_output.reshape(-1)

        return  decoder_input, expected_output, expected_output_flat
