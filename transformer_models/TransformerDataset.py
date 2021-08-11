import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class TokenIDX():
    PAD_IDX = 0
    UNK_IDX = 1
    SOS_IDX = 2
    EOS_IDX = 3


class TransformerDataset(Dataset):
    def __init__(self, decoder_input, expected_output, batch_size, max_len):
        self.max_len = max_len + 1

        # expects decoder_input and expected_output to be a list of shape [N, sequence]
        # cuts off end of list if it does not fit into batch_size
        self.decoder_input = self.insert_SOS(decoder_input[0:len(decoder_input) // batch_size * batch_size])
        self.expected_output = self.insert_EOS(expected_output[0:len(expected_output) // batch_size * batch_size])

    def __len__(self):
        return len(self.expected_output)


    def __getitem__(self, idx):
        decoder_input = self.decoder_input[idx]
        expected_output = self.expected_output[idx]
        return decoder_input, expected_output


    def insert_EOS(self, dataset):
        # appends EOS to dataset consiting of sequences
        return [torch.cat(( torch.tensor(sequence), torch.tensor([TokenIDX.EOS_IDX]), torch.tensor((self.max_len - len(sequence))*[TokenIDX.PAD_IDX]) )) for sequence in dataset]


    def insert_SOS(self, dataset):
        # insert SOS in dataset consiting of 
        return [torch.cat(( torch.tensor([TokenIDX.SOS_IDX]), torch.tensor(sequence), torch.tensor((self.max_len - len(sequence))*[TokenIDX.PAD_IDX]) )) for sequence in dataset]

    
    def transform_to_tensor(self, dataset):
        # transform to tensor without insert EOS or SOS
        return [torch.tensor(sequence) for sequence in dataset]


    @staticmethod
    def collate_fn(batch):
        decoder_input, expected_output = zip(*batch)
        
        # pad batch
        #decoder_input = pad_sequence(decoder_input, batch_first=True, padding_value = TokenIDX.PAD_IDX)
        #expected_output = pad_sequence(expected_output, batch_first=True, padding_value = TokenIDX.PAD_IDX)
        decoder_input = torch.stack(decoder_input)
        expected_output = torch.stack(expected_output)
        # reshape
        expected_output_flat = expected_output.reshape(-1)

        return  decoder_input, expected_output, expected_output_flat
