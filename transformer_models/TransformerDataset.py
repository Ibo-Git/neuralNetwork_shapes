import torch
from torch.nn.utils.rnn import pad_packed_sequence, pad_sequence
from torch.utils.data import Dataset


class TokenIDX():
    PAD_IDX = 0
    UNK_IDX = 1
    SOS_IDX = 2
    EOS_IDX = 3


class TransformerDataset(Dataset):
    def __init__(self, decoder_input, expected_output, batch_size, max_len):
        # expects decoder_input and expected_output to be a list of shape [N, sequence]
        # cuts off end of list if it does not fit into batch_size
        self.decoder_input = self.insert_SOS(decoder_input[0:len(decoder_input) // batch_size * batch_size])
        self.expected_output = self.insert_EOS(expected_output[0:len(expected_output) // batch_size * batch_size])
        self.decoder_input = pad_sequence(self.decoder_input, batch_first=True, padding_value=TokenIDX.PAD_IDX)
        self.expected_output = pad_sequence(self.expected_output, batch_first=True, padding_value=TokenIDX.PAD_IDX)
        self.max_len = max_len + 1

    def __len__(self):
        return len(self.expected_output)


    def __getitem__(self, idx):
        decoder_input = self.decoder_input[idx]
        expected_output = self.expected_output[idx]
        return decoder_input, expected_output, self.max_len


    def insert_EOS(self, dataset):
        # appends EOS to dataset consiting of sequences
        return [torch.cat(( torch.tensor(sequence), torch.tensor([TokenIDX.EOS_IDX]))) for sequence in dataset]


    def insert_SOS(self, dataset):
        # insert SOS in dataset consiting of 
        return [torch.cat(( torch.tensor([TokenIDX.SOS_IDX]), torch.tensor(sequence))) for sequence in dataset]

    
    def transform_to_tensor(self, dataset):
        # transform to tensor without insert EOS or SOS
        return [torch.tensor(sequence) for sequence in dataset]


    @staticmethod
    def collate_fn(batch):
        decoder_input, expected_output, max_len = zip(*batch)
        max_len = max_len[0]

        decoder_input = list(decoder_input)
        expected_output = list(expected_output)
        # adding tensor of max length to data in order to pad every single batch to the same length
        for i in range(len(decoder_input)):
            decoder_input[i] = torch.cat(( decoder_input[i], torch.tensor([TokenIDX.PAD_IDX]*(max_len-len(decoder_input[i]))) ))
            expected_output[i] = torch.cat(( expected_output[i], torch.tensor([TokenIDX.PAD_IDX]*(max_len-len(expected_output[i]))) ))
       
       #pad_tensor = torch.tensor(max_len)
        # pad batch
        #decoder_input = pad_sequence(decoder_input, batch_first=True, padding_value = TokenIDX.PAD_IDX)
        #expected_output = pad_sequence(expected_output, batch_first=True, padding_value = TokenIDX.PAD_IDX)

        decoder_input = torch.stack(decoder_input)
        expected_output = torch.stack(expected_output)

        # reshape
        expected_output_flat = expected_output.reshape(-1)

        return  decoder_input, expected_output, expected_output_flat
