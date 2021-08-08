
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class TransformerDataset(Dataset):

    def __init__(self, decoder_input, expected_output, vocab, batch_size, minibatch_size=None):
        self.vocab = vocab

        # expects decoder_input and expected_output to be a list of shape [N, sequence]
        # cuts off end of list if it does not fit into batch_size
        self.decoder_input = self.transform_to_tensor(decoder_input[0:len(decoder_input) // batch_size * batch_size])
        self.expected_output = self.insert_EOS(expected_output[0:len(expected_output) // batch_size * batch_size])

        # define shape
        if minibatch_size is not None:
            self.reshape_size = (batch_size // minibatch_size, minibatch_size, -1)
            self.reshape_flat = (batch_size // minibatch_size, -1)
        else:
            self.reshape_size = (batch_size, -1)
            self.reshape_flat = (-1)


    def __len__(self):
        return len(self.expected_output)


    def __getitem__(self, idx):
        decoder_input = self.decoder_input[idx]
        expected_output = self.expected_output[idx]
        return decoder_input, expected_output


    def insert_EOS(self, dataset):
        # appends EOS to dataset consiting of sequences
        return [torch.cat((torch.tensor(sequence), torch.tensor([self.vocab['<EOS>']]))) for sequence in dataset]


    def insert_SOS(self, dataset):
        # insert SOS in dataset consiting of sequences
        return [torch.cat((torch.tensor([self.vocab['<SOS>']]), torch.tensor(sequence))) for sequence in dataset]
    
    def transform_to_tensor(self, dataset):
        # transform to tensor without insert EOS or SOS
        return [torch.tensor(sequence) for sequence in dataset]



    def collate_fn(self, batch):

        decoder_input, expected_output = zip(*batch)

        # pad batch
        decoder_input = pad_sequence(decoder_input, batch_first=True, padding_value = self.vocab['<PAD>'])
        expected_output = pad_sequence(expected_output, batch_first=True, padding_value = self.vocab['<PAD>'])

        # reshape
        decoder_input = decoder_input.reshape(self.reshape_size)        
        expected_output = expected_output.reshape(self.reshape_size)
        expected_output_flat = expected_output.reshape(self.reshape_flat)

        return  decoder_input, expected_output, expected_output_flat
