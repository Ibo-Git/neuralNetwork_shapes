import torch
import torch
import torch.nn as nn
import os

class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()

        self.kernel_size_1 = 5
        self.kernel_size_2 = 5
        self.stride_1 = 1
        self.stride_2 = 2
        self.padding_1 = 10
        self.padding_2 = 10

        self.channel_output_conv_1 = 24
        self.channel_output_conv_2 = 36

        convw = self.conv2d_size_out( self.conv2d_size_out( w, self.kernel_size_1, self.stride_1, self.padding_1 ), self.kernel_size_2, self.stride_2, self.padding_2 )
        convh = self.conv2d_size_out( self.conv2d_size_out( h, self.kernel_size_1, self.stride_1, self.padding_1 ), self.kernel_size_2, self.stride_2, self.padding_2 )
        linear_input_size = self.channel_output_conv_2 * convw * convh

        self.net = nn.Sequential(
            nn.Conv2d(2, self.channel_output_conv_1, kernel_size=self.kernel_size_1, stride=self.stride_1, padding=self.padding_1),
            nn.BatchNorm2d(self.channel_output_conv_1),
            nn.ReLU(),
            nn.Conv2d(self.channel_output_conv_1, self.channel_output_conv_2, kernel_size=self.kernel_size_2, stride=self.stride_2, padding=self.padding_2),
            nn.BatchNorm2d(self.channel_output_conv_2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(linear_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, outputs),
        )

        self.softmax = nn.Softmax(dim=1)
  
    def forward(self, x):
        x = self.net(x)
        #x = self.softmax(x)
        return x
    
    
    def conv2d_size_out(self, size, kernel_size, stride, padding):
            return ((size - kernel_size + 2 * padding) // stride ) + 1


    def save(self, file_name = 'snake_model.pth'):
        model_folder_path = '.\saved_files'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


    def load(self, filename, model, device):
        model.load_state_dict(torch.load(os.path.join('.\saved_files', filename + '.pth'), map_location=torch.device(device)))
        model.eval()


        

