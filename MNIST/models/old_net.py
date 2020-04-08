'''
Neural Network models for training and testing implemented in PyTorch
'''



from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# Standard 4 layer CNN implementation
class CNN(nn.Module):

    # 2 Conv layers, 2 Fc layers, 1 output layer

    def __init__(self):
        super(CNN, self).__init__()
        # 1 input image channel, 20 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 20, 3, 1, bias = True)
        self.conv2 = nn.Conv2d(20, 50, 5, 1, bias = True)
        self.fc1 = nn.Linear(4 * 4 * 50, 1000, bias = True)
        self.fc2 = nn.Linear(1000, 1000, bias = True)
        self.fc3 = nn.Linear(1000, 10)


    # If you want to see sparsity of neuron outputs for each layer, set show_sparsity = True
    def forward(self, x):


        self.features1 = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))

        # if show_sparsity:
        #     print('\nSparsity of first layer : {:.3f}'.format(np.count_nonzero(x.data.detach().cpu().clone().numpy())/x.data.numel()))

        self.features2 = F.max_pool2d(F.relu(self.conv2(self.features1)), (2, 2))

        # if show_sparsity:
        #     print('Sparsity of second layer : {:.3f}'.format(np.count_nonzero(x.data.detach().cpu().clone().numpy())/x.data.numel()))

        x = self.features2.view(-1, self.num_flat_features(self.features2))
        x = F.relu(self.fc1(x))
        
        # if show_sparsity:
        #     print('Sparsity of third layer : {:.3f}'.format(np.count_nonzero(x.data.detach().cpu().clone().numpy())/x.data.numel()))
            
        x = F.relu(self.fc2(x))
        
        # if show_sparsity:
        #     print('Sparsity of fourth layer : {:.3f}'.format(np.count_nonzero(x.data.detach().cpu().clone().numpy())/x.data.numel()))
        
        x = self.fc3(x)
        
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class FcNN(nn.Module):


    def __init__(self):
        super(FcNN, self).__init__()
        # 1 input image channel, 200 output channels, 5x5 square convolution
        # kernel
        self.fc1 = nn.Linear(784, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 10)

    # If you want to see sparsity of neuron outputs for each layer, set show_sparsity = True
    def forward(self, x):

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        
        # if show_sparsity:
        #     print('Sparsity of first layer : {:.3f}'.format(np.count_nonzero(x.data.detach().cpu().clone().numpy())/x.data.numel()))
            
        x = F.relu(self.fc2(x))
        
        # if show_sparsity:
        #     print('Sparsity of second layer : {:.3f}'.format(np.count_nonzero(x.data.detach().cpu().clone().numpy())/x.data.numel()))
        
        x = self.fc3(x)
        
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Fc5NN(nn.Module):


    def __init__(self):
        super(Fc5NN, self).__init__()
        # 1 input image channel, 200 output channels, 5x5 square convolution
        # kernel
        self.fc1 = nn.Linear(784, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 1000)
        self.fc4 = nn.Linear(1000, 1000)
        self.fc5 = nn.Linear(1000, 1000)
        self.fc6 = nn.Linear(1000, 10)

    # If you want to see sparsity of neuron outputs for each layer, set show_sparsity = True
    def forward(self, x):

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



