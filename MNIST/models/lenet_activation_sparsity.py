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

# simple Module to normalize an image
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)
    def forward(self, x):
        return (x - self.mean.type_as(x)[None,:,None,None]) / self.std.type_as(x)[None,:,None,None]


# Fixed Bias 4 layer CNN implementation
class CNN(nn.Module):

    # Fixed Bias initialize with bias scalars
    # b_i = ||w_i||_1 * bias_scalar * alpha
    # 2 Conv layers, 2 Fc layers, 1 output layer

    def __init__(self, bias_scalar):
        super(CNN, self).__init__()
       
        self.norm = Normalize(mean=[0.1307], std=[0.3081])
        self.conv1 = nn.Conv2d(1, 32, kernel_size = 5, stride = 1, padding = 2, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 5, stride = 1, padding = 2, bias=False)
        self.fc1 = nn.Linear(7 * 7 * 64, 1024, bias=False)
        self.fc2 = nn.Linear(1024, 10, bias=True)
        self.bias_scalar = bias_scalar

    # If you want to see sparsity of neuron outputs for each layer, set show_sparsity = True
    def forward(self, x, alpha = 1.0, show_sparsity = False):

        x = self.norm(x)
        bias = torch.sum(torch.abs(self.conv1.weight), dim=(1,2,3)).view(self.conv1.weight.shape[0],1,1) * self.bias_scalar['delta1'] * alpha     
        x = F.max_pool2d(F.relu(self.conv1(x) - bias), (2, 2))
        bias = torch.sum(torch.abs(self.conv2.weight), dim=(1,2,3)).view(self.conv2.weight.shape[0],1,1) * self.bias_scalar['delta2'] * alpha
        x = F.max_pool2d(F.relu(self.conv2(x) - bias), (2, 2))
        x = x.view(x.size(0), -1)
        bias = torch.sum(torch.abs(self.fc1.weight), dim=1) * self.bias_scalar['delta3'] * alpha
        x = F.relu(self.fc1(x) - bias)
        x = self.fc2(x)
        
        return x

# Fixed Bias 2 layer fully connected neural network implementation
class FcNN(nn.Module):

    # Fixed Bias initialize with bias scalars
    # b_i = ||w_i||_1 * bias_scalar * alpha
    # 2 Fc layers, 1 output layer

    def __init__(self, bias_scalar):
        super(FcNN, self).__init__()
        # 1 input image channel, 200 output channels, 5x5 square convolution
        # kernel
        self.fc1 = nn.Linear(784, 1000, bias = False)
        self.fc2 = nn.Linear(1000, 1000, bias = False)
        self.fc3 = nn.Linear(1000, 10)
        self.bias_scalar = bias_scalar

    # If you want to see sparsity of neuron outputs for each layer, set show_sparsity = True
    def forward(self, x, alpha = 1.0, show_sparsity = False):

        x = x.view(x.size(0), -1)
        bias = torch.sum(torch.abs(self.fc1.weight), dim=1) * self.bias_scalar['delta1'] * alpha
        x = F.relu(self.fc1(x) - bias)
        bias = torch.sum(torch.abs(self.fc2.weight), dim=1) * self.bias_scalar['delta2'] * alpha
        x = F.relu(self.fc2(x) - bias)
        x = self.fc3(x)
        
        return x

