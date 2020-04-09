"""
Neural Network models for training and testing implemented in PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# simple Module to normalize an image
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)
    def forward(self, x):
        return (x - self.mean.type_as(x)[None,:,None,None]) / self.std.type_as(x)[None,:,None,None]

# Standard 4 layer CNN implementation
class CNN(nn.Module):

    # 2 Conv layers, 2 Fc layers

    def __init__(self):
        super(CNN, self).__init__()

        self.norm = Normalize(mean=[0.1307], std=[0.3081])
        self.conv1 = nn.Conv2d(1, 32, kernel_size = 5, stride = 1, padding = 2, bias=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 5, stride = 1, padding = 2, bias=True)
        self.fc1 = nn.Linear(7 * 7 * 64, 1024, bias=True)
        self.fc2 = nn.Linear(1024, 10, bias=True)
       

    def forward(self, x):

        x = self.norm(x)
        n1 = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        n2 = F.max_pool2d(F.relu(self.conv2(n1)), (2, 2))
        n3 = x.view(n2.size(0), -1)
        n4 = F.relu(self.fc1(n3))
        x = self.fc2(n4)

        return x


