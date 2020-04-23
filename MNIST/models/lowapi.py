
"""
Neural Network models for training and testing implemented in PyTorch
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from deep_adv.MNIST.models.tools import Normalize


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.norm = Normalize(mean=[0.1307], std=[0.3081])
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5,
                               stride=1, padding=2, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5,
                               stride=1, padding=2, bias=False)
        self.fc1 = nn.Linear(7 * 7 * 64, 1024, bias=False)
        self.fc2 = nn.Linear(1024, 10, bias=True)

    def forward(self, x):

        x = self.norm(x)
        self.NN = [None] * 4
        self.NN[0] = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) + self.d[0]
        self.NN[1] = F.max_pool2d(F.relu(self.conv2(self.NN[0])), (2, 2))
        self.NN[1] = self.NN[1].view(self.NN[1].size(0), -1) + self.d[1]
        self.NN[2] = F.relu(self.fc1(self.NN[1])) + self.d[2]
        self.NN[3] = self.fc2(self.NN[2])

        return self.NN[3]

    def create_parameters(self):

        self.d = [None] * 3
