
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

        self.norm = Normalize(mean=[0.], std=[1.])
        # self.norm = Normalize(mean=[0.1307], std=[0.3081])
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5,
                               stride=1, padding=2, bias=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5,
                               stride=1, padding=2, bias=True)
        self.fc1 = nn.Linear(7 * 7 * 64, 1024, bias=True)
        self.fc2 = nn.Linear(1024, 10, bias=True)
        self.create_parameters()
        self.forward(torch.zeros(1, 1, 28, 28))

    def forward(self, x):

        x = self.norm(x)
        self.n = [None] * 5
        self.n[0] = x + self.d[0]
        self.n[1] = F.max_pool2d(F.leaky_relu(self.conv1(self.n[0])), (2, 2)) + self.d[1]
        self.n[2] = F.max_pool2d(F.leaky_relu(self.conv2(self.n[1])), (2, 2))
        self.n[2] = self.n[2].view(self.n[2].size(0), -1) + self.d[2]
        self.n[3] = F.leaky_relu(self.fc1(self.n[2])) + self.d[3]
        self.n[4] = self.fc2(self.n[3])

        return self.n[4]

    def create_parameters(self):

        self.d = [0] * 4