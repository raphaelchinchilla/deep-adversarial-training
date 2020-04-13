
"""
Neural Network models for training and testing implemented in PyTorch
"""
import math
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

class FCN(nn.Module):

	def __init__(self, device, num_neurons = [128, 64, 10]):
		super(FCN, self).__init__()

		self.device = device
		self.num_neurons = num_neurons
		self.norm = Normalize(mean=[0.1307], std=[0.3081])
		self.create_parameters()

	def forward(self, x):

		x = self.norm(x)
		x = x.view(x.size(0), -1)
		self.NN = [None] * len(self.num_neurons)
		self.NN[0] = F.relu(x @ self.w[0] + self.b[0]) + self.d[0]
		self.NN[1] = F.relu(self.NN[0] @ self.w[1] + self.b[1]) + self.d[1]
		self.NN[2] = F.softmax(self.NN[1] @ self.w[2], -1)

		return self.NN[2]
		
	def create_parameters(self):

		self.w = [None] * len(self.num_neurons)
		self.w[0]=torch.tensor(0.1*torch.randn([784, self.num_neurons[0]]),requires_grad=True, device=self.device)
		self.w[1]=torch.tensor(0.1*torch.randn([self.num_neurons[0], self.num_neurons[1]]), requires_grad=True, device=self.device)
		self.w[2]=torch.tensor(0.1*torch.randn([self.num_neurons[1], self.num_neurons[2]]), requires_grad=True, device=self.device)

		self.b = [None] * (len(self.num_neurons) - 1)
		self.b[0] = torch.zeros(self.num_neurons[0], requires_grad=True, device=self.device)
		self.b[1] = torch.zeros(self.num_neurons[1], requires_grad=True, device=self.device)

		self.d = [None] * (len(self.num_neurons) - 1)
	
	