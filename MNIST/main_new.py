# Simple MNIST fully-connected network for Raphael's paper. A lot of it is adapted from
# https://towardsdatascience.com/handwritten-digit-mnist-pytorch-977b5338e627.

import torch
import math
from time import time
from torchvision import datasets, transforms
from torch import nn

import torch.optim as optim
# import ipdb
from torch.nn.functional import relu
from torch.nn.functional import softmax

from deep_adv.adversary.norm_ball_attacks import ProjectedGradientDescent as PGD
from deep_adv.MNIST.parameters import get_arguments
from deep_adv.MNIST.read_datasets import MNIST
from deep_adv.MNIST.models.lowapi import FCN
from deep_adv.train_test_functions import (
	train,
	train_adversarial,
	train_deep_adversarial,
	test,
	test_adversarial,
)


def detach_list(x):
	xd= [None] * len(x)
	for i in range(len(x)):
		xd[i]=x[i].clone().detach()
	return xd

def rho(x):
	return torch.sum(x**2,1)

def main():

	args = get_arguments()

	torch.manual_seed(args.seed)

	use_cuda = not args.no_cuda and torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")

	train_loader, test_loader = MNIST(args)
	x_min = 0.0
	x_max = 1.0


	num_neurons=[128,64,10]
	activation = 'softmax'
	loss_type = 'mse'

	model = FCN(device = device).to(device)
	optimizer = optim.SGD(model.w + model.b, lr=args.learning_rate, momentum=args.momentum)

	print(model)



	weight_momen=list()
	for k in range(len(num_neurons)): weight_momen.append(torch.zeros(model.w[k].size()))
	bias_momen=torch.zeros(2)

	# cross_ent = nn.CrossEntropyLoss()
	criterion = nn.MSELoss()




	lr = args.learning_rate
	time0 = time()
	epochs = 3
	# training
	for e in range(epochs):
		running_loss = 0
		for batch_idx, (data, target) in enumerate(train_loader):

			target = torch.eye(10)[target]
			data, target = data.to(device), target.to(device)


			# Initializing perturbations
			lamb = 0.4 # budget for perturbation
			mu = 0.5	# adverserial step
			model.d[0] = math.sqrt(lamb)*torch.randn(data.size(0), num_neurons[0]).to(device)
			model.d[1] = math.sqrt(lamb)*torch.randn(data.size(0), num_neurons[1]).to(device)
			with torch.no_grad():
				output = model(data)

			model.NN[0].requires_grad_(True)
			model.NN[1].requires_grad_(True)
			wd=detach_list(model.w)
			bd=detach_list(model.b)
			# breakpoint()

			for count in range(10):
				perturb=torch.sum((target-softmax(model.NN[1]@wd[2],-1))**2,1)-lamb*rho(model.NN[0]-relu(data.view(data.size(0),-1)@wd[0]+bd[0]))-lamb*rho(model.NN[1]-relu(model.NN[0]@wd[1]+bd[1]))
				# Calculating the Jacobian
				perturb.view(-1,1).repeat(1,data.size(0)).backward(torch.eye(data.size(0)).to(device))
	#			# This one line does the same as this:
	# 			perturb=torch.sum((labels-softmax(NN[1]@wd[2],-1))**2,1)-1*rho(NN[0]-relu(images@wd[0]+bd[0]))-1*rho(NN[1]-relu(NN[0]@wd[1]+bd[1]))
	# 			for i in range(64):
	# 				perturb[i].backward(retain_graph=True)
				with torch.no_grad():
					model.NN[0]+=mu*model.NN[0].grad
					model.NN[1]+=mu*model.NN[1].grad
				model.NN[0].grad.zero_()
				model.NN[1].grad.zero_()
				if torch.any(torch.isnan(model.NN[0])) or torch.any(torch.isnan(model.NN[1])):
					breakpoint()

			model.d[0]=(model.NN[0]-relu(data.view(data.size(0),-1)@wd[0]+bd[0])).detach()
			model.d[1]=(model.NN[1]-relu(model.NN[0]@wd[1]+bd[1])).detach()
			# calculate output of model given batch of images
			output = model(data)
	# 		output = forward(images)
			# calculating the loss

			# loss = cross_ent(output, target)
			loss = criterion(output, target)
			# backprop

			optimizer.zero_grad()

			loss.backward()
			optimizer.step()
			# breakpoint()
			# with torch.no_grad():
			# 	for k in range(len(num_neurons)): weight_momen[k]=.9*weight_momen[k]-lr*model.w[k].grad
			# 	for k in range(len(num_neurons)): model.w[k]+=weight_momen[k]

			# 	bias_momen=.9*bias_momen-lr*b.grad
			# 	b+=bias_momen		# optimize weights
			# # clear old gradients
			# for k in range(len(num_neurons)): model.w[k].grad.zero_()
			# model.b.grad.zero_()



			running_loss += loss.item()
		else:
			print("Epoch {} - Training loss: {}".format(e, running_loss/len(train_loader)))
			print("\nTraining Time (in minutes) =",(time()-time0)/60)

	model.d[0] = 0
	model.d[1] = 0
	test(args, model, device, test_loader)

	data_params = {"x_min": x_min, "x_max": x_max}
	attack_params = {
		"norm": "inf",
		"eps": args.epsilon,
		"step_size": args.step_size,
		"num_steps": args.num_iterations,
		"random_start": args.rand,
		"num_restarts": args.num_restarts,
	}
	test_adversarial(
		args,
		model,
		device,
		test_loader,
		data_params=data_params,
		attack_params=attack_params,
	)

# saving the model
# torch.save(model, './my_mnist_model.pt')

if __name__ == '__main__':
	main()