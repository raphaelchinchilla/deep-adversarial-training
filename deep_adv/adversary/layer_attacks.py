"""
Authors: Metehan Cekic and Raphael Chinchilla
Date: 2020-03-09


"""

from tqdm import tqdm
import numpy as np
import math
from apex import amp
import time

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim


#from deepillusion.torchattacks._utils import clip


def DistortNeuronsConjugateGradient(model, x, y_true, attack_params):
    '''
    Descriptions:
        Conjugate Gradient
        No line search but diminishing step
        Penalty for n[0] (abs(n[0]-x)>0.3, n[0]-0.5>=0.5)
    '''
    # Parameters
    alpha = attack_params["step_size"] # attack step size
    num_iters = attack_params["num_steps"] # number of attack iterations
    eps_input = attack_params["eps"] # value to clamp input disturbance
    lamb_in = attack_params["lamb_in"] # weight for input layer
    lamb_la = attack_params["lamb_la"] # weight for inside layers
    eps_init_layers= math.sqrt(1/(lamb_la)) # "std" for the initialization of disturbances
    debug = False # set to true to activate break points and prints
    #Code
    model.eval()
    device = model.parameters().__next__().device
    criterion = nn.CrossEntropyLoss(reduction="none")

    for p in model.parameters():
        p.requires_grad = False

    layers = list(model.children())

    # Initializing the Nodes
    n = [None] * (len(model.n)-1)
    for i in range(len(n)):
        n[i] = torch.empty(model.n[i].size()).to(device)

    # Initializing the auxiliary variables
    aux = [None] * (len(model.n)-2)
    for i in range(len(aux)):
        aux[i] = torch.empty(model.n[i+1].size()).to(device)


    # Initializing several auxiliary lists as empty
    direct = [None] * (len(n))
    for i in range(len(n)):
        direct[i] = torch.zeros(model.n[i].size()).to(device)
    beta=torch.zeros((x.size(0),1),device=device)
    norm_grad=torch.zeros((x.size(0),1),device=device)
    loss=torch.zeros((x.size(0)),device=device)


    # Defining some function
    crit=torch.nn.MSELoss(reduction='none')
    def rho(z,w):
        return crit(z.view(x.size(0), -1),w.view(x.size(0), -1)).sum(1)
    def reg(z):
        return (z.view(x.size(0), -1)**2).sum(1)
    def batch_dot_prod(z,w):
        return (z.view(x.size(0), -1)*w.view(x.size(0), -1)).sum(1).view(x.size(0), -1)


    with torch.no_grad():
        x=layers[0](x)

    # Initializing the value of the nodes
    with torch.no_grad():
        n[0]=x+eps_input*(2*torch.rand(x.size())-1).to(device)
        n[0]=torch.clamp(n[0],0.,1.)
    n[0].requires_grad_(True)
    aux[0]=F.max_pool2d(F.leaky_relu(layers[1](n[0])), (2, 2))
    with torch.no_grad():
        n[1]=aux[0]+eps_init_layers*(2*torch.rand(model.n[1].size())-1).to(device)
    n[1].requires_grad_(True)
    aux[1]=F.max_pool2d(F.leaky_relu(layers[2](n[1])), (2, 2)).view(x.size(0), -1)
    with torch.no_grad():
        n[2]=aux[1]+eps_init_layers*(2*torch.rand(model.n[2].size())-1).to(device)
    n[2].requires_grad_(True)
    aux[2]=F.leaky_relu(layers[3](n[2]))
    with torch.no_grad():
        n[3]=aux[2]+eps_init_layers*(2*torch.rand(model.n[3].size())-1).to(device)
    n[3].requires_grad_(True)




    solver = 'running'
    iter=0
    while iter<= num_iters and solver=='running':
        iter+=1
        loss_prev=loss.clone()
        # Calculating -1*loss
        # The loss is a vector because we need to keep track of each loss in the
        # batch to possibly restart the search direction if direction is not ascent
        loss = lamb_in*reg(((n[0] - x).abs()-eps_input).relu()) # penalizing input disturbances larger than eps_input
        loss += lamb_in*reg(((n[0] - 0.5).abs()-0.5).relu()) # penalizing input disturbances that cause first layer to no be in [0,1]
        loss += lamb_la*rho(n[1],aux[0]) 
        loss += lamb_la*rho(n[2],aux[1]) 
        loss += lamb_la*rho(n[3],aux[2])
        loss += -criterion(layers[-1](n[3]), y_true)

        loss.mean().backward()


        with torch.no_grad():
            # We now compute the Conjugate Gradient search direction
            norm_grad_prev=norm_grad.clone()
            norm_grad=torch.zeros((x.size(0),1),device=device)
            for i in range(len(n)):
                norm_grad+=batch_dot_prod(n[i].grad,n[i].grad)
            if iter > 1:
                loss_diff=loss_prev-loss
                # Calculating beta
                beta=norm_grad/norm_grad_prev
                beta[loss_diff<0]=0 # restart direction if not descent direction
            if debug:
                print("Iter:", iter, ", Loss max: ", loss.max().data.cpu().numpy(), ", Loss min:", loss.min().data.cpu().numpy(), ", Output Loss", criterion(layers[-1](n[3]), y_true).max().data.cpu().numpy())
                print("      Grad n[0]: ", n[0].grad.abs().mean().data.cpu().numpy()," Grad n[1]: ", n[1].grad.abs().mean().data.cpu().numpy()," Grad n[2]: ", n[2].grad.abs().mean().data.cpu().numpy()," Grad n[3]: ", n[3].grad.abs().mean().data.cpu().numpy())
                print("      Beta:", beta.abs().max().data.cpu().numpy(),"norm_grad:", norm_grad.max().data.cpu().numpy(), "alpha: ", alpha.min().data.cpu().numpy())
            for i in range(len(n)):
                # Updating the search directions direct according to the conjugate gradient algorithm
                direct[i].view(x.size(0),-1).mul_(beta) # Multiply the previous step direction of each batch by the value of beta that is equivalent. Equivalent to direct[i]*=beta.view([-1]+[1]*(direct[i].ndim-1)).expand_as(direct[i])
                direct[i]+=-n[i].grad # Update the search direction with the new step to go
                n[i]+=alpha*direct[i] # Using a decreasing step size alpha/iter to take into account progression in the search direction
                n[i].grad.zero_()


        # Simulating the system
        n[0].requires_grad_(True)
        aux[0]=F.max_pool2d(F.leaky_relu(layers[1](n[0])), (2, 2))
        n[1].requires_grad_(True)
        aux[1]=F.max_pool2d(F.leaky_relu(layers[2](n[1])), (2, 2)).view(x.size(0), -1)
        n[2].requires_grad_(True)
        aux[2]=F.leaky_relu(layers[3](n[2]))
        n[3].requires_grad_(True)
        
        with torch.no_grad():
            if torch.any(torch.isnan(loss)) or torch.any(torch.isnan(n[0])) or torch.any(torch.isnan(n[1])) or torch.any(torch.isnan(n[2])):
                print("Iter:", iter)
                raise ValueError('Diverged')


    with torch.no_grad():
        # Calculating the layer disturbance d
        d = [None] * (len(n))
        for i in range(len(n)):
            d[i] = torch.zeros_like(model.n[i])

        d[0] = (n[0] - x)
        d[1] = (n[1] - aux[0])
        d[2] = (n[2] - aux[1])
        d[3] = (n[3] - aux[2])


        if debug:
            print(" d[0]: ", d[0].abs().max().data.cpu().numpy() ," d[1]: ", d[1].abs().max().data.cpu().numpy()," d[2]: ", d[2].abs().max().data.cpu().numpy()," d[3]: ", d[3].abs().max().data.cpu().numpy())
            # breakpoint()
            # time.sleep(0.000001)

        for i in range(len(n)):
            model.d[i] = d[i].clone()


    for p in model.parameters():
        p.requires_grad = True

