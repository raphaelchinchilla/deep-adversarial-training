"""
Authors: Metehan Cekic and Raphael Chinchilla
Date: 2020-03-09

Description: Attack models with l_{p} norm constraints

Attacks: FastGradientSignMethod(FGSM), ProjectedGradientDescent(PGD)
"""

from tqdm import tqdm
import numpy as np
import math
from apex import amp
# from copy import deep_copy
import time

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim


def DistortNeuronsConjugateGradient(model, x, y_true, lamb, mu, optimizer=None):
    model.eval()
    num_iters = 200
    eps_input = 0.3 # value to clamp input disturbance
    eps_layers = 2 # value to clamp layer disturbance
    lamb_layers = 20 # how many times regularization in inside layers
    eps_init_layers= 1# math.sqrt(1/(lamb*lamb_layers)) # "variance" for the initialization of disturbances
    lamb_reg = 0.1 # regularization of the intermediate layers
    debug = False # set to true to activate break points and prints
    solver = 'CG'
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


    # Initializing several auxilia ry lists as empty
    grad_prev = [None] * (len(n))
    direct = [None] * (len(n))
    for i in range(len(n)):
        grad_prev[i] = torch.empty(model.n[i].size()).to(device)
        direct[i] = torch.zeros(model.n[i].size()).to(device)
    beta=torch.zeros((x.size(0),1),device=device)
    norm_grad=torch.zeros((x.size(0),1),device=device)
    norm_grad_prev=torch.zeros((x.size(0),1),device=device)


    with torch.no_grad():
        x=layers[0](x)

    # Initializing the value of the nodes

    # with torch.no_grad():
    #     n[0]=x+eps_input*(2*torch.rand(x.size())-1).to(device)
    #     n[0]=torch.clamp(n[0],0.,1.)
    # n[0].requires_grad_(True)
    # aux[0]=F.max_pool2d(F.leaky_relu(layers[1](n[0])), (2, 2))
    # with torch.no_grad():
    #     n[1]=aux[0]+eps_init_layers*torch.randn(model.n[1].size()).to(device)
    # n[1].requires_grad_(True)
    # aux[1]=F.max_pool2d(F.leaky_relu(layers[2](n[1])), (2, 2)).view(x.size(0), -1)
    # with torch.no_grad():
    #     n[2]=aux[1]+eps_init_layers*torch.randn(model.n[2].size()).to(device)
    # n[2].requires_grad_(True)
    # aux[2]=F.leaky_relu(layers[3](n[2]))
    # with torch.no_grad():
    #     n[3]=aux[2]+eps_init_layers*torch.randn(model.n[3].size()).to(device)
    # n[3].requires_grad_(True)

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


    # crit=torch.nn.L1Loss(reduction='none')
    crit=torch.nn.MSELoss(reduction='none')
    def rho(z,w):
        return crit(z.view(x.size(0), -1),w.view(x.size(0), -1)).sum(1)

    def reg(z):
        return (z.view(x.size(0), -1)**2).sum(1)


    def batch_dot_prod(z,w):
        return (z.view(x.size(0), -1)*w.view(x.size(0), -1)).sum(1).view(x.size(0), -1)

    iter=0
    while iter<= num_iters and solver!='solved' and solver!='failed':
        iter+=1

        # Calculating the loss
        loss =  rho(n[0],x) - lamb_reg*reg(n[0])
        loss += lamb_layers*rho(n[1],aux[0]) - lamb_reg*reg(n[1])
        loss += lamb_layers*rho(n[2],aux[1]) - lamb_reg*reg(n[2])
        loss += lamb_layers*rho(n[3],aux[2]) - lamb_reg*reg(n[3])
        loss += -criterion(layers[-1](n[3]), y_true)/lamb

        loss.backward(gradient=torch.ones_like(y_true, dtype=torch.float))


        with torch.no_grad():
            if debug:
                print("Iter:", iter, ", Solver: ", solver,  ", Loss max: ", loss.max().data.cpu().numpy(), ", Loss min:", loss.min().data.cpu().numpy(), ", Output Loss", criterion(layers[-1](n[3]), y_true).max().data.cpu().numpy())
                print(" Grad n[0]: ", n[0].grad.abs().mean().data.cpu().numpy()," Grad n[1]: ", n[1].grad.abs().mean().data.cpu().numpy()," Grad n[2]: ", n[2].grad.abs().mean().data.cpu().numpy()," Grad n[3]: ", n[3].grad.abs().mean().data.cpu().numpy())
            if solver=='CG':
                if iter == 3: # Not really using the initial oto allow some noise to be cleared out
                    norm_grad_init=torch.zeros((x.size(0),1),device=device)
                    for i in range(len(n)):
                        norm_grad_init+=batch_dot_prod(n[i].grad,n[i].grad)
                if iter>1:
                    beta=torch.zeros((x.size(0),1),device=device)
                    for i in range(len(n)):
                        beta+=batch_dot_prod(n[i].grad,n[i].grad)
                        # beta+=batch_dot_prod(n[i].grad,n[i].grad-1*grad_prev[i])
                    beta/=norm_grad
                    beta=torch.relu(beta)
                if debug:
                    print("Beta:", beta.abs().max().data.cpu().numpy(),"norm_grad:", norm_grad.max().data.cpu().numpy())
                norm_grad_prev=norm_grad.clone()
                norm_grad=torch.zeros((x.size(0),1),device=device)
                for i in range(len(n)):
                    direct[i].view(x.size(0),-1).mul_(beta) # Multiply each line of bactch by the value of beta that is equivalent. Equivalent to direct[i]*=beta.view([-1]+[1]*(direct[i].ndim-1)).expand_as(direct[i])
                    direct[i]+=-n[i].grad
                    n[i]+=mu/iter*direct[i]
                    norm_grad+=batch_dot_prod(n[i].grad,n[i].grad)
                    grad_prev[i]=n[i].grad.clone()
                    n[i].grad.zero_()
                if iter>=10:
                    if norm_grad_prev.max()<norm_grad.max():
                        solver='failed'
                    if (norm_grad/norm_grad_init).max()<1e-3:
                        solver='solved'
            elif solver=='GD':
                # norm_grad=float("-inf")
                # if iter==1:
                    # for i in range(len(n)):
                        # norm_grad_init_inv[i] = 1/torch.norm(n[i].grad)
                for i in range(len(n)):
                    # norm_grad=max(norm_grad,torch.norm(norm_grad_init_inv[i]*n[i].grad))
                    # direct[i].mul_(0.1).add_(-mu/iter*n[i].grad)
                    # n[i].add_(direct[i])
                    n[i]+=-mu/iter*n[i].grad
                    n[i].grad.zero_()

        # Simulating the systemq
        with torch.no_grad():
            n[0]=x+torch.clamp(n[0] - x,-eps_input,eps_input)
            n[0]=torch.clamp(n[0],0.,1.)
        n[0].requires_grad_(True)
        aux[0]=F.max_pool2d(F.leaky_relu(layers[1](n[0])), (2, 2))
        with torch.no_grad():
            n[1]=aux[0]+torch.clamp(n[1] - aux[0],-eps_layers,eps_layers)
        n[1].requires_grad_(True)
        aux[1]=F.max_pool2d(F.leaky_relu(layers[2](n[1])), (2, 2)).view(x.size(0), -1)
        with torch.no_grad():
            n[2]=aux[1]+torch.clamp(n[2] - aux[1],-eps_layers,eps_layers)
        n[2].requires_grad_(True)
        aux[2]=F.leaky_relu(layers[3](n[2]))
        with torch.no_grad():
            n[3]=aux[2]+torch.clamp(n[3] - aux[2],-eps_layers,eps_layers)
        n[3].requires_grad_(True)

    if torch.any(torch.isnan(loss)) or torch.any(torch.isnan(n[0])) or torch.any(torch.isnan(n[1])) or torch.any(torch.isnan(n[2])):
        raise ValueError('Diverged')
    # if torch.any(loss.abs()>1e5) or torch.any(n[0].abs()>1e3) or torch.any(n[1].abs()>1e3) or torch.any(n[2].abs()>1e3) or torch.any(n[3].abs()>1e3):
        # breakpoint()

    with torch.no_grad():
        if solver=='solved':
            model.d[0] = n[0] - x
            model.d[1] = n[1] - aux[0]
            model.d[2] = n[2] - aux[1]
            model.d[3] = n[3] - aux[2]
            if debug:
                print(" d[0]: ", model.d[0].abs().max().data.cpu().numpy() ," d[1]: ", model.d[1].abs().max().data.cpu().numpy()," d[2]: ", model.d[2].abs().max().data.cpu().numpy()," d[3]: ", model.d[3].abs().max().data.cpu().numpy())
                # breakpoint()
                # time.sleep(0.000001)
        else:
            model.d[0] = 0
            model.d[1] = 0
            model.d[2] = 0
            model.d[3] = 0
            if debug:
                print("NOT SOLVED")
                # breakpoint()
                # time.sleep(0.000001)

    for p in model.parameters():
        p.requires_grad = True

def DistortNeuronsManual(model, x, y_true, lamb, mu, optimizer=None):
    model.eval()
    num_iters = 10
    eps_input = 0.3 # value to clamp input disturbance
    eps_layers = 10 # value to clamp layers disturbance
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

    # Initializing the SGD momentum
    momentum = [None] * (len(model.n)-1)
    for i in range(len(n)):
        momentum[i] = torch.zeros(model.n[i].size()).to(device)

    with torch.no_grad():
        x=layers[0](x)

    # Initializing the value of the nodes

    with torch.no_grad():
        n[0]=x+eps_input*(2*torch.rand(x.size())-1).to(device)
    # verify device and verify requires_grad
    n[0].requires_grad_(True)
    aux[0]=F.max_pool2d(F.leaky_relu(layers[1](n[0])), (2, 2))
    with torch.no_grad():
        n[1]=aux[0]+eps_layers*(2*torch.rand(model.n[1].size())-1).to(device)
    n[1].requires_grad_(True)
    aux[1]=F.max_pool2d(F.leaky_relu(layers[2](n[1])), (2, 2)).view(x.size(0), -1)
    with torch.no_grad():
        n[2]=aux[1]+eps_layers*(2*torch.rand(model.n[2].size())-1).to(device)
    n[2].requires_grad_(True)
    aux[2]=F.leaky_relu(layers[3](n[2]))
    with torch.no_grad():
        n[3]=aux[2]+eps_layers*(2*torch.rand(model.n[3].size())-1).to(device)
    n[3].requires_grad_(True)


    def rho(z):
        # return torch.sum(z**2,1)
        return torch.norm(z,p=1,dim=1)

    # The idea of the next following steps is to partially normalize the gradients
    # such that mu is more a less constant in all epochs and batches. I will be calculate on the first batch
    norm_grad_init_inv = [None] * (len(model.n)-1)


    iter=0
    norm_grad=float("inf")
    while iter<= num_iters: #and norm_grad<=0.001:
        iter+=1

        # Calculating the loss
        loss = lamb * rho((n[0] - x).view(x.size(0), -1))
        loss += lamb * rho((n[1] - aux[0]).view(x.size(0), -1))
        loss += lamb * rho(n[2] - aux[1])
        loss += lamb * rho(n[3] - aux[2])
        loss += -criterion(layers[-1](n[3]), y_true)


        if torch.any(torch.isnan(loss)) or torch.any(torch.isnan(n[0])) or torch.any(torch.isnan(n[1])) or torch.any(torch.isnan(n[2])):
            breakpoint()
        if torch.any(loss.abs()>1e5) or torch.any(n[0].abs()>1e3) or torch.any(n[1].abs()>1e3) or torch.any(n[2].abs()>1e3) or torch.any(n[3].abs()>1e3):
            breakpoint()

        if optimizer is not None:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward(gradient=torch.ones_like(y_true, dtype=torch.float))
        else:
            loss.backward(gradient=torch.ones_like(y_true, dtype=torch.float))


        norm_grad=float("-inf")
        with torch.no_grad():
            if iter==1:
                for i in range(len(n)):
                    norm_grad_init_inv[i] = 1/torch.norm(n[i].grad)
            for i in range(len(n)):
                norm_grad=max(norm_grad,torch.norm(norm_grad_init_inv[i]*n[i].grad))
                momentum[i].mul_(0.01).add_(-mu*n[i].grad)
                n[i].add_(momentum[i])
                n[i].grad.zero_()


        # Simulating the system
        with torch.no_grad():
            n[0]=x+torch.clamp(n[0] - x,-eps_input,eps_input)
            n[0]=torch.clamp(n[0],-1,1)
        n[0].requires_grad_(True)
        aux[0]=F.max_pool2d(F.leaky_relu(layers[1](n[0])), (2, 2))
        with torch.no_grad():
            n[1]=aux[0]+torch.clamp(n[1] - aux[0],-eps_layers,eps_layers)
        n[1].requires_grad_(True)
        aux[1]=F.max_pool2d(F.leaky_relu(layers[2](n[1])), (2, 2)).view(x.size(0), -1)
        with torch.no_grad():
            n[2]=aux[1]+torch.clamp(n[2] - aux[1],-eps_layers,eps_layers)
        n[2].requires_grad_(True)
        aux[2]=F.leaky_relu(layers[3](n[2]))
        with torch.no_grad():
            n[3]=aux[2]+torch.clamp(n[3] - aux[2],-eps_layers,eps_layers)
        n[3].requires_grad_(True)


    with torch.no_grad():
        # print("Norm grad: ",norm_grad, " Loss: ", torch.max(loss))
        model.d[0] = n[0] - x
        model.d[1] = n[1] - aux[0]
        model.d[2] = n[2] - aux[1]
        model.d[3] = n[3] - aux[2]

    for p in model.parameters():
        p.requires_grad = True


def DistortNeuronsWithInput(model, x, y_true, lamb, mu, optimizer=None):
    model.eval()
    num_iters = 100
    device = model.parameters().__next__().device
    model.d[0] = (2*torch.randn(x.size())).to(device)
    model.d[1] = (2*torch.randn(model.n[1].size())).to(device)
    model.d[2] = (2*torch.randn(model.n[2].size())).to(device)
    model.d[3] = (2*torch.randn(model.n[3].size())).to(device)


    _ = model(x)

    criterion = nn.CrossEntropyLoss(reduction="none")

    layers = list(model.children())

    for p in model.parameters():
        p.requires_grad = False

    n_new = [None] * (len(model.n)-1)
    for i in range(len(n_new)):
        n_new[i] = model.n[i].detach().clone()
        n_new[i].requires_grad_(True)

    # breakpoint()
    # optimizer_dn = optim.Adam(n_new, lr=mu)
    optimizer_dn = optim.SGD(n_new, lr=mu,momentum=0.9)
    with torch.no_grad():
        x=layers[0](x)

    def rho(z):
        return torch.sum(z**2,1)
        # return torch.norm(z,p=1,dim=1)

    # def rho(z,w):
    #     zw=(z.view(x.size(0), -1)-w.view(x.size(0), -1))
    #     return torch.sum(zw**2,1)
    #     # return torch.norm(zw,p=1,dim=1)


    for _ in range(num_iters):
        # loss layer is added from last layer to input layer
        loss = -criterion(layers[-1](n_new[3]), y_true)
        loss += lamb * rho(n_new[3] - F.leaky_relu(layers[3](n_new[2])))
        loss += lamb * rho(n_new[2] - F.max_pool2d(F.leaky_relu(layers[2](n_new[1])), (2, 2)).view(x.size(0), -1))
        loss += lamb * rho((n_new[1] - F.max_pool2d(F.leaky_relu(layers[1](n_new[0])), (2, 2))).view(x.size(0), -1))
        loss += lamb * rho((n_new[0] - x).view(x.size(0), -1))


        if torch.any(torch.isnan(loss)) or torch.any(torch.isnan(n_new[0])) or torch.any(torch.isnan(n_new[1])) or torch.any(torch.isnan(n_new[2])):
            breakpoint()
        if torch.any(loss.abs()>1e5) or torch.any(n_new[0].abs()>1e3) or torch.any(n_new[1].abs()>1e3) or torch.any(n_new[2].abs()>1e3) or torch.any(n_new[3].abs()>1e3):
            breakpoint()
        if optimizer is not None:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward(gradient=torch.ones_like(
                    y_true, dtype=torch.float))
        else:
            loss.backward(gradient=torch.ones_like(y_true, dtype=torch.float))

        optimizer_dn.step()
        optimizer_dn.zero_grad()

        with torch.no_grad():
            n_new[0]=torch.clamp(n_new[0],-1,1)



    with torch.no_grad():
        model.d[0] = n_new[0] - x
        model.d[1] = n_new[1] - F.max_pool2d(F.leaky_relu(layers[1](n_new[0])), (2, 2))
        model.d[2] = n_new[2] - F.max_pool2d(F.leaky_relu(layers[2](n_new[1])), (2, 2)).view(x.size(0), -1)
        model.d[3] = n_new[3] - F.leaky_relu(layers[3](n_new[2]))




    for p in model.parameters():
        p.requires_grad = True

def DistortNeuronsBounded(model, x, y_true, lamb, mu, optimizer=None):
    model.eval()
    num_iters = 100
    eps = 0.3 # value to clamp tensor
    device = model.parameters().__next__().device
    model.d[0] = eps*(2*torch.rand(x.size())-1).to(device)
    model.d[1] = eps*(2*torch.rand(model.n[1].size())-1).to(device)
    model.d[2] = eps*(2*torch.rand(model.n[2].size())-1).to(device)
    model.d[3] = eps*(2*torch.rand(model.n[3].size())-1).to(device)



    _ = model(x)

    criterion = nn.CrossEntropyLoss(reduction="none")

    layers = list(model.children())

    for p in model.parameters():
        p.requires_grad = False

    n_new = [None] * (len(model.n)-1)
    for i in range(len(n_new)):
        n_new[i] = model.n[i].detach().clone()
        n_new[i].requires_grad_(True)

    aux = [None] * (len(model.n)-2)


    # breakpoint()
    optimizer_dn = optim.Adam(n_new, lr=mu)
    # optimizer_dn = optim.SGD(n_new, lr=mu,momentum=0.9)
    with torch.no_grad():
        x=layers[0](x)

    def rho(z):
        return torch.sum(z**2,1)
        # return torch.norm(z,p=1,dim=1)

    # def rho(z,w):
    #     zw=(z.view(x.size(0), -1)-w.view(x.size(0), -1))
    #     return torch.sum(zw**2,1)
    #     # return torch.norm(zw,p=1,dim=1)


    for _ in range(num_iters):
        # loss layer is added from last layer to input layer

        with torch.no_grad():
            n_new[0]+=-n_new[0]+x+torch.clamp(n_new[0] - x,-eps,eps)
        aux[0]=F.max_pool2d(F.leaky_relu(layers[1](n_new[0])), (2, 2))
        with torch.no_grad():
            n_new[1]+=-n_new[1]+aux[0]+torch.clamp(n_new[1] - aux[0],-eps,eps)
        aux[1]=F.max_pool2d(F.leaky_relu(layers[2](n_new[1])), (2, 2)).view(x.size(0), -1)
        with torch.no_grad():
            n_new[2]+=-n_new[2]+aux[1]+torch.clamp(n_new[2] - aux[1],-eps,eps)
        aux[2]=F.leaky_relu(layers[3](n_new[2]))
        with torch.no_grad():
            n_new[3]+=-n_new[3]+aux[2]+torch.clamp(n_new[3] - aux[2],-eps,eps)


        loss = -criterion(layers[-1](n_new[3]), y_true)
        loss += lamb * rho(n_new[3] - aux[2])
        loss += lamb * rho(n_new[2] - aux[1])
        loss += lamb * rho((n_new[1] - aux[0]).view(x.size(0), -1))
        loss += lamb * rho((n_new[0] - x).view(x.size(0), -1))


        if torch.any(torch.isnan(loss)) or torch.any(torch.isnan(n_new[0])) or torch.any(torch.isnan(n_new[1])) or torch.any(torch.isnan(n_new[2])):
            breakpoint()
        if torch.any(loss.abs()>1e5) or torch.any(n_new[0].abs()>1e3) or torch.any(n_new[1].abs()>1e3) or torch.any(n_new[2].abs()>1e3) or torch.any(n_new[3].abs()>1e3):
            breakpoint()
        if optimizer is not None:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward(gradient=torch.ones_like(y_true, dtype=torch.float))
        else:
            loss.backward(gradient=torch.ones_like(y_true, dtype=torch.float))




        optimizer_dn.step()
        optimizer_dn.zero_grad()

        with torch.no_grad():
            n_new[0]=torch.clamp(n_new[0],-1,1)



    with torch.no_grad():
        model.d[0] = torch.clamp(n_new[0] - x,-eps,+eps)
        model.d[1] = torch.clamp(n_new[1] - F.max_pool2d(F.leaky_relu(layers[1](n_new[0])), (2, 2)),-eps,+eps)
        model.d[2] = torch.clamp(n_new[2] - F.max_pool2d(F.leaky_relu(layers[2](n_new[1])), (2, 2)).view(x.size(0), -1),-eps,+eps)
        model.d[3] = torch.clamp(n_new[3] - F.leaky_relu(layers[3](n_new[2])),-eps,+eps)




    for p in model.parameters():
        p.requires_grad = True

def DistortNeuronsStepeestDescent(model, x, y_true, lamb, mu, optimizer=None):

    num_iters = 10
    device = model.parameters().__next__().device

    model.d[0] = torch.randn(x.size(0), 32, 14, 14).to(
        device) / math.sqrt(lamb)
    model.d[1] = torch.randn(x.size(0), 3136).to(device) / math.sqrt(lamb)
    model.d[2] = torch.randn(x.size(0), 1024).to(device) / math.sqrt(lamb)

    _ = model(x)

    criterion = nn.CrossEntropyLoss(reduction="none")

    layers = list(model.children())[1:]

    for _ in range(num_iters):
        for i in range(len(model.NN)-1):
            model.NN[i].requires_grad_(True).retain_grad()

        loss = criterion(layers[-1](model.NN[-2]), y_true)
        loss -= lamb * \
            torch.norm(
                (model.NN[0] - F.max_pool2d(F.leaky_relu(layers[0](x)), (2, 2))).view(x.size(0), -1), p=1, dim=1)
        loss -= lamb * torch.norm((model.NN[1] - F.max_pool2d(F.leaky_relu(
            layers[1](model.NN[0])), (2, 2)).view(x.size(0), -1)).view(x.size(0), -1), p=1, dim=1)
        loss -= lamb * \
            torch.norm(model.NN[2] - F.leaky_relu(layers[2]
                                                  (model.NN[1])), p=1, dim=1)


        if optimizer is not None:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward(gradient=torch.ones_like(
                    y_true, dtype=torch.float), retain_graph=True)
        else:
            loss.backward(gradient=torch.ones_like(y_true, dtype=torch.float), retain_graph=True)

        with torch.no_grad():
            # print(model.NN[0].grad[0, 0, 0, 0])
            model.NN[0] = model.NN[0] + mu * model.NN[0].grad.sign()
            model.NN[1] = model.NN[1] + mu * model.NN[1].grad.sign()
            model.NN[2] = model.NN[2] + mu * model.NN[2].grad.sign()

        # print(model.NN[0][0, 0, 0, 0])
        #
        # model.NN[1].grad.zero_()
        # model.NN[2].grad.zero_()
    # breakpoint()

    for i in range(len(model.NN)-1):
        model.NN[i].requires_grad_(False)

    model.d[0] = model.NN[0] - F.max_pool2d(F.relu(layers[0](x)), (2, 2))
    model.d[1] = model.NN[1] - \
        F.max_pool2d(F.relu(layers[1](model.NN[0])),
                     (2, 2)).view(x.size(0), -1)
    model.d[2] = model.NN[2] - F.relu(layers[2](model.NN[1]))


def DistortNeurons(model, x, y_true, lamb, mu, optimizer=None):
    model.eval()
    num_iters = 30
    device = model.parameters().__next__().device

    # model.d[0] = torch.randn(x.size(0), 32, 14, 14).to(
    #     device) / math.sqrt(lamb)
    # model.d[1] = torch.randn(x.size(0), 3136).to(device) / math.sqrt(lamb)
    # model.d[2] = torch.randn(x.size(0), 1024).to(device) / math.sqrt(lamb)
    _ = model(x)

    criterion = nn.CrossEntropyLoss(reduction="none")

    layers = list(model.children())

    for p in model.parameters():
        p.requires_grad = False

    n_new = [None] * (len(model.n)-1)
    for i in range(len(n_new)):
        n_new[i] = model.n[i].detach().clone()
        n_new[i].requires_grad_(True)

    # breakpoint()
    optimizer_dn = optim.Adam(n_new, lr=mu)
    # optimizer_dn = optim.SGD(n_new, lr=mu,momentum=0.1)
    with torch.no_grad():
        x_new=F.max_pool2d(F.leaky_relu(layers[1](layers[0](x))), (2, 2))

    for _ in range(num_iters):
        loss = -criterion(layers[-1](n_new[-1]), y_true)
        loss += lamb * \
            torch.norm((n_new[0] - x_new).view(x.size(0), -1), p=1, dim=1)
        loss += lamb * torch.norm((n_new[1] - F.max_pool2d(F.leaky_relu(
            layers[2](n_new[0])), (2, 2)).view(x.size(0), -1)).view(x.size(0), -1), p=1, dim=1)
        loss += lamb * \
            torch.norm(n_new[2] - F.leaky_relu(layers[3](n_new[1])), p=1, dim=1)

        # loss += lamb * \
        #     ((n_new[0] - F.max_pool2d(F.leaky_relu(layers[1](layers[0](x))), (2, 2))).view(x.size(0), -1)).pow(2).sum(1)
        # loss += lamb * ((n_new[1] - F.max_pool2d(F.leaky_relu(
        #     layers[2](n_new[0])), (2, 2)).view(x.size(0), -1)).view(x.size(0), -1)).pow(2).sum(1)
        # loss += lamb *(n_new[2] - F.leaky_relu(layers[3](n_new[1]))).pow(2).sum(1)


        if torch.any(torch.isnan(loss)) or torch.any(torch.isnan(n_new[0])) or torch.any(torch.isnan(n_new[1])) or torch.any(torch.isnan(n_new[2])):
            breakpoint()
        if torch.any(loss.abs()>1e2) or torch.any(n_new[0].abs()>1e2) or torch.any(n_new[1].abs()>1e2) or torch.any(n_new[2].abs()>1e2):
            breakpoint()
        if optimizer is not None:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward(gradient=torch.ones_like(
                    y_true, dtype=torch.float))
        else:
            loss.backward(gradient=torch.ones_like(y_true, dtype=torch.float))


        # with torch.no_grad():
        #     # print(model.n[0].grad[0, 0, 0, 0])
        #     model.n[0] = model.n[0] + mu * model.n[0].grad
        #     model.n[1] = model.n[1] + mu * model.n[1].grad
        #     model.n[2] = model.n[2] + mu * model.n[2].grad

        optimizer_dn.step()
        optimizer_dn.zero_grad()

        # print(model.n[0][0, 0, 0, 0])
        #
        # model.n[1].grad.zero_()
        # model.n[2].grad.zero_()
    # breakpoint()

    # for i in range(len(model.n)-1):
    #     model.n[i].requires_grad_(False)
    with torch.no_grad():
        model.d[0] = n_new[0] - F.max_pool2d(F.leaky_relu(layers[1](layers[0](x))), (2, 2))
        model.d[1] = n_new[1] - \
            F.max_pool2d(F.leaky_relu(layers[2](n_new[0])),
                         (2, 2)).view(x.size(0), -1)
        model.d[2] = n_new[2] - F.leaky_relu(layers[3](n_new[1]))



    for p in model.parameters():
        p.requires_grad = True



def distort_before_activation_old(model, x, y_true, lamb, mu, optimizer=None):

    num_iters = 10
    device = model.parameters().__next__().device

    model.d[0] = torch.randn(x.size(0), *tuple(model.m[0].shape[1:])).to(
        device) * 0.3
    # model.d[1] = torch.randn(x.size(0), *tuple(model.m[1].shape[1:])).to(device) * 0.3
    # model.d[2] = torch.randn(x.size(0), *tuple(model.m[2].shape[1:])).to(device) * 0.3

    _ = model(x)

    criterion = nn.CrossEntropyLoss(reduction="none")

    layers = list(model.children())[1:]

    m_new = [None] * (len(model.m)-1)
    # for i in range(len(m_new)):
    #     m_new[i] = model.m[i].detach().clone()
    #     m_new[i].requires_grad_(True).retain_grad()
    m_new[0] = model.m[0].detach().clone()
    m_new[0].requires_grad_(True).retain_grad()

    optimizer_dn = optim.Adam(m_new, lr=mu)

    for _ in range(num_iters):

        loss = -criterion(layers[-1](m_new[-1]), y_true)
        loss += lamb * \
            torch.norm(
                (m_new[0] - layers[0](x)).view(x.size(0), -1), p=1, dim=1)
        # loss += lamb * torch.norm((m_new[1] -
        #                            layers[1](F.max_pool2d(F.leaky_relu(m_new[0]), (2, 2)))).view(x.size(0), -1), p=1, dim=1)
        # loss += lamb * \
        #     torch.norm(m_new[2] - layers[2]
        #                (F.max_pool2d(F.leaky_relu(m_new[1]), (2, 2)).view(m_new[1].size(0), -1)), p=1, dim=1)

        if optimizer is not None:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward(gradient=torch.ones_like(
                    y_true, dtype=torch.float), retain_graph=True)
        else:
            loss.backward(gradient=torch.ones_like(y_true, dtype=torch.float), retain_graph=True)

        # with torch.no_grad():
        #     # print(model.m[0].grad[0, 0, 0, 0])
        #     model.m[0] = model.m[0] + mu * model.m[0].grad
        #     model.m[1] = model.m[1] + mu * model.m[1].grad
        #     model.m[2] = model.m[2] + mu * model.m[2].grad

        optimizer_dn.step()
        optimizer_dn.zero_grad()

        # print(model.m[0][0, 0, 0, 0])
        #
        # model.m[1].grad.zero_()
        # model.m[2].grad.zero_()
    # breakpoint()

    # for i in range(len(model.m)-1):
    #     model.m[i].requires_grad_(False)

    model.d[0] = m_new[0] - layers[0](x)
    # model.d[1] = m_new[1] - layers[1](F.max_pool2d(F.leaky_relu(m_new[0]), (2, 2)))
    # model.d[2] = m_new[2] - \
    #     layers[2](F.max_pool2d(F.leaky_relu(m_new[1]), (2, 2)).view(m_new[1].size(0), -1))


def distort_before_activation(model, x, y_true, lamb, mu, optimizer=None):

    num_iters = 10
    device = model.parameters().__next__().device

    model.d[0] = torch.randn(x.size(0), *tuple(model.m[0].shape[1:])).to(
        device) * 1.5
    # model.d[1] = torch.randn(x.size(0), *tuple(model.m[1].shape[1:])).to(device) * 0.3
    # model.d[2] = torch.randn(x.size(0), *tuple(model.m[2].shape[1:])).to(device) * 0.3

    criterion = nn.CrossEntropyLoss(reduction="none")

    model.d[0].requires_grad_(True)

    optimizer_dn = optim.Adam([model.d[0]], lr=mu)

    for _ in range(num_iters):

        output = model(x)

        loss = -criterion(output, y_true)
        loss += lamb * torch.norm(model.d[0].view(x.size(0), -1), p=1, dim=1)
        loss += lamb * torch.norm((m_new[1] -
                                    layers[1](F.max_pool2d(F.leaky_relu(m_new[0]), (2, 2)))).view(x.size(0), -1), p=1, dim=1)
        loss += lamb * \
            torch.norm(m_new[2] - layers[2]
                        (F.max_pool2d(F.leaky_relu(m_new[1]), (2, 2)).view(m_new[1].size(0), -1)), p=1, dim=1)

        if optimizer is not None:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward(gradient=torch.ones_like(
                    y_true, dtype=torch.float), retain_graph=True)
        else:
            loss.backward(gradient=torch.ones_like(y_true, dtype=torch.float), retain_graph=True)

        optimizer_dn.step()
        optimizer_dn.zero_grad()


def SingleStep(net, layers, y_true, eps, lamb, norm="inf"):
    """
    Input :
        net : Neural Network (Classifier)
        l : Specific Layer
        y_true : Labels
        eps : attack budget
        norm : attack budget norm
    Output:
        perturbation : Single step perturbation
    """
    distortions = [None] * len(layers)
    for i, layer in enumerate(layers):
        distortions[i] = torch.zeros_like(layer, requires_grad=True)

    # layer_functions = [net.l3, net.l4]
    # a = net.l2(layers[0]+distortions[0])
    # for i, f in enumerate(layer_functions):
    #     a = f(a+distortions[i+1])
    # y_hat = a

    y_hat = net.l4(
        net.l3(net.l2(layers[0] + distortions[0])+distortions[1])+distortions[2])

    reg = 0
    for distortion in distortions:
        reg += torch.norm(distortion.view(distortion.size(0), -1), 'fro', (1))

    criterion = nn.CrossEntropyLoss(reduction="none")
    # breakpoint()
    loss = criterion(y_hat, y_true) - lamb * reg

    loss.backward(
        gradient=torch.ones_like(y_true, dtype=torch.float), retain_graph=True
        )

    dist_grads = [None] * len(distortions)
    for i, distortion in enumerate(distortions):
        dist_grads[i] = distortion.grad.data.cpu().numpy()
    # breakpoint()

    return dist_grads
