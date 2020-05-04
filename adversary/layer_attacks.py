"""
Authors: Metehan Cekic
Date: 2020-03-09

Description: Attack models with l_{p} norm constraints

Attacks: FastGradientSignMethod(FGSM), ProjectedGradientDescent(PGD)
"""

from tqdm import tqdm
import numpy as np
import math
from apex import amp
# from copy import deep_copy

import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

def DistortNeuronsWithInput(model, x, y_true, lamb, mu, optimizer=None):
    model.eval()
    num_iters = 50
    device = model.parameters().__next__().device
    model.d[0] = 1*torch.randn(x.size()).to(device)
    model.d[1] = 1*torch.randn(model.n[1].size()).to(device)
    model.d[2] = 1*torch.randn(model.n[2].size()).to(device)
    model.d[3] = 1*torch.randn(model.n[3].size()).to(device)



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
        x=layers[0](x)

    def rho(z):
        return torch.sum(z**2,1)
        # return torch.norm(z,p=1,dim=1)


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
        model.d[0] = n_new[0] - x
        model.d[1] = n_new[1] - F.max_pool2d(F.leaky_relu(layers[1](n_new[0])), (2, 2))
        model.d[2] = n_new[2] - F.max_pool2d(F.leaky_relu(layers[2](n_new[1])), (2, 2)).view(x.size(0), -1)
        model.d[3] = n_new[3] - F.leaky_relu(layers[3](n_new[2]))




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
        dist_grads[i] = distortion.grad.data
    # breakpoint()

    return dist_grads
