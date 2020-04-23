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

import torch
import torchvision
from torch import nn
import torch.nn.functional as F


def DistortNeurons(model, x, y_true, lamb, mu, optimizer=None):

    num_iters = 10
    device = model.parameters().__next__().device

    # model.d[0] = torch.randn(x.size(0), 32, 14, 14).to(
    #     device) / math.sqrt(lamb)
    # model.d[1] = torch.randn(x.size(0), 3136).to(device) / math.sqrt(lamb)
    # model.d[2] = torch.randn(x.size(0), 1024).to(device) / math.sqrt(lamb)

    _ = model(x)

    criterion = nn.CrossEntropyLoss(reduction="none")

    layers = list(model.children())[1:]

    for _ in range(num_iters):
        for i in range(len(model.NN)-1):
            model.NN[i].requires_grad_(True).retain_grad()

        loss = criterion(layers[-1](model.NN[-2]), y_true)
        loss -= lamb * \
            torch.norm(
                (model.NN[0] - F.max_pool2d(F.relu(layers[0](x)), (2, 2))).view(x.size(0), -1), p=1, dim=1)
        loss -= lamb * torch.norm((model.NN[1] - F.max_pool2d(F.relu(
            layers[1](model.NN[0])), (2, 2)).view(x.size(0), -1)).view(x.size(0), -1), p=1, dim=1)
        loss -= lamb * \
            torch.norm(model.NN[2] - F.relu(layers[2]
                                            (model.NN[1])), p=1, dim=1)

        if optimizer is not None:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward(gradient=torch.ones_like(
                    y_true, dtype=torch.float), retain_graph=True)
        else:
            loss.backward(gradient=torch.ones_like(y_true, dtype=torch.float), retain_graph=True)

        with torch.no_grad():
            # print(model.NN[0].grad[0, 0, 0, 0])
            model.NN[0] = model.NN[0] + mu * model.NN[0].grad
            model.NN[1] = model.NN[1] + mu * model.NN[1].grad
            model.NN[2] = model.NN[2] + mu * model.NN[2].grad

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
