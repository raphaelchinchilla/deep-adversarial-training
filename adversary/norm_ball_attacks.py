"""
Authors: Metehan Cekic
Date: 2020-03-09

Description: Attack models with l_{p} norm constraints

Attacks: FastGradientSignMethod(FGSM), ProjectedGradientDescent(PGD)
"""

from tqdm import tqdm
import numpy as np

import torch
import torchvision
from torch import nn
import torch.nn.functional as F


def FastGradientSignMethod(net, x, y_true, eps, norm="inf"):
    """
    Input : 
        net : Neural Network (Classifier) 
        x : Inputs to the net
        y_true : Labels
        eps : attack budget
        norm : attack budget norm
    Output:
        perturbation : Single step perturbation
    """
    e = torch.zeros_like(x, requires_grad=True)
    y_hat = net(x + e)
    criterion = nn.CrossEntropyLoss(reduction="none")
    loss = criterion(y_hat, y_true)
    loss.backward(
        gradient=torch.ones_like(y_true, dtype=torch.float), retain_graph=True
    )

    e_grad = e.grad.data
    if norm == "inf":
        perturbation = eps * e_grad.sign()
    else:
        perturbation = (
            e_grad
            * eps
            / e_grad.view(e.shape[0], -1).norm(p=norm, dim=-1).view(-1, 1, 1, 1)
        )

    return perturbation


def ProjectedGradientDescent(
    net,
    x,
    y_true,
    device,
    verbose=True,
    data_params={"x_min": 0, "x_max": 1},
    attack_params={
        "norm": "inf",
        "eps": 8.0 / 255.0,
        "step_size": 8.0 / 255.0 / 10,
        "num_steps": 100,
        "random_start": True,
        "num_restarts": 1,
    },
):
    """
    Input : 
        net : Neural Network (Classifier) 
        x : Inputs to the net
        y_true : Labels
        data_params: Data parameters as dictionary
                x_min : Minimum legal value for elements of x
                x_max : Maximum legal value for elements of x
        attack_params : Attack parameters as a dictionary
                norm : Norm of attack
                eps : Attack budget
                step_size : Attack budget for each iteration
                num_steps : Number of iterations
                random_start : Randomly initialize image with perturbation
                num_restarts : Number of restarts
    Output:
        perturbs : Perturbations for given batch
    """
    criterion = nn.CrossEntropyLoss(reduction="none")

    # fooled_indices = np.array(y_true.shape[0])
    perturbs = torch.zeros_like(x)

    if verbose == True and attack_params["num_restarts"] > 1:
        restarts = tqdm(range(attack_params["num_restarts"]))
    else:
        restarts = range(attack_params["num_restarts"])

    for i in restarts:

        if attack_params["random_start"] == True or attack_params["num_restarts"] > 1:
            if attack_params["norm"] == "inf":
                perturb = (2 * torch.rand_like(x) - 1) * attack_params["eps"]
            else:
                e = 2 * torch.rand_like(x) - 1
                perturb = (
                    e
                    * attack_params["eps"]
                    / e.view(x.shape[0], -1)
                    .norm(p=attack_params["norm"], dim=-1)
                    .view(-1, 1, 1, 1)
                )
        else:
            perturb = torch.zeros_like(x, dtype=torch.float)

        if verbose == True:
            iters = tqdm(range(attack_params["num_steps"]))
        else:
            iters = range(attack_params["num_steps"])

        for j in iters:
            perturb += FastGradientSignMethod(
                net,
                torch.clamp(x + perturb, data_params["x_min"], data_params["x_max"]),
                y_true,
                attack_params["step_size"],
                attack_params["norm"],
            )
            if attack_params["norm"] == "inf":
                perturb = torch.clamp(
                    perturb, -attack_params["eps"], attack_params["eps"]
                )
            else:
                perturb = (
                    perturb
                    * attack_params["eps"]
                    / perturb.view(x.shape[0], -1)
                    .norm(p=attack_params["norm"], dim=-1)
                    .view(-1, 1, 1, 1)
                )

        if i == 0:
            perturbs = perturb.data
        else:
            output = net(
                torch.clamp(x + perturb, data_params["x_min"], data_params["x_max"])
            )
            y_hat = output.argmax(dim=1, keepdim=True)

            fooled_indices = (y_hat != y_true.view_as(y_hat)).nonzero()
            perturbs[fooled_indices] = perturb[fooled_indices].data

    return perturbs
