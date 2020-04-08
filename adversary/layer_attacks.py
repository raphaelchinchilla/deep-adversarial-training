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


def SingleStep(net, l, y_true, eps, lamb, layer_num = 2, norm="inf"):
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
    d = torch.zeros_like(l, requires_grad=True)
    if layer_num ==2:
        y_hat = net.l4(net.l3(net.l2(l + d)))
    elif layer_num ==3:
        y_hat = net.l4(net.l3(l + d))
    elif layer_num ==4:
        y_hat = net.l4(l + d)
    reg = torch.norm(d.view(d.size(0),-1), 'fro',(1))
    criterion = nn.CrossEntropyLoss(reduction="none")
    # breakpoint()
    loss = criterion(y_hat, y_true) - lamb * reg
    
    loss.backward(
        gradient=torch.ones_like(y_true, dtype=torch.float), retain_graph=True
    )

    d_grad = d.grad.data
    if norm == "inf":
        perturbation = eps * d_grad.sign()
    elif norm == "ascend":
        perturbation = d_grad * eps
    else:
        perturbation = (
            d_grad
            * eps
            / d_grad.view(d.shape[0], -1).norm(p=norm, dim=-1).view(-1, 1, 1, 1)
        )

    return perturbation


# def ProjectedGradientDescent(
#     net,
#     x,
#     y_true,
#     device,
#     verbose=True,
#     data_params={"x_min": 0, "x_max": 1},
#     attack_params={
#         "norm": "inf",
#         "eps": 8.0 / 255.0,
#         "step_size": 8.0 / 255.0 / 10,
#         "num_steps": 100,
#         "random_start": True,
#         "num_restarts": 1,
#     },
# ):
#     """
#     Input : 
#         net : Neural Network (Classifier) 
#         x : Inputs to the net
#         y_true : Labels
#         data_params: Data parameters as dictionary
#                 x_min : Minimum legal value for elements of x
#                 x_max : Maximum legal value for elements of x
#         attack_params : Attack parameters as a dictionary
#                 norm : Norm of attack
#                 eps : Attack budget
#                 step_size : Attack budget for each iteration
#                 num_steps : Number of iterations
#                 random_start : Randomly initialize image with perturbation
#                 num_restarts : Number of restarts
#     Output:
#         perturbs : Perturbations for given batch
#     """
#     criterion = nn.CrossEntropyLoss(reduction="none")

#     # fooled_indices = np.array(y_true.shape[0])
#     perturbs = torch.zeros_like(x)

#     if verbose == True and attack_params["num_restarts"] > 1:
#         restarts = tqdm(range(attack_params["num_restarts"]))
#     else:
#         restarts = range(attack_params["num_restarts"])

#     for i in restarts:

#         if attack_params["random_start"] == True or attack_params["num_restarts"] > 1:
#             if attack_params["norm"] == "inf":
#                 perturb = (2 * torch.rand_like(x) - 1) * attack_params["eps"]
#             else:
#                 e = 2 * torch.rand_like(x) - 1
#                 perturb = (
#                     e
#                     * attack_params["eps"]
#                     / e.view(x.shape[0], -1)
#                     .norm(p=attack_params["norm"], dim=-1)
#                     .view(-1, 1, 1, 1)
#                 )
#         else:
#             perturb = torch.zeros_like(x, dtype=torch.float)

#         if verbose == True:
#             iters = tqdm(range(attack_params["num_steps"]))
#         else:
#             iters = range(attack_params["num_steps"])

#         for j in iters:
#             perturb += FastGradientSignMethod(
#                 net,
#                 torch.clamp(x + perturb, data_params["x_min"], data_params["x_max"]),
#                 y_true,
#                 attack_params["step_size"],
#                 attack_params["norm"],
#             )
#             if attack_params["norm"] == "inf":
#                 perturb = torch.clamp(
#                     perturb, -attack_params["eps"], attack_params["eps"]
#                 )
#             else:
#                 perturb = (
#                     perturb
#                     * attack_params["eps"]
#                     / perturb.view(x.shape[0], -1)
#                     .norm(p=attack_params["norm"], dim=-1)
#                     .view(-1, 1, 1, 1)
#                 )

#         if i == 0:
#             perturbs = perturb.data
#         else:
#             output = net(
#                 torch.clamp(x + perturb, data_params["x_min"], data_params["x_max"])
#             )
#             y_hat = output.argmax(dim=1, keepdim=True)

#             fooled_indices = (y_hat != y_true.view_as(y_hat)).nonzero()
#             perturbs[fooled_indices] = perturb[fooled_indices].data

#     return perturbs
