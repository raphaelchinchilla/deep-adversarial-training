# Simple MNIST fully-connected network for Raphael's paper. A lot of it is adapted from
# https://towardsdatascience.com/handwritten-digit-mnist-pytorch-977b5338e627.

# example use python -m deep_adv.MNIST.main_master -tr -adv --epochs 5

import torch
import math
from time import time
from torchvision import datasets, transforms
from torch import nn
import os

import torch.optim as optim
# import ipdb
from torch.nn.functional import relu
from torch.nn.functional import softmax

from deep_adv.adversary.norm_ball_attacks import ProjectedGradientDescent as PGD
from deep_adv.MNIST.parameters import get_arguments
from deep_adv.MNIST.read_datasets import MNIST
from deep_adv.MNIST.models.lowapi import CNN
from deep_adv.train_test_functions import (
    train,
    train_adversarial,
    train_deep_adversarial,
    test,
    test_adversarial,
)


def main():

    args = get_arguments()

    torch.manual_seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, test_loader = MNIST(args)
    x_min = 0.0
    x_max = 1.0

    model = CNN().to(device)
    optimizer = optim.SGD(
        model.parameters(), lr=args.learning_rate, momentum=args.momentum)

    print(model)

    if args.train:
        if not args.adversarial:
            for epoch in range(1, args.epochs + 1):
                train(args, model, device, train_loader, optimizer, epoch - 1)
                test(args, model, device, test_loader)
                # scheduler.step()

            # Save model parameters
            if args.save_model:
                if not os.path.exists(args.directory + "checkpoints/"):
                    os.makedirs(args.directory + "checkpoints/")
                torch.save(
                    model.state_dict(), args.directory + "checkpoints/" + args.model + ".pt"
                    )
        else:
            data_params = {"x_min": x_min, "x_max": x_max}
            attack_params = {
                "norm": args.tr_norm,
                "eps": args.tr_epsilon,
                "step_size": args.tr_step_size,
                "num_steps": args.tr_num_iterations,
                "random_start": args.tr_rand,
                "num_restarts": args.tr_num_restarts,
                }

            lamb = 0.01  # budget for perturbation
            mu = 0.5  # adverserial step
            for epoch in range(1, args.epochs + 1):
                train_deep_adversarial(
                    args,
                    model,
                    device,
                    train_loader,
                    optimizer,
                    epoch - 1,
                    lamb,
                    mu,
                    data_params,
                    attack_params,
                    )
                model.d = [0, 0, 0]
                test(args, model, device, test_loader)
                # scheduler.step()

            # Save model parameters
            if args.save_model:
                if not os.path.exists(args.directory + "checkpoints/"):
                    os.makedirs(args.directory + "checkpoints/")
                torch.save(
                    model.state_dict(),
                    args.directory
                    + "checkpoints/"
                    + args.model
                    + "_deep_adv"
                    + "_"
                    + str(lamb)
                    + "_"
                    + str(mu)
                    # + args.tr_norm
                    # + "_"
                    # + str(args.tr_epsilon)
                    + ".pt",
                    )

    else:
        if not args.adversarial:
            checkpoint_name = args.directory + "checkpoints/" + args.model + ".pt"
            model.load_state_dict(torch.load(checkpoint_name))
        else:
            checkpoint_name = args.directory + "checkpoints/" + args.model + \
                "_deep_adv_" + ".pt"
            model.load_state_dict(torch.load(checkpoint_name))

        print("#--------------------------------------------------------------------#")
        print(f"#--     {checkpoint_name}     --#")
        print("#--------------------------------------------------------------------#")

        print("Clean test accuracy")
        test(args, model, device, test_loader)

    # Attack network if args.attack_network is set to True (You can set that true by calling '-at' flag, default is False)
    if args.attack_network:
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
