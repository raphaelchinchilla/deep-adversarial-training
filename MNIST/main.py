"""
Main running script for testing and training of fixed bias model implemented with PyTorch

Example Run

python -m deep_adv.MNIST.main -tr -adv -sm -at 


"""


from __future__ import print_function

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

import numpy as np
from tqdm import tqdm
import matplotlib as mpl
from matplotlib import pyplot as plt
import os

from adv_ml.MNIST.models.lenet import CNN, FcNN
from adv_ml.train_test_functions import (
    train,
    train_adversarial,
    test,
    test_adversarial,
)
from adv_ml.MNIST.parameters import get_arguments
from adv_ml.adversary.norm_ball_attacks import ProjectedGradientDescent as PGD
from adv_ml.MNIST.read_datasets import MNIST


def main():

    args = get_arguments()

    # Get same results for each training with same parameters !!
    torch.manual_seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, test_loader = MNIST(args)
    x_min = 0.0
    x_max = 1.0

    # Decide on which model to use
    if args.model == "CNN":
        model = CNN().to(device)
    elif args.model == "FcNN":
        model = FcNN().to(device)
    else:
        raise NotImplementedError


    print(model)

    # Which optimizer to be used for training
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    # optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # scheduler = MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

    # Train network if args.train is set to True (You can set that true by calling '-tr' flag, default is False)
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

            for epoch in range(1, args.epochs + 1):
                train_adversarial(
                    args,
                    model,
                    device,
                    train_loader,
                    optimizer,
                    epoch - 1,
                    data_params,
                    attack_params,
                )
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
                    + "_adv_"
                    + args.tr_norm
                    + "_"
                    + str(args.tr_epsilon)
                    + ".pt",
                )

   
    else:
        if not args.adversarial:
            checkpoint_name = args.directory + "checkpoints/" + args.model + ".pt"
            model.load_state_dict(torch.load(checkpoint_name))
        else:
            checkpoint_name = args.directory + "checkpoints/" + args.model + "_adv_" + args.tr_norm + "_" + str(args.tr_epsilon) + ".pt"
            model.load_state_dict(torch.load(checkpoint_name))

        print("#--------------------------------------------------------------------#")
        print(f"#--     {checkpoint_name}     --#")
        print("#--------------------------------------------------------------------#")

        print("Clean test accuracy")
        test(args, model, device, test_loader)

    # plot_filters(model)

    # if True:
    #     model.eval()
    #     norm = None
    #     # data_params = {'x_min': x_min,
    #     #                'x_max': x_max}
    #     # attack_params = {'norm': 2,
    #     #                  'eps': args.epsilon,
    #     #                  'step_size': args.step_size,
    #     #                  'num_steps': args.num_iterations,
    #     #                  'random_start': args.rand,
    #     #                  'num_restarts': args.num_restarts}
    #     data_params = {"x_min": x_min, "x_max": x_max}
    #     attack_params = {
    #         "norm": "inf",
    #         "eps": 0.3,
    #         "step_size": 0.01,
    #         "num_steps": 100,
    #         "random_start": True,
    #         "num_restarts": 1,
    #     }
    #     for data, target in test_loader:
    #         data, target = data.to(device), target.to(device)
    #         perturbs = PGD(
    #             model,
    #             data,
    #             target,
    #             device,
    #             data_params=data_params,
    #             attack_params=attack_params,
    #         )
    #         plot_perturbations(
    #             model,
    #             data,
    #             target,
    #             torch.clamp(data + perturbs, 0, 1),
    #             norm=attack_params["norm"],
    #             budget=attack_params["eps"],
    #             x_min=x_min,
    #             x_max=x_max,
    #         )

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


if __name__ == "__main__":
    main()
