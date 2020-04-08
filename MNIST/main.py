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

from deep_adv.MNIST.models.layers import LeNet
from deep_adv.train_test_functions import (
    train,
    train_adversarial,
    train_deep_adversarial,
    test,
    test_adversarial,
)
from deep_adv.MNIST.parameters import get_arguments
from deep_adv.adversary.norm_ball_attacks import ProjectedGradientDescent as PGD
from deep_adv.MNIST.read_datasets import MNIST


def main():

    args = get_arguments()

    # Get same results for each training with same parameters !!
    torch.manual_seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, test_loader = MNIST(args)
    x_min = 0.0
    x_max = 1.0


    model = LeNet().to(device)

    print(model)

    # breakpoint()

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
                train_deep_adversarial(
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
