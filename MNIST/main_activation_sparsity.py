'''
Main running script for testing and training of fixed bias model implemented with PyTorch

Example Run

python -m adv_ml.MNIST.main_activation_sparsity -at


'''


from __future__ import print_function

import torch
import torch.optim as optim
from torchvision import datasets, transforms

import numpy as np
from tqdm import tqdm
import matplotlib as mpl
from matplotlib import pyplot as plt
import os

from adv_ml.MNIST.models.lenet_activation_sparsity import CNN, FcNN
from adv_ml.train_test_functions_activation_sparsity import train, train_adversarial, test, test_adversarial
from adv_ml.MNIST.parameters import get_arguments
from adv_ml.MNIST.analysis.plotter import plot_perturbations, plot_filters, plot_features
from adv_ml.MNIST.read_datasets import MNIST

from adv_ml.adversary.norm_ball_attacks import ProjectedGradientDescent as PGD


def main():
    

    args = get_arguments()

    # Get same results for each training with same parameters !! 
    torch.manual_seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")


    # Bias Scalar
    bias_scalar={"delta1": args.delta1,
                 "delta2": args.delta2,
                 "delta3": args.delta3}

    print("#--------------- Bias Scalars -------------------#")
    print(bias_scalar)
    
    train_loader, test_loader = MNIST(args)
    x_min = 0.
    x_max = 1.

    # Decide on which model to use
    if args.model == 'CNN':
        model = CNN(bias_scalar).to(device)
    elif args.model == 'FcNN':
        model = FcNN(bias_scalar).to(device)

    print(model)

    # Which optimizer to be used for training
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    # optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # scheduler = MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    
    # Initialize alpha
    alpha = 1.0

    # Train network if args.train_network is set to True ( You can set that true by calling '-tr' flag, default is False )
    if args.train:
        if not args.adversarial:
            # Initialize model with standard model with layers without biases
            if args.initialize_model and args.model == 'CNN':
                model.load_state_dict(torch.load(args.directory + "checkpoints/" + args.dataset + "_" + args.model + "_0_0_0.pt"))
                print('Model initialized with following test accuracy: ')
                test(args, model, device, test_loader, alpha)


            for epoch in range(1, args.epochs + 1):
                alpha = train(args, model, device, train_loader, optimizer, epoch-1)
                test(args, model, device, test_loader, alpha)
                # scheduler.step()

            # Save model parameters
            if args.save_model:
                if not os.path.exists(args.directory + "checkpoints/"):
                    os.makedirs(args.directory + "checkpoints/")
                torch.save(model.state_dict(), args.directory + "checkpoints/" + args.dataset + "_" + args.model + f"_{int(bias_scalar['delta1']*100)}_{int(bias_scalar['delta2']*100)}_{int(bias_scalar['delta3']*100)}.pt")
        

        else:
            if args.initialize_model and args.model == 'CNN':
                model.load_state_dict(torch.load(args.directory + "checkpoints/" + args.dataset + "_" + args.model + "_adv_" + args.tr_norm + "_" + str(args.tr_eps) + "_0_0_0.pt"))
                print('Model initialized with following test accuracy: ')
                test(args, model, device, test_loader, alpha)

            data_params = {"x_min": x_min, "x_max": x_max}
            attack_params = {
                    "norm": args.tr_norm,
                    "eps": args.tr_epsilon,
                    "step_size": args.tr_step_size,
                    "num_steps": args.tr_num_iterations,
                    "random_start": args.tr_rand,
                    "num_restarts": args.tr_num_restarts}
            for epoch in range(1, args.epochs + 1):
                alpha = train_adversarial(args, model, device, train_loader, optimizer, epoch-1, data_params= data_params, attack_params =attack_params)
                test(args, model, device, test_loader, alpha)
                # scheduler.step()
            if args.save_model:
                if not os.path.exists(args.directory + "checkpoints/"):
                    os.makedirs(args.directory + "checkpoints/")
                torch.save(model.state_dict(), args.directory + "checkpoints/" + args.dataset + "_" + args.model + "_adv_" + args.tr_norm + "_" + str(args.tr_epsilon) + f"_{int(bias_scalar['delta1']*100)}_{int(bias_scalar['delta2']*100)}_{int(bias_scalar['delta3']*100)}.pt")

    else:
        if not args.adversarial:
            checkpoint_name = args.directory + "checkpoints/" + args.dataset + "_" + args.model + f"_{int(bias_scalar['delta1']*100)}_{int(bias_scalar['delta2']*100)}_{int(bias_scalar['delta3']*100)}.pt"
            model.load_state_dict(torch.load(checkpoint_name))
        else:
            checkpoint_name = args.directory + "checkpoints/" + args.dataset + "_" + args.model + "_adv_" + args.tr_norm + "_" + str(args.tr_epsilon) + f"_{int(bias_scalar['delta1']*100)}_{int(bias_scalar['delta2']*100)}_{int(bias_scalar['delta3']*100)}.pt"
            model.load_state_dict(torch.load(checkpoint_name))

        print("#--------------------------------------------------------------------#")
        print(f"#--     {checkpoint_name}     --#")
        print("#--------------------------------------------------------------------#")

        print("Clean test accuracy")
        test(args, model, device, test_loader)


    # plot_filters(args, model, fig_name = 'adv_')
    # breakpoint()
    # data_params = {'x_min': x_min,
    #                'x_max': x_max} 
    # attack_params = {'norm': 'inf',
    #                  'eps': args.eps, 
    #                  'step_size': args.Ss, 
    #                  'num_steps': args.Ni, 
    #                  'random_start': args.rand, 
    #                  'num_restarts': args.Nrest}
    # plot_features(args, model, test_loader, device, data_params, attack_params, fig_name = 'adv_')

    # Attack network if args.attack_network is set to True 
    # You can set that true by calling '-at' flag, default is "False"
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
        test_adversarial(args, model, device, test_loader, data_params = data_params, attack_params = attack_params, alpha = alpha)

        
if __name__ == '__main__':
    main()



