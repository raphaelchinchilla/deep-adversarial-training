"""
Main running script for testing and training of fixed bias model implemented with PyTorch

Example Run

python -m deep_adv.MNIST.main -at -tra dn -l 0.001 -m 0.1 -tr -sm --epochs 10
"""


from __future__ import print_function

import numpy as np
from tqdm import tqdm
import matplotlib as mpl
from matplotlib import pyplot as plt
import os
from os import path
import logging
import time


from apex import amp
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from deep_adv.MNIST.models.layers import LeNet
from deep_adv.MNIST.models.lowapi import CNN1, CNN2, CNN3
from deep_adv.train_test_functions import (
    train,
    train_adversarial,
    train_deep_adversarial,
    train_fgsm_adversarial,
    test,
    test_adversarial,
    test_fgsm
)
from deep_adv.MNIST.parameters import get_arguments
from deep_adv.MNIST.read_datasets import MNIST

logger = logging.getLogger(__name__)


def main():

    args = get_arguments()

    if not os.path.exists(args.directory + 'logs'):
        os.mkdir(args.directory + 'logs')

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        # filename=args.directory + 'logs/' + args.model + '_' + args.tr_attack + '.log',
        handlers=[
            logging.FileHandler(args.directory + 'logs/' + args.model + \
                                '_' + args.tr_attack + '.log'),
            logging.StreamHandler()
            ])
    logger.info(args)
    logger.info("\n")

    # Get same results for each training with same parameters !!
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, test_loader = MNIST(args)
    x_min = 0.0
    x_max = 1.0

    if args.tr_attack == "None" or args.tr_attack == "fgsm" or args.tr_attack == "pgd":
        model = LeNet().to(device)
    elif args.tr_attack == "dn":
        model = CNN1().to(device)
    elif args.tr_attack == "dba":
        model = CNN2().to(device)
    elif args.tr_attack == "dnwi":
        model = CNN3().to(device)
    else:
        raise NotImplementedError

    # logging.info(model)

    # Which optimizer to be used for training
    # optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    # optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # scheduler = MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    optimizer = optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    lr_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.lr_min,
                                                  max_lr=args.lr_max, step_size_up=lr_steps/2,
                                                  step_size_down=lr_steps/2)

    amp_args = dict(opt_level=args.opt_level, loss_scale=args.loss_scale, verbosity=False)
    if args.opt_level == 'O2':
        amp_args['master_weights'] = args.master_weights
    model, optimizer = amp.initialize(model, optimizer, **amp_args)

    lamb = args.lamb  # budget for perturbation
    mu = args.mu  # adverserial step
    if args.train:
        if args.tr_attack == "None":
            logger.info("Standard training")
            logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')
            for epoch in range(1, args.epochs + 1):
                start_time = time.time()
                train_loss, train_acc = train(model, train_loader, optimizer, scheduler)
                test_loss, test_acc = test(model, test_loader)
                end_time = time.time()
                lr = scheduler.get_lr()[0]
                logger.info(f'{epoch} \t {end_time - start_time:.0f} \t \t {lr:.4f} \t {train_loss:.4f} \t {train_acc:.4f}')
                logger.info(f'Test  \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}')

            # Save model parameters
            if args.save_model:
                if not os.path.exists(args.directory + "checkpoints/"):
                    os.makedirs(args.directory + "checkpoints/")
                torch.save(model.state_dict(),
                           path.join(args.directory + "checkpoints/", args.model + ".pt"))
        elif args.tr_attack == "dn" or args.tr_attack == "dba" or args.tr_attack == "dnwi":
            logger.info("Distorting neurons")
            logger.info('Epoch \t Seconds \t LR \t \t Clean Loss \t Clean Acc \t Dist Loss \t Dist Acc')
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
                start_time = time.time()
                deep_adv_train_args = dict(model=model,
                                           train_loader=train_loader,
                                           optimizer=optimizer,
                                           scheduler=scheduler,
                                           lamb=lamb,
                                           mu=mu,
                                           data_params=data_params,
                                           attack_params=attack_params)
                # breakpoint()
                clean_loss, clean_acc, dist_loss, dist_acc = train_deep_adversarial(
                    **deep_adv_train_args)
                model.d = [0, 0, 0, 0]
                test_loss, test_acc = test(model, test_loader)
                end_time = time.time()
                lr = scheduler.get_lr()[0]
                logger.info(f'{epoch} \t {end_time - start_time:.0f} \t \t {lr:.4f} \t {clean_loss:.4f} \t {clean_acc:.4f} \t {dist_loss:.4f} \t {dist_acc:.4f}')
                logger.info(f'Test  \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}')
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
        elif args.tr_attack == "fgsm":
            logger.info("FGSM adversarial training")
            logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')
            data_params = {"x_min": x_min, "x_max": x_max}
            attack_params = {
                "norm": args.tr_norm,
                "eps": args.tr_epsilon,
                "alpha": args.tr_alpha,
                "step_size": args.tr_step_size,
                "num_steps": args.tr_num_iterations,
                "random_start": args.tr_rand,
                "num_restarts": args.tr_num_restarts,
                }

            for epoch in range(1, args.epochs + 1):
                start_time = time.time()
                fgsm_train_args = dict(model=model,
                                       train_loader=train_loader,
                                       optimizer=optimizer,
                                       scheduler=scheduler,
                                       data_params=data_params,
                                       attack_params=attack_params)
                # breakpoint()
                train_loss, train_acc = train_fgsm_adversarial(**fgsm_train_args)

                test_loss, test_acc = test(model, test_loader)
                end_time = time.time()
                lr = scheduler.get_lr()[0]
                logger.info(f'{epoch} \t {end_time - start_time:.0f} \t \t {lr:.4f} \t {train_loss:.4f} \t {train_acc:.4f}')
                logger.info(f'Test  \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}')
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
                    + "_fgsm_adv"
                    + "_"
                    + str(attack_params["alpha"])
                    + "_"
                    + str(attack_params["eps"])
                    # + args.tr_norm
                    # + "_"
                    # + str(args.tr_epsilon)
                    + ".pt",
                    )
        elif args.tr_attack == "pgd":
            logger.info("PGD adversarial training")
            logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')
            data_params = {"x_min": x_min, "x_max": x_max}
            attack_params = {
                "norm": args.tr_norm,
                "eps": args.tr_epsilon,
                "alpha": args.tr_alpha,
                "step_size": args.tr_step_size,
                "num_steps": args.tr_num_iterations,
                "random_start": args.tr_rand,
                "num_restarts": args.tr_num_restarts,
                }

            for epoch in range(1, args.epochs + 1):
                start_time = time.time()
                pgd_train_args = dict(model=model,
                                      train_loader=train_loader,
                                      optimizer=optimizer,
                                      scheduler=scheduler,
                                      data_params=data_params,
                                      attack_params=attack_params)
                # breakpoint()
                train_loss, train_acc = train_adversarial(**pgd_train_args)

                test_loss, test_acc = test(model, test_loader)
                end_time = time.time()
                lr = scheduler.get_lr()[0]
                logger.info(f'{epoch} \t {end_time - start_time:.0f} \t \t {lr:.4f} \t {train_loss:.4f} \t {train_acc:.4f}')
                logger.info(f'Test  \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}')
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
                    + "_pgd_adv"
                    + "_"
                    + str(attack_params["eps"])
                    # + args.tr_norm
                    # + "_"
                    # + str(args.tr_epsilon)
                    + ".pt",
                    )
        else:
            raise NotImplementedError

    else:
        if args.tr_attack == "None":
            model.load_state_dict(
                torch.load(path.join(args.directory + 'checkpoints/', args.model + ".pt",))
                )
        elif args.tr_attack == "dn" or args.tr_attack == "dba" or args.tr_attack == "dnwi":
            checkpoint_name = args.directory + "checkpoints/" + \
                args.model + "_deep_adv" + "_" + str(lamb) + "_" + str(mu) + ".pt"
            model.load_state_dict(torch.load(checkpoint_name))
        elif args.tr_attack == "fgsm":
            checkpoint_name = args.directory + "checkpoints/" + \
                args.model + "_fgsm_adv" + "_" + \
                str(args.tr_alpha) + "_" + str(args.tr_epsilon) + ".pt"
            model.load_state_dict(torch.load(checkpoint_name))
        elif args.tr_attack == "pgd":
            checkpoint_name = args.directory + "checkpoints/" + \
                args.model + "_pgd_adv" + "_" + str(args.tr_epsilon) + ".pt"
            model.load_state_dict(torch.load(checkpoint_name))
        else:
            raise NotImplementedError

        print("Clean test accuracy")
        test_loss, test_acc = test(model, test_loader)
        logger.info(f'Test  \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}')

    # Attack network if args.attack_network is set to True (You can set that true by calling '-at' flag, default is False)
    # breakpoint()
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
        for key in attack_params:
            logger.info(key + ': ' + str(attack_params[key]))

        test_loss, test_acc = test_adversarial(
            model,
            test_loader,
            data_params=data_params,
            attack_params=attack_params,
            )
        logger.info(f'{args.attack} attacked \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}\n')

        attack_params = {
            "norm": "inf",
            "eps": args.epsilon}

        for key in attack_params:
            logger.info(key + ': ' + str(attack_params[key]))

        test_loss, test_acc = test_fgsm(
            model,
            test_loader,
            data_params=data_params,
            attack_params=attack_params,
            )
        logger.info(f'FGSM attacked \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}')


if __name__ == "__main__":
    main()
