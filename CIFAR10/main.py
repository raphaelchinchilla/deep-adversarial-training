"""

Example Run

python -m adv_ml.CIFAR10.main --model ResNetMadry -tr -tra fgsm -at -Ni 7

"""
import time
import os
from os import path
from tqdm import tqdm
import numpy as np

import logging

from apex import amp
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

from adv_ml.CIFAR10.models.resnet import ResNet34
from adv_ml.CIFAR10.models.resnet_new import ResNet, ResNetWide
from adv_ml.CIFAR10.models.preact_resnet import PreActResNet18
from adv_ml.train_test_functions import train, train_fgsm_adversarial, train_adversarial, test, test_adversarial

from adv_ml.CIFAR10.parameters import get_arguments
from adv_ml.CIFAR10.read_datasets import cifar10, cifar10_black_box


logger = logging.getLogger(__name__)


def main():
    """ main function to run the experiments """

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

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, test_loader = cifar10(args)
    x_min = 0.0
    x_max = 1.0

    # Decide on which model to use
    if args.model == "ResNet":
        model = ResNet34().to(device)
    elif args.model == "ResNetMadry":
        model = ResNet().to(device)
    elif args.model == "ResNetMadryWide":
        model = ResNetWide().to(device)
    elif args.model == "ResNet18":
        model = PreActResNet18().to(device)
    else:
        raise NotImplementedError

    if device == "cuda":
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    logger.info(model)
    logger.info("\n")

    # Which optimizer to be used for training
    optimizer = optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum,
                          weight_decay=args.weight_decay)

    amp_args = dict(opt_level=args.opt_level, loss_scale=args.loss_scale, verbosity=False)
    if args.opt_level == 'O2':
        amp_args['master_weights'] = args.master_weights
    model, optimizer = amp.initialize(model, optimizer, **amp_args)

    lr_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.lr_min,
                                                  max_lr=args.lr_max, step_size_up=lr_steps/2,
                                                  step_size_down=lr_steps/2)
    # scheduler = MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

    # Train network if args.train is set to True (You can set that true by calling '-tr' flag, default is False)
    logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')

    if args.train:
        if args.tr_attack == "None":
            logger.info("Standard training")
            for epoch in tqdm(range(1, args.epochs + 1)):
                start_time = time.time()
                train_loss, train_acc = train(model, device, train_loader, optimizer, scheduler)
                test_loss, test_acc = test(model, device, test_loader)
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

        elif args.tr_attack == "fgsm":
            logger.info("FGSM adversarial training")
            for epoch in tqdm(range(1, args.epochs + 1)):
                start_time = time.time()
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
                train_loss, train_acc = train_fgsm_adversarial(model, device, train_loader,
                                                               optimizer, scheduler, data_params,
                                                               attack_params)
                test_loss, test_acc = test(model, device, test_loader)
                end_time = time.time()
                lr = scheduler.get_lr()[0]
                logger.info(f'{epoch} \t {end_time - start_time:.0f} \t \t {lr:.4f} \t {train_loss:.4f} \t {train_acc:.4f}')
                logger.info(f'Test  \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}')

            # Save model parameters
            if args.save_model:
                if not os.path.exists(args.directory + "checkpoints/"):
                    os.makedirs(args.directory + "checkpoints/")
                torch.save(model.state_dict(),
                           path.join(args.directory + "checkpoints/", args.model + '_'
                                     + args.tr_attack + '_' + args.tr_norm + "_"
                                     + str(np.int(np.round(args.tr_epsilon * 255)))
                                     + '_' + str(np.int(np.round(args.tr_alpha * 255)))
                                     + ".pt"))
    # Train network if args.train_adversarial is set to True (You can set that true by calling '-tra' flag, default is False)
        elif args.tr_attack == "pgd":
            logger.info("PGD adversarial training")
            data_params = {"x_min": x_min, "x_max": x_max}
            attack_params = {
                "norm": args.tr_norm,
                "eps": args.tr_epsilon,
                "step_size": args.tr_step_size,
                "num_steps": args.tr_num_iterations,
                "random_start": args.tr_rand,
                "num_restarts": args.tr_num_restarts,
                }

            for epoch in tqdm(range(1, args.epochs + 1)):
                start_time = time.time()
                train_loss, train_acc = train_adversarial(model, device, train_loader, optimizer,
                                                          scheduler, data_params, attack_params)
                test_loss, test_acc = test(model, device, test_loader)
                end_time = time.time()
                lr = scheduler.get_lr()[0]
                logger.info(f'{epoch} \t {end_time - start_time:.0f} \t \t {lr:.4f} \t {train_loss:.4f} \t {train_acc:.4f}')
                logger.info(f'Test  \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}')

            # Save model parameters
            if args.save_model:
                if not os.path.exists(args.directory + "checkpoints/"):
                    os.makedirs(args.directory + "checkpoints/")
                torch.save(
                    model.state_dict(),
                    path.join(
                        args.directory + "checkpoints/",
                        args.model + '_'
                        + args.tr_attack + '_'
                        + args.tr_norm + "_"
                        + str(np.int(np.round(args.tr_epsilon * 255)))
                        + ".pt",
                        ),
                    )
        else:
            raise NotImplementedError

    else:
        if args.tr_attack == "None":
            model.load_state_dict(
                torch.load(path.join(args.directory + 'checkpoints/', args.model + ".pt",))
                )
        elif args.tr_attack == "fgsm":
            model.load_state_dict(torch.load(path.join(args.directory + "checkpoints/", args.model + "_"
                                                       + args.tr_attack + "_" + args.tr_norm + "_"
                                                       + str(np.int(np.round(args.tr_epsilon * 255)))
                                                       + "_"
                                                       + str(np.int(np.round(args.tr_alpha * 255)))
                                                       + ".pt")))
        elif args.tr_attack == "pgd":
            model.load_state_dict(
                torch.load(
                    path.join(
                        args.directory + "checkpoints/",
                        args.model + "_"
                        + args.tr_attack + "_"
                        + args.tr_norm + "_"
                        + str(np.int(np.round(args.tr_epsilon * 255)))
                        + ".pt",
                        )
                    )
                )
        elif args.tr_attack == "old":
            model.load_state_dict(
                torch.load(path.join(args.directory + 'checkpoints/', args.model + "_old.pt",))
                )
            # breakpoint()
        else:
            raise NotImplementedError

        logger.info("Clean test accuracy")
        test_loss, test_acc = test(model, device, test_loader)
        logger.info(f'Test  \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}')

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
        attack_loss, attack_acc = test_adversarial(model, device, test_loader,
                                                   data_params=data_params,
                                                   attack_params=attack_params)
        logger.info(f'Attack  \t loss: {attack_loss:.4f} \t acc: {attack_acc:.4f}')

    # if args.black_box:
    #     attack_loader = cifar10_black_box(args)

    #     test(model, device, attack_loader)


if __name__ == "__main__":
    main()
