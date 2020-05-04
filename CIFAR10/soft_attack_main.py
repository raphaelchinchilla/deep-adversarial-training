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

from deepillusion.torchattacks import iterative_soft_attack


def main():
    """ main function to run the experiments """

    args = get_arguments()

    if not os.path.exists(args.directory + 'logs'):
        os.mkdir(args.directory + 'logs')

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

    model.load_state_dict(torch.load(path.join(args.directory + "checkpoints/", args.model + "_"
                                               + args.tr_attack + "_" + args.tr_norm + "_"
                                               + str(np.int(np.round(args.tr_epsilon * 255)))
                                               + "_"
                                               + str(np.int(np.round(args.tr_alpha * 255)))
                                               + ".pt")))

    data, target = next(iter(test_loader))
    data, target = data.to(device), target.to(device)

    data_params = {"x_min": x_min, "x_max": x_max}
    attack_params = {
        "norm": "inf",
        "eps": args.epsilon,
        "step_size": args.step_size,
        "num_steps": args.num_iterations,
        "random_start": args.rand,
        "num_restarts": args.num_restarts,
        }

    # y_soft_vector = torch.rand(args.test_batch_size, 10)
    y_soft_vector = torch.ones(args.test_batch_size, 10)

    y_soft_vector /= torch.sum(y_soft_vector, dim=1).view(-1, 1)
    y_soft_vector = y_soft_vector.to(device)
    perturbs = iterative_soft_attack(model, data, y_soft_vector,
                                     data_params=data_params, attack_params=attack_params)

    out = torch.nn.functional.softmax(model(data))
    out_adv = torch.nn.functional.softmax(model(data+perturbs))

    breakpoint()

    img = data[0] + perturbs[0]
    img = img.cpu().numpy()
    img = img.transpose(1, 2, 0)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.imshow(img)
    plt.savefig("Mt")


if __name__ == "__main__":
    main()
