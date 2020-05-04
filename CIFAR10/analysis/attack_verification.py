
"""
Analysis
PyTorch

Example Run

python -m deep_adv.CIFAR10.analysis.attack_verification

"""


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from os import path
import logging
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets, transforms

from deep_adv.utils import plot_settings
# from deep_adv.adversary.norm_ball_attacks import ProjectedGradientDescent as PGD
from deepillusion.torchattacks import PGD
from deepillusion.torchattacks.analysis import get_perturbation_stats


mpl.rc("text", usetex=True)
mpl.rc("text.latex", preamble=r"\usepackage{amsmath}, \usepackage{sfmath}")
logger = logging.getLogger(__name__)


def layer_statistics(args, model, loader, data_params, attack_params):

    device = model.parameters().__next__().device

    for data, target in tqdm(loader):
        data, target = data.to(device), target.to(device)
        perturbs = PGD(model, data, target,
                       data_params=data_params, attack_params=attack_params)
        layer_clean = F.relu(model.bn1(model.conv1(data)))
        layer_adv = F.relu(model.bn1(model.conv1(data+perturbs)))
        output = model(data + perturbs)
        pred = output.argmax(dim=1, keepdim=True)
        test_correct = pred.eq(target.view_as(pred)).sum().item()
        print("attack acc")
        print(test_correct/data.size(0))
        difference = layer_clean - layer_adv
        layer_clean = layer_clean.view(data.size(0), -1)
        difference = difference.view(data.size(0), -1)
        l1_norm = torch.norm(difference, p=1, dim=1)/torch.norm(layer_clean, p=1, dim=1)
        l2_norm = torch.norm(difference, p=2, dim=1)/torch.norm(layer_clean, p=2, dim=1)
        breakpoint()


def attack_statistics(args, model, loader, data_params, attack_params):

    device = model.parameters().__next__().device

    from deep_adv.train_test_functions import test

    budget = 0.031372
    test_loss = 0
    test_correct = 0
    cross_ent = nn.CrossEntropyLoss()
    for data, target in tqdm(loader):
        data, target = data.to(device), target.to(device)
        perturbs = PGD(model, data, target,
                       data_params=data_params, attack_params=attack_params)
        e = get_perturbation_stats(data, data + perturbs, budget)

        output = model(data + perturbs)
        test_loss += cross_ent(output, target).item() * target.size(0)
        pred = output.argmax(dim=1, keepdim=True)
        test_correct += pred.eq(target.view_as(pred)).sum().item()

    test_size = len(loader.dataset)

    print(f'Attack  \t loss: {test_loss/test_size:.4f} \t acc: {test_correct/test_size:.4f}')

    # plot_perturbations(
    #     args, clean_images, labels, attacked_images, preds, budget, title=None
    #     )


def plot_perturbations(args, x, y, x_adv, y_hat, budget, title=None):

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
        )

    fig = plt.figure(figsize=(10, 10))

    plt.suptitle(r"Adversarial Attack with PGD($l_{\infty}$)", y=0.995)

    for i in range(5):

        plt.subplot(5, 3, i * 3 + 1)
        plt.imshow(x[i], vmin=0, vmax=1)
        plt.title(f"Original {classes[y[i]]}")
        plt.axis("off")

        plt.subplot(5, 3, i * 3 + 2)
        plt.imshow(x_adv[i], vmin=0, vmax=1)
        plt.title(f"Perturbed to {classes[y_hat[i]]}")
        plt.axis("off")

        plt.subplot(5, 3, i * 3 + 3)
        plt.imshow(x_adv[i] - x[i], cmap=plot_settings.cm, vmin=-budget, vmax=+budget)
        plt.title(r"Perturbation $l_\infty$ = {:.0f}".format(np.round(255 * budget)))
        plt.colorbar()
        plt.axis("off")

    plt.tight_layout()

    if not os.path.exists(args.directory + "figures/"):
        os.makedirs(args.directory + "figures/")
    plt.savefig(args.directory + "figures/" + "l_inf_attack")


def get_started():

    from deep_adv.CIFAR10.models.resnet import ResNet34
    from deep_adv.CIFAR10.models.resnet_new import ResNet, ResNetWide
    from deep_adv.CIFAR10.models.preact_resnet import PreActResNet18
    from deep_adv.CIFAR10.parameters import get_arguments
    from deep_adv.CIFAR10.read_datasets import cifar10, cifar10_black_box

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

    model.load_state_dict(torch.load(path.join(args.directory + "checkpoints/", args.model + "_"
                                               + args.tr_attack + "_" + args.tr_norm + "_"
                                               + str(np.int(np.round(args.tr_epsilon * 255)))
                                               + "_"
                                               + str(np.int(np.round(args.tr_alpha * 255)))
                                               + ".pt")))
    # model.load_state_dict(
    #     torch.load(path.join(args.directory + 'checkpoints/', args.model + ".pt",))
    #     )

    if device == "cuda":
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    x_min = 0.0
    x_max = 1.0
    data_params = {"x_min": x_min, "x_max": x_max}
    attack_params = {
        "norm": "inf",
        "eps": args.epsilon,
        "step_size": args.step_size,
        "num_steps": args.num_iterations,
        "random_start": args.rand,
        "num_restarts": args.num_restarts,
        }

    return args, model, train_loader, test_loader, data_params, attack_params, device


def main():
    args, model, train_loader, test_loader, data_params, attack_params, device = get_started()
    # attack_statistics(args, model, test_loader, data_params, attack_params)
    layer_statistics(args, model, test_loader, data_params, attack_params)


if __name__ == "__main__":
    main()
