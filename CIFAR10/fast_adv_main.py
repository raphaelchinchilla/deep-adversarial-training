"""
python -m adv_ml.CIFAR10.fast_adv_main
"""

import torch
from tqdm import tqdm
import os
from os import path

from deepillusion.torchattacks import PGD
from adv_ml.train_test_functions import (
    train,
    train_adversarial,
    test,
    test_adversarial,
)
from adv_ml.CIFAR10.models.preact_resnet import PreActResNet18


from adv_ml.CIFAR10.parameters import get_arguments
from adv_ml.CIFAR10.read_datasets import cifar10, cifar10_black_box

args = get_arguments()

torch.manual_seed(args.seed)

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

train_loader, test_loader = cifar10(args)
x_min = 0.0
x_max = 1.0


# norm = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
model = PreActResNet18().to(device)

model.load_state_dict(
    torch.load(path.join(args.directory + 'checkpoints/', "model.pth",))
    )

print("Clean test accuracy")
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
