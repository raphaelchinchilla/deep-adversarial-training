"""
Authors: Metehan Cekic
Date: 2020-03-09

Description: Training and testing functions for neural models

functions: 
    train: Performs a single training epoch
    train_adversarial: Performs a single training epoch with adversarial images
    test: Evaluates model by computing accuracy with test set
    test: Evaluates model by computing accuracy with adversarial test set
"""

import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from adv_ml.adversary.norm_ball_attacks import ProjectedGradientDescent as PGD


def train(args, model, device, train_loader, optimizer, epoch):
    start_time = time.time()
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):

        # Feed data to device ( e.g, GPU )
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        cross_ent = nn.CrossEntropyLoss()
        loss = cross_ent(output, target)
        loss.backward()
        optimizer.step()

        if (batch_idx+1) % args.log_interval == 0 and batch_idx != 0:
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            print(
                f"Train Epoch: {epoch+1} {100. * (batch_idx+1) / len(train_loader):.2f}%\tLoss: {loss.item():.2f}\tAcc: {100.*correct/output.shape[0]:.2f}%\t Time: {time.time() - start_time:.2f} sec"
            )
            start_time = time.time()


def train_adversarial(
    args, model, device, train_loader, optimizer, epoch, data_params, attack_params
):
    start_time = time.time()
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):

        # Feed data to device ( e.g, GPU )
        data, target = data.to(device), target.to(device)

        # Adversary
        perturbs = PGD(
            model,
            data,
            target,
            device,
            verbose=False,
            data_params=data_params,
            attack_params=attack_params,
        )
        data_adv = data + perturbs
        data_adv = torch.clamp(data_adv, 0, 1)

        optimizer.zero_grad()
        output = model(data_adv)
        cross_ent = nn.CrossEntropyLoss()
        loss = cross_ent(output, target)
        loss.backward()
        optimizer.step()

        if (batch_idx+1) % args.log_interval == 0 and batch_idx != 0:

            pred_adv = output.argmax(dim=1, keepdim=True)
            correct_adv = pred_adv.eq(target.view_as(pred_adv)).sum().item()

            output = model(data)
            pred_clean = output.argmax(dim=1, keepdim=True)
            correct_clean = pred_clean.eq(target.view_as(pred_clean)).sum().item()

            print(
                f"Train Epoch: {epoch+1} {100. * (batch_idx+1) / len(train_loader):.2f}%\tLoss: {loss.item():.2f}\tClean Acc: {100.*correct_clean/output.shape[0]:.2f}%\tAdv Acc: {100.*correct_adv/output.shape[0]:.2f}%\t Time: {time.time() - start_time:.2f} sec"
            )
            start_time = time.time()


def test(args, model, device, test_loader):

    model.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    # Divide summed-up loss by the number of datapoints in dataset
    test_loss /= len(test_loader.dataset)

    # Print out Loss and Accuracy for test set
    print(
        f"\nTest set: Average loss: {test_loss:.2f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.2f}%)\n"
    )


def test_adversarial(args, model, device, test_loader, data_params, attack_params):

    for key in attack_params:
        print(key + ': ' + str(attack_params[key]))
    # Set phase to testing
    model.eval()

    test_loss = 0
    correct = 0
    for data, target in tqdm(test_loader):

        data, target = data.to(device), target.to(device)

        # Attacks
        perturbs = PGD(
            model,
            data,
            target,
            device,
            data_params=data_params,
            attack_params=attack_params,
        )
        data += perturbs
        data = torch.clamp(data, 0, 1)

        output = model(data)
        test_loss += F.cross_entropy(output, target, reduction="sum").item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    # Divide summed-up loss by the number of datapoints in dataset
    test_loss /= len(test_loader.dataset)

    # Print out Loss and Accuracy for test set
    print(
        f"\nAdversarial test set (l_{attack_params['norm']}): Average loss: {test_loss:.2f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.2f}%)\n"
    )
