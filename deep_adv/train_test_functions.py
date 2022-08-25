"""
Authors: Metehan Cekic and Raphael Chinchilla
Date: 2020-03-09

Description: Training and testing functions for neural models

functions:
    train: Performs a single training epoch
    train_adversarial: Performs a single training epoch with adversarial images
    test: Evaluates model by computing accuracy with test set
    test: Evaluates model by computing accuracy with adversarial test set
"""

from tqdm import tqdm
from apex import amp

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepillusion.torchattacks import PGD, RFGSM, FGSM

from deep_adv.adversary.layer_attacks import DistortNeuronsConjugateGradient

def train(model, train_loader, optimizer, scheduler):
    """ Train given model with train_loader and optimizer """

    model.train()
    device = model.parameters().__next__().device

    train_loss = 0
    train_correct = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        cross_ent = nn.CrossEntropyLoss()
        loss = cross_ent(output, target)
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1, keepdim=True)
        train_correct += pred.eq(target.view_as(pred)).sum().item()

    train_size = len(train_loader.dataset)

    return train_loss/train_size, train_correct/train_size


def train_deep_adversarial(model, train_loader, optimizer, scheduler, lamb, mu,
                           data_params, attack_params):

    model.train()

    device = model.parameters().__next__().device

    dist_loss = 0
    dist_correct = 0
    clean_loss = 0
    clean_correct = 0
    for data, target in train_loader:

        # Feed data to device ( e.g, GPU )
        data, target = data.to(
            device), target.to(device)

        model.d = [0, 0, 0, 0]
        out_clean = model(data)

        dn_args = dict(model=model,
                       x=data,
                       y_true=target,
                       lamb=lamb,
                       mu=mu)

        DistortNeuronsConjugateGradient(**dn_args)


        optimizer.zero_grad()
        output = model(data)
        cross_ent = nn.CrossEntropyLoss()
        loss = cross_ent(output, target)
        loss_clean = cross_ent(out_clean, target)
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        scheduler.step()

        clean_loss += loss_clean.item() * data.size(0)
        pred_clean = out_clean.argmax(dim=1, keepdim=True)
        clean_correct += pred_clean.eq(target.view_as(pred_clean)).sum().item()

        dist_loss += loss.item() * data.size(0)
        pred_adv = output.argmax(dim=1, keepdim=True)
        dist_correct += pred_adv.eq(target.view_as(pred_adv)).sum().item()

    train_size = len(train_loader.dataset)

    return clean_loss/train_size, clean_correct/train_size, dist_loss/train_size, dist_correct/train_size


def train_fgsm_adversarial(model, train_loader, optimizer, scheduler, data_params, attack_params):
    """ Train given model with train_loader and optimizer with RFGSM adversarial examples """

    model.train()

    device = model.parameters().__next__().device

    train_loss = 0
    train_correct = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        fgsm_args = dict(net=model,
                         x=data,
                         y_true=target,
                         data_params=data_params,
                         attack_params=attack_params)
        perturbs = RFGSM(**fgsm_args)

        data_adv = data + perturbs

        optimizer.zero_grad()
        output = model(data_adv)
        cross_ent = nn.CrossEntropyLoss()
        loss = cross_ent(output, target)
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.item() * data.size(0)
        pred_adv = output.argmax(dim=1, keepdim=True)
        train_correct += pred_adv.eq(target.view_as(pred_adv)).sum().item()

    train_size = len(train_loader.dataset)

    return train_loss/train_size, train_correct/train_size


def train_adversarial(model, train_loader, optimizer, scheduler, data_params, attack_params):

    model.train()

    device = model.parameters().__next__().device

    train_loss = 0
    train_correct = 0
    for data, target in train_loader:

        data, target = data.to(device), target.to(device)

        # Adversary
        pgd_args = dict(net=model,
                        x=data,
                        y_true=target,
                        verbose=False,
                        data_params=data_params,
                        attack_params=attack_params)
        perturbs = PGD(**pgd_args)

        data_adv = data + perturbs

        optimizer.zero_grad()
        output = model(data_adv)
        cross_ent = nn.CrossEntropyLoss()
        loss = cross_ent(output, target)
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.item() * data.size(0)
        pred_adv = output.argmax(dim=1, keepdim=True)
        train_correct += pred_adv.eq(target.view_as(pred_adv)).sum().item()

    train_size = len(train_loader.dataset)

    return train_loss/train_size, train_correct/train_size


def test(model, test_loader):

    model.eval()

    device = model.parameters().__next__().device

    test_loss = 0
    test_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            cross_ent = nn.CrossEntropyLoss()
            test_loss += cross_ent(output, target).item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            test_correct += pred.eq(target.view_as(pred)).sum().item()
    test_size = len(test_loader.dataset)

    return test_loss/test_size, test_correct/test_size


def test_adversarial(model, test_loader, data_params, attack_params):

    device = model.parameters().__next__().device

    model.eval()

    test_loss = 0
    test_correct = 0
    for data, target in tqdm(test_loader):

        data, target = data.to(device), target.to(device)

        # Attacks
        pgd_args = dict(net=model,
                        x=data,
                        y_true=target,
                        data_params=data_params,
                        attack_params=attack_params)
        perturbs = PGD(**pgd_args)
        data += perturbs
        # breakpoint()

        output = model(data)

        cross_ent = nn.CrossEntropyLoss()
        test_loss += cross_ent(output, target).item() * data.size(0)

        pred = output.argmax(dim=1, keepdim=True)
        test_correct += pred.eq(target.view_as(pred)).sum().item()

    test_size = len(test_loader.dataset)

    return test_loss/test_size, test_correct/test_size


def test_fgsm(model, test_loader, data_params, attack_params):

    device = model.parameters().__next__().device

    model.eval()

    test_loss = 0
    test_correct = 0
    for data, target in tqdm(test_loader):

        data, target = data.to(device), target.to(device)

        # Attacks
        fgsm_args = dict(net=model,
                         x=data,
                         y_true=target,
                         data_params=data_params,
                         attack_params=attack_params)
        perturbs = FGSM(**fgsm_args)
        data += perturbs
        # breakpoint()

        output = model(data)

        cross_ent = nn.CrossEntropyLoss()
        test_loss += cross_ent(output, target).item() * data.size(0)

        pred = output.argmax(dim=1, keepdim=True)
        test_correct += pred.eq(target.view_as(pred)).sum().item()

    test_size = len(test_loader.dataset)

    return test_loss/test_size, test_correct/test_size
