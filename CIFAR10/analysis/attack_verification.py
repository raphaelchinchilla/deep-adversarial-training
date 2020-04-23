"""
Analysis
PyTorch

Example Run

python -m cvx_demixing.CIFAR10.analysis.attack_verification

"""


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

from torchvision import datasets, transforms

from deep_adv.utils import plot_settings
from deep_adv.CIFAR10.parameters import get_arguments


mpl.rc("text", usetex=True)
mpl.rc("text.latex", preamble=r"\usepackage{amsmath}, \usepackage{sfmath}")


def plot_attacked_images(args):

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )

    clean_images = np.array(
        datasets.CIFAR10(
            root="/home/canbakiskan/cvx_demixing/CIFAR10/data/original_dataset",
            train=False,
            download=False,
            transform=transform_test,
            ).data
        )
    clean_images = clean_images / np.max(clean_images)

    attacked_dict = np.load(
        "/home/canbakiskan/cvx_demixing/CIFAR10/data/attacked_dataset/ResNet.npz"
        )

    labels = attacked_dict["labels"].reshape(-1)
    preds = attacked_dict["preds"].reshape(-1)

    attacked_images = attacked_dict["images"].reshape(-1, 3, 32, 32)
    attacked_images = np.moveaxis(attacked_images, [1, 2, 3], [-1, 1, 2])

    budget = 0.031372
    e = perturbation_properties(attacked_images, budget, clean_images)

    plot_perturbations(
        args, clean_images, labels, attacked_images, preds, budget, title=None
        )


def perturbation_properties(adversarial_images, epsilon, test_images):

    e = adversarial_images.reshape(
        -1,
        adversarial_images.shape[1]
        * adversarial_images.shape[2]
        * adversarial_images.shape[3],
        ) - test_images.reshape(
        [
            -1,
            adversarial_images.shape[1]
            * adversarial_images.shape[2]
            * adversarial_images.shape[3],
            ]
        )

    print(f"Attack budget: {epsilon}")

    print(
        f"Percent of images perturbation is added: {np.count_nonzero(np.max(np.abs(e),axis = 1))/100.} %"
        )
    print(f"L_inf distance: {np.abs(e).max():.3f}")
    print(f"Avg magnitude: {np.abs(e).mean():.3f}")

    tol = 1e-5

    num_eps = (
        ((np.abs(e) < epsilon + tol) & (np.abs(e) > epsilon - tol)).sum(axis=1).mean()
        )

    print(
        f"Percent of pixels with mag=eps: {100*num_eps/(adversarial_images.shape[1]*adversarial_images.shape[2]*adversarial_images.shape[3])}"
        )

    return e


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


def main():

    args = get_arguments()
    plot_attacked_images(args)


if __name__ == "__main__":
    main()
