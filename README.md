# Deep adversarial training

Implementation of the deep adversarial training algorithm which is explained in the folder `theory`. The core idea is to do adversarial training by attacking all the layers, not just the first one. The layer attack is obtained by reformulating the maximization and writing it as an optimization over the value of the layers. By using the Conjugate Gradient algorithm, computing this maximization can be obtained very efficiently.

## Codes
The folder deep_adv contains the pytorch codes for deep adversarial training of MNIST. We were never able to achieve any significant protection against PGD attack, even if in theory a deep adversarial attack contains all possible traditional adversarial attacks. We believe that achieving the maximum with respect to the layer attack was ultimately not feasible, and even if Conjugate Gradient was very efficient, actually doing a deep layer attack would consume too much time without necessaraly robustifying the neural network.

The codes are a collaboration between Metehan Cekic, who wrote most of the general structure of the module, and Raphael Chinchilla, who implemented the algorithm and did most of the testing and debugging.

The adversarial attack is implemented in the file `adversary/layer_attacks.py`

Run the codes by typing: `python -m deep_adv.MNIST.main -at -tra dnwi -tr -sm`

Besides pytorch and some other standard packages, the code also requires the deepillusion package, developped by Metehan Cekic and avaiable on pip and NVIDIA apex, avaiable on https://github.com/NVIDIA/apex
