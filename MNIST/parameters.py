'''
Author: Metehan Cekic
Hyper-parameters
'''

import argparse

def get_arguments():

    parser = argparse.ArgumentParser(description='PyTorch MNIST')
    
    # Directory
    parser.add_argument('--directory', type=str, default='/home/metehan/deep_adv/MNIST/', metavar='', help='Directory for checkpoint and stuff')

    # Dataset
    parser.add_argument('--dataset', type=str, default='mnist', choices=["mnist","fashion"], metavar='mnist/fashion', help='Which dataset to use (default: mnist)')

    # Neural Model
    parser.add_argument('--model', type=str, default='CNN', metavar='FcNN/CNN', help='Which model to use (default: CNN)')


    # Optimizer
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--weight_decay', type=float, default=0.0005, metavar='WD', help='Weight decay (default: 0.0005)')

    # Batch Sizes & #Epochs
    parser.add_argument('--batch_size', type=int, default=50, metavar='N', help='input batch size for training (default: 50)')
    parser.add_argument('--test_batch_size', type=int, default=10000, metavar='N', help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 10)')

    # Adversarial training parameters
    parser.add_argument('--tr_norm', type=str, default='inf', metavar='inf/p', help='Which attack norm to use for training')
    parser.add_argument('-tr_eps', '--tr_epsilon', type=float, default=0.3, metavar='', help='attack budget for training')
    parser.add_argument('-tr_Ss', '--tr_step_size', type=float, default=0.01, metavar='', help='Step size for PGD, adv training')
    parser.add_argument('-tr_Ni', '--tr_num_iterations', type=int, default=40, metavar='', help='Number of iterations for PGD, adv training')
    parser.add_argument('--tr-rand', action='store_false', default=True, help='randomly initialize PGD attack for training')
    parser.add_argument('-tr_Nrest', '--tr_num_restarts', type=int, default=1, metavar='', help='number of restarts for pgd for training')

    # Adversarial testing parameters
    parser.add_argument('--norm', type=str, default='inf', metavar='inf/p', help='Which attack norm to use')
    parser.add_argument('-eps', '--epsilon', type=float, default=0.3, metavar='', help='attack budget')
    parser.add_argument('-Ss', '--step_size', type=float, default=0.01, metavar='', help='Step size for PGD')
    parser.add_argument('-Ni', '--num_iterations', type=int, default=100, metavar='', help='Number of iterations for PGD')
    parser.add_argument('--rand', action='store_true', default=False, help='randomly initialize PGD attack')
    parser.add_argument('-Nrest', '--num_restarts', type=int, default=1, metavar='', help='number of restarts for pgd')

    # Others
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=600, metavar='N', help='how many batches to wait before logging training status')

    # Actions
    parser.add_argument('-tr', '--train', action='store_true', help='Train network, default = False')
    parser.add_argument("-adv", "--adversarial", action="store_true", help="Train (or load) model adversarially, default = False",)
    parser.add_argument('-at', '--attack_network', action='store_true', help='Attack network, default = False')
    parser.add_argument("-bb", "--black_box", action="store_true", help="Attack network, default = False",)
    parser.add_argument('-sm', '--save-model', action='store_true', default=False, help='For Saving the current Model, default = False ')
    parser.add_argument('-im', '--initialize-model', action='store_true', default=False, help='For initializing the Model from checkpoint with standard parameters')

    args = parser.parse_args()

    return args