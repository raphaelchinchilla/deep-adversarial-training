"""
Authors: Metehan Cekic and Raphael Chinchilla
Date: 2020-03-09

Description: Attack models with l_{p} norm constraints

Attacks: FastGradientSignMethod(FGSM), ProjectedGradientDescent(PGD)

To run this file with adversary attack, be sure to have this folder in your path, deepillusion installed. To run the code, go to the command line and run:
python -m deep_adv.MNIST.main -at -tra dnwi -l 2 -m 0.5 -tr -sm --epochs 10 --weight_decay 0.001

"""

from tqdm import tqdm
import numpy as np
import math
from apex import amp
# from copy import deep_copy
import time

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim


#from deepillusion.torchattacks._utils import clip


def DistortNeuronsConjugateGradient(model, x, y_true, lamb, mu, optimizer=None):
    '''
    Descriptions:
        Conjugate Gradient
        No line search but diminishing step
        Projection of n[0] into correct set (abs(n[0]-x)<0.3, 0<=n[0]<=1)
    '''
    # Parameters
    num_iters = 200
    eps_input = 0.3 # value to clamp input disturbance
    eps_layers = 2 # value to clamp layer disturbance
    lamb_layers = 20 # how many times regularization in inside layers
    eps_init_layers= 0.5# math.sqrt(1/(lamb*lamb_layers)) # "std" for the initialization of disturbances
    tol_grad=1e-3 # value gradient problem considered solved
    #lamb_reg = 0.01 # regularization of the intermediate layers
    debug = True # set to true to activate break points and prints
    #Code
    model.eval()
    device = model.parameters().__next__().device
    criterion = nn.CrossEntropyLoss(reduction="none")

    for p in model.parameters():
        p.requires_grad = False

    layers = list(model.children())

    # Initializing the Nodes
    n = [None] * (len(model.n)-1)
    for i in range(len(n)):
        n[i] = torch.empty(model.n[i].size()).to(device)

    # Initializing the auxiliary variables
    aux = [None] * (len(model.n)-2)
    for i in range(len(aux)):
        aux[i] = torch.empty(model.n[i+1].size()).to(device)


    # Initializing several auxiliary lists as empty
    direct = [None] * (len(n))
    for i in range(len(n)):
        direct[i] = torch.zeros(model.n[i].size()).to(device)
    alpha=torch.ones((x.size(0),1),device=device)
    beta=torch.zeros((x.size(0),1),device=device)
    norm_grad=torch.zeros((x.size(0),1),device=device)
    loss=torch.zeros((x.size(0)),device=device)


    # Defining some function
    # crit=torch.nn.L1Loss(reduction='none')
    crit=torch.nn.MSELoss(reduction='none')
    def rho(z,w):
        return crit(z.view(x.size(0), -1),w.view(x.size(0), -1)).sum(1)
    def reg(z):
        return (z.view(x.size(0), -1)**2).sum(1)
    def batch_dot_prod(z,w):
        return (z.view(x.size(0), -1)*w.view(x.size(0), -1)).sum(1).view(x.size(0), -1)


    with torch.no_grad():
        x=layers[0](x)

    # Initializing the value of the nodes

    # with torch.no_grad():
    #     n[0]=x+eps_input*(2*torch.rand(x.size())-1).to(device)
    #     n[0]=torch.clamp(n[0],0.,1.)
    # n[0].requires_grad_(True)
    # aux[0]=F.max_pool2d(F.leaky_relu(layers[1](n[0])), (2, 2))
    # with torch.no_grad():
    #     n[1]=aux[0]+eps_init_layers*torch.randn(model.n[1].size()).to(device)
    # n[1].requires_grad_(True)
    # aux[1]=F.max_pool2d(F.leaky_relu(layers[2](n[1])), (2, 2)).view(x.size(0), -1)
    # with torch.no_grad():
    #     n[2]=aux[1]+eps_init_layers*torch.randn(model.n[2].size()).to(device)
    # n[2].requires_grad_(True)
    # aux[2]=F.leaky_relu(layers[3](n[2]))
    # with torch.no_grad():
    #     n[3]=aux[2]+eps_init_layers*torch.randn(model.n[3].size()).to(device)
    # n[3].requires_grad_(True)

    with torch.no_grad():
        n[0]=x+eps_input*(2*torch.rand(x.size())-1).to(device)
        n[0]=torch.clamp(n[0],0.,1.)
    n[0].requires_grad_(True)
    aux[0]=F.max_pool2d(F.leaky_relu(layers[1](n[0])), (2, 2))
    with torch.no_grad():
        n[1]=aux[0]+eps_init_layers*(2*torch.rand(model.n[1].size())-1).to(device)
    n[1].requires_grad_(True)
    aux[1]=F.max_pool2d(F.leaky_relu(layers[2](n[1])), (2, 2)).view(x.size(0), -1)
    with torch.no_grad():
        n[2]=aux[1]+eps_init_layers*(2*torch.rand(model.n[2].size())-1).to(device)
    n[2].requires_grad_(True)
    aux[2]=F.leaky_relu(layers[3](n[2]))
    with torch.no_grad():
        n[3]=aux[2]+eps_init_layers*(2*torch.rand(model.n[3].size())-1).to(device)
    n[3].requires_grad_(True)




    solver = 'running'
    iter=0
    while iter<= num_iters and solver=='running':
        iter+=1
        loss_prev=loss.clone()
        # Calculating the loss
        loss =  rho(n[0],x) #- lamb_reg*reg(n[0])
        loss += lamb_layers*rho(n[1],aux[0]) #- lamb_reg*reg(n[1])
        loss += lamb_layers*rho(n[2],aux[1]) #- lamb_reg*reg(n[2])
        loss += lamb_layers*rho(n[3],aux[2]) #- lamb_reg*reg(n[3])
        loss += -criterion(layers[-1](n[3]), y_true)/lamb

        loss.backward(gradient=torch.ones_like(y_true, dtype=torch.float))


        with torch.no_grad():
            norm_grad_prev=norm_grad.clone()
            norm_grad=torch.zeros((x.size(0),1),device=device)
            for i in range(len(n)):
                norm_grad+=batch_dot_prod(n[i].grad,n[i].grad)
            if iter == 1:
                norm_grad_init=norm_grad.clone()
                loss_init=loss.detach().clone()
            if iter>1:
                loss_diff=loss_prev-loss
                # Calculating beta
                beta=norm_grad/norm_grad_prev
                beta[loss_diff<0]=0 # restart direction if not search direction
                # Ajusting step size alpha depending on whether cost was increased or decreased
                # alpha[loss_diff<0]*=0.5 # decrease by half if cost has increased
                # alpha[(loss_init-loss)<0]*=0.2 # decrease by another five (total of ten) if cost is worst than original cost
                # alpha[loss_diff>0]*=1.05
            if debug:
                print("Iter:", iter, ", Loss max: ", loss.max().data.cpu().numpy(), ", Loss min:", loss.min().data.cpu().numpy(), ", Output Loss", criterion(layers[-1](n[3]), y_true).max().data.cpu().numpy())
                print("      Grad n[0]: ", n[0].grad.abs().mean().data.cpu().numpy()," Grad n[1]: ", n[1].grad.abs().mean().data.cpu().numpy()," Grad n[2]: ", n[2].grad.abs().mean().data.cpu().numpy()," Grad n[3]: ", n[3].grad.abs().mean().data.cpu().numpy())
                print("      Beta:", beta.abs().max().data.cpu().numpy(),"norm_grad:", norm_grad.max().data.cpu().numpy(), "alpha: ", alpha.min().data.cpu().numpy())
            for i in range(len(n)):
                direct[i].view(x.size(0),-1).mul_(beta) # Multiply the previous step direction of each batch by the value of beta that is equivalent. Equivalent to direct[i]*=beta.view([-1]+[1]*(direct[i].ndim-1)).expand_as(direct[i])
                direct[i]+=-n[i].grad # Update the search direction with the new step to go
                # n[i]+=mu*direct[i].view(x.size(0),-1).mul(alpha).view_as(n[i]) # Update the new value of the neurons with the step direction alpha.
                n[i]+=mu/iter*direct[i]
                n[i].grad.zero_()
            if iter>=10:
                norm_ratio=norm_grad/norm_grad_init
                if norm_ratio.max()<tol_grad:
                    solver='solved'
                elif iter>=20 and (norm_ratio.min()>=1 or (loss_init-loss).max()<0):
                    solver='failed'
                elif iter>=50 and norm_ratio.squeeze().median()<tol_grad:
                    solver='partially solved'
                # elif norm_grad_prev.max()<norm_grad.max():
                    # solver='failed'




        # Simulating the system
        with torch.no_grad():
            n[0]=x+torch.clamp(n[0] - x,-eps_input,eps_input)
            n[0]=torch.clamp(n[0],0.,1.)
        n[0].requires_grad_(True)
        aux[0]=F.max_pool2d(F.leaky_relu(layers[1](n[0])), (2, 2))
        with torch.no_grad():
            n[1]=aux[0]+torch.clamp(n[1] - aux[0],-eps_layers,eps_layers)
        n[1].requires_grad_(True)
        aux[1]=F.max_pool2d(F.leaky_relu(layers[2](n[1])), (2, 2)).view(x.size(0), -1)
        with torch.no_grad():
            n[2]=aux[1]+torch.clamp(n[2] - aux[1],-eps_layers,eps_layers)
        n[2].requires_grad_(True)
        aux[2]=F.leaky_relu(layers[3](n[2]))
        with torch.no_grad():
            n[3]=aux[2]+torch.clamp(n[3] - aux[2],-eps_layers,eps_layers)
        n[3].requires_grad_(True)

    if torch.any(torch.isnan(loss)) or torch.any(torch.isnan(n[0])) or torch.any(torch.isnan(n[1])) or torch.any(torch.isnan(n[2])):
        raise ValueError('Diverged')
    # if torch.any(loss.abs()>1e5) or torch.any(n[0].abs()>1e3) or torch.any(n[1].abs()>1e3) or torch.any(n[2].abs()>1e3) or torch.any(n[3].abs()>1e3):
        # breakpoint()

    with torch.no_grad():
        d = [None] * (len(n))
        for i in range(len(n)):
            d[i] = torch.zeros_like(model.n[i])

        # Input disturbance is always considered as solved. Worst case it is just noise
        d[0] = (n[0] - x)
        norm_ratio.squeeze_()
        # For those where it was solved, input the solved solution
        d[1][norm_ratio<tol_grad] = (n[1] - aux[0])[norm_ratio<tol_grad]
        d[2][norm_ratio<tol_grad] = (n[2] - aux[1])[norm_ratio<tol_grad]
        d[3][norm_ratio<tol_grad] = (n[3] - aux[2])[norm_ratio<tol_grad]

        # For those where it was not, input random small disturbance
        d[1][norm_ratio>=tol_grad] = 0.05*torch.randn_like(d[1][norm_ratio>=tol_grad])
        d[2][norm_ratio>=tol_grad] = 0.05*torch.randn_like(d[2][norm_ratio>=tol_grad])
        d[3][norm_ratio>=tol_grad] = 0.05*torch.randn_like(d[3][norm_ratio>=tol_grad])


        if debug:
            print(" d[0]: ", d[0].abs().max().data.cpu().numpy() ," d[1]: ", d[1].abs().max().data.cpu().numpy()," d[2]: ", d[2].abs().max().data.cpu().numpy()," d[3]: ", d[3].abs().max().data.cpu().numpy())
            # breakpoint()
            # time.sleep(0.000001)

        for i in range(len(n)):
            model.d[i] = d[i].clone()


    for p in model.parameters():
        p.requires_grad = True

def DistortNeuronsConjugateGradientLineSearch(model, x, y_true, lamb, mu, optimizer=None):
    '''
    Descriptions:
        Conjugate Gradient
        Line search using Armijo Rule
        Projection of n[0] into correct set (abs(n[0]-x)<0.3, 0<=n[0]<=1)
    '''
    # Parameters
    num_iters = 200
    eps_input = 0.3 # value to clamp input disturbance
    eps_layers = 2 # value to clamp layer disturbance
    lamb_layers = 20 # how many times regularization in inside layers
    eps_init_layers= 2# math.sqrt(1/(lamb*lamb_layers)) # "std" for the initialization of disturbances
    tol_grad=1e-3 # value gradient problem considered solved
    #lamb_reg = 0.01 # regularization of the intermediate layers
    debug = False# set to true to activate break points and prints
    #Code
    model.eval()
    device = model.parameters().__next__().device
    criterion = nn.CrossEntropyLoss(reduction="none")

    for p in model.parameters():
        p.requires_grad = False

    layers = list(model.children())

    # Initializing the nodes, the descent direction and the next node
    n = [None] * (len(model.n)-1)
    direct = [None] * (len(model.n)-1)
    n_next = [None] * (len(model.n)-1)
    for i in range(len(n)):
        n[i] = torch.zeros(model.n[i].size()).to(device)
        direct[i] = torch.zeros(model.n[i].size()).to(device)
        n_next[i] = torch.zeros(model.n[i].size()).to(device)

    # Initializing the same list but after matrix multiplication so operations can be reused
    Mn = [None] * (len(model.n)-1)
    Mdirect = [None] * (len(model.n)-1)
    Mn_next = [None] * (len(model.n)-1)
    grad_prev = [None] * (len(model.n)-1)

    aux = [None] * (len(model.n)-2)
    norm_grad=torch.zeros((x.size(0)),device=device)
    alpha=mu*torch.ones((x.size(0),1),device=device)
    loss_next=torch.zeros((x.size(0)),device=device)

    # Defining some function
    crit=torch.nn.MSELoss(reduction='none')
    def rho(z,w):
        return crit(z.view(z.size(0), -1),w.view(w.size(0), -1)).sum(1)
    def reg(z):
        return (z.view(x.size(0), -1)**2).sum(1)
    def batch_dot_prod(z,w):
        return (z.view(x.size(0), -1)*w.view(x.size(0), -1)).sum(1)

    def compute_loss(n,Mn,serching_step=torch.tensor([True]*x.size(0))):
        # Calculating the loss
        loss =  rho(n[0][serching_step],x[serching_step])
        loss += lamb_layers*rho(n[1][serching_step],F.max_pool2d(F.leaky_relu(Mn[0]), (2, 2))[serching_step])
        loss += lamb_layers*rho(n[2][serching_step],F.max_pool2d(F.leaky_relu(Mn[1]), (2, 2)).view(x.size(0), -1)[serching_step])
        loss += lamb_layers*rho(n[3][serching_step],F.leaky_relu(Mn[2][serching_step]))
        loss += -criterion(Mn[3][serching_step], y_true[serching_step])/lamb
        return loss

    with torch.no_grad():
        x=layers[0](x)
        n[0]=x+eps_input*(2*torch.rand(x.size())-1).to(device)
        n[0]=torch.clamp(n[0],0.,1.)
    n[0].requires_grad_(True)
    Mn[0]=layers[1](n[0])
    with torch.no_grad():
        aux[0]=F.max_pool2d(F.leaky_relu(Mn[0]), (2, 2))
        n[1]=aux[0]+eps_init_layers*(2*torch.rand(model.n[1].size())-1).to(device)
        # n[1]=aux[0]+eps_init_layers*torch.randn(model.n[1].size()).to(device)
    n[1].requires_grad_(True)
    Mn[1]=layers[2](n[1])
    with torch.no_grad():
        aux[1]=F.max_pool2d(F.leaky_relu(Mn[1]), (2, 2)).view(x.size(0), -1)
        n[2]=aux[1]+eps_init_layers*(2*torch.rand(model.n[2].size())-1).to(device)
        # n[2]=aux[1]+eps_init_layers*torch.randn(model.n[2].size()).to(device)
    n[2].requires_grad_(True)
    Mn[2]=layers[3](n[2])
    with torch.no_grad():
        aux[2]=F.leaky_relu(Mn[2])
        n[3]=aux[2]+eps_init_layers*(2*torch.rand(model.n[3].size())-1).to(device)
        n[3]=aux[2]+eps_init_layers*torch.randn(model.n[3].size()).to(device)
    n[3].requires_grad_(True)
    Mn[3]=layers[4](n[3])



    solver = 'running'
    iter=0
    while iter<= num_iters and solver=='running':
        iter+=1

        # for i in range(len(n)):
        #     n[i].requires_grad_(True)
        #     Mn[i]=layers[i+1](n[i])



        loss=compute_loss(n,Mn)

        loss.backward(gradient=torch.ones_like(y_true, dtype=torch.float))

        with torch.no_grad():
            norm_grad_prev=norm_grad.clone()
            norm_grad.zero_()

            # First part: Finding conjugated direction for descent
            for i in range(len(n)):
                norm_grad+=batch_dot_prod(n[i].grad,n[i].grad)
            if iter == 1:
                norm_grad_init=norm_grad.clone()
                loss_init=loss.detach().clone()
                for i in range(len(n)):
                    direct[i]=-n[i].grad.clone() # Update the search direction with the new step to go
                    grad_prev[i]=n[i].grad.clone()
                grad_prod_direct=-norm_grad
                beta=torch.zeros((x.size(0),1),device=device)
            else:
                # Calculating beta
                beta=(norm_grad/norm_grad_prev).view(x.size(0),-1)
                for i in range(len(n)):
                    beta-=(batch_dot_prod(n[i].grad,grad_prev[i])/norm_grad_prev).view(x.size(0),-1)
                    grad_prev[i]=n[i].grad.clone()
                beta.relu_()
                grad_prod_direct.zero_()
                for i in range(len(n)):
                    direct[i].view(x.size(0),-1).mul_(beta) # Multiply the previous step direction of each batch by the value of beta that is equivalent. Equivalent to direct[i]*=beta.view([-1]+[1]*(direct[i].ndim-1)).expand_as(direct[i])
                    direct[i]+=-n[i].grad # Update the search direction with the new step to go
                    grad_prod_direct+=batch_dot_prod(n[i].grad,direct[i]) # This value is used to check whether the new direction is a descent direction and used in the selection of the optimal descent step


                # Checking if it is a descent direction and correcting it in the negative case
                isdescent=grad_prod_direct>0
                if torch.any(isdescent):
                    for i in range(len(n)):
                        direct[i][isdescent]=-n[i].grad[isdescent].clone()
                    grad_prod_direct[isdescent]=-norm_grad[isdescent]
                    if debug:
                        beta[isdescent]=0


            # Second part: Computing optimal step size using Armijo Rule

            # Maximum value that step can take
            alpha*=10
            loss_next.zero_()

            if iter ==1:
                for i in range(len(n)):
                    Mdirect[i]=layers[i+1](direct[i])
                    Mn_next[i]=torch.zeros_like(Mn[i])
            else:
                for i in range(len(n)):
                    Mdirect[i]=layers[i+1](direct[i])
                    Mn_next[i].zero_()

            serching_step=torch.tensor([True]*x.size(0))
            while torch.any(serching_step) and alpha.min()>=mu*1e-15:
                # Update the new value of the neurons with the step direction alpha.
                # n_next[0][serching_step]=n[0][serching_step]+direct[0].view(x.size(0),-1).mul(alpha).view_as(n[0])[serching_step]
                # n_next[0][serching_step]=x[serching_step]+torch.clamp(n_next[0][serching_step] - x[serching_step],-eps_input,eps_input)
                # n_next[0][serching_step]=torch.clamp(n_next[0][serching_step],0.,1.)
                # Mn_next[0][serching_step]=layers[1](n_next[0][serching_step])

                # for i in range(len(n)):
                #     n_next[i][serching_step]=n[i][serching_step]+direct[i].view(x.size(0),-1).mul(alpha).view_as(n[i])[serching_step]
                #     Mn_next[i][serching_step]=Mn[i][serching_step]+Mdirect[i].view(x.size(0),-1).mul(alpha).view_as(Mn[i])[serching_step]

                for i in range(len(n)):
                    n_next[i][serching_step]=n[i][serching_step]+direct[i].view(x.size(0),-1)[serching_step].mul(alpha[serching_step]).view_as(n[i][serching_step])
                    Mn_next[i][serching_step]=Mn[i][serching_step]+Mdirect[i].view(x.size(0),-1)[serching_step].mul(alpha[serching_step]).view_as(Mn[i][serching_step])

                # Calculating the next loss. This step could be optimize by avoiding to compute the loss for the directions already found
                loss_next[serching_step]=compute_loss(n_next,Mn_next,serching_step)
                # Checking whether the Armijo criteria is satisfied
                serching_step[(loss-loss_next)>=0*-1e-3*alpha.squeeze()*grad_prod_direct]=False

                # For the elements in the batch where it is not, ie, serching_step=True, update the step
                alpha[serching_step]*=0.3


            for i in range(len(n)):
                n[i].grad.zero_()
                n[i]=n_next[i].clone()
            # breakpoint()

            if iter>=10:
                norm_ratio=norm_grad/norm_grad_init
                if norm_ratio.max()<tol_grad:
                    solver='solved'
                elif iter>=20 and (norm_ratio.min()>=1 or (loss_init-loss).max()<0):
                    solver='failed'
                elif iter>=40 and norm_ratio.median()<tol_grad:
                    solver='partially solved'
                # elif norm_grad_prev.max()<norm_grad.max():
                    # solver='failed'

            if debug:
                print("Iter:", iter, ", Loss max: ", loss.max().data.cpu().numpy(), ", Loss min:", loss.min().data.cpu().numpy(), ", Output Loss", criterion(Mn_next[3], y_true).max().data.cpu().numpy())
                # print("      Grad n[0]: ", n[0].grad.abs().mean().data.cpu().numpy()," Grad n[1]: ", n[1].grad.abs().mean().data.cpu().numpy()," Grad n[2]: ", n[2].grad.abs().mean().data.cpu().numpy()," Grad n[3]: ", n[3].grad.abs().mean().data.cpu().numpy())
                print("      Beta:", beta.median().data.cpu().numpy(),"norm_grad:", (norm_grad/norm_grad_init).max().data.cpu().numpy(), "alpha: ", alpha.min().data.cpu().numpy())


        with torch.no_grad():
            n[0]=x+torch.clamp(n[0] - x,-eps_input,eps_input)
            n[0]=torch.clamp(n[0],0.,1.)
        n[0].requires_grad_(True)
        Mn[0]=layers[1](n[0])
        with torch.no_grad():
            aux[0]=F.max_pool2d(F.leaky_relu(Mn[0]), (2, 2))
            n[1]=aux[0]+torch.clamp(n[1] - aux[0],-eps_layers,eps_layers)
        n[1].requires_grad_(True)
        Mn[1]=layers[2](n[1])
        with torch.no_grad():
            aux[1]=F.max_pool2d(F.leaky_relu(Mn[1]), (2, 2)).view(x.size(0), -1)
            n[2]=aux[1]+torch.clamp(n[2] - aux[1],-eps_layers,eps_layers)
        n[2].requires_grad_(True)
        Mn[2]=layers[3](n[2])
        with torch.no_grad():
            aux[2]=F.leaky_relu(Mn[2])
            n[3]=aux[2]+torch.clamp(n[3] - aux[2],-eps_layers,eps_layers)
        n[3].requires_grad_(True)
        Mn[3]=layers[4](n[3])

    if torch.any(torch.isnan(loss)) or torch.any(torch.isnan(n[0])) or torch.any(torch.isnan(n[1])) or torch.any(torch.isnan(n[2])):
        raise ValueError('Diverged')
    # if torch.any(loss.abs()>1e5) or torch.any(n[0].abs()>1e3) or torch.any(n[1].abs()>1e3) or torch.any(n[2].abs()>1e3) or torch.any(n[3].abs()>1e3):
        # breakpoint()

    with torch.no_grad():

        disturbance = [None] * (len(model.n)-1)
        # for i in range(1,len(n)):
        #     disturbance[i] = torch.zeros_like(model.n[i])

        # Input disturbance is always considered as solved. Worst case it is just noise

        disturbance[0] = (n[0] - x)

        disturbance[1] = (n[1] - aux[0])
        disturbance[2] = (n[2] - aux[1])
        disturbance[3] = (n[3] - aux[2])

        # For those where it was solved, input the solved solution
        # disturbance[1][norm_ratio<tol_grad] = n_next[1][norm_ratio<tol_grad] - F.max_pool2d(F.leaky_relu(Mn_next[0]), (2, 2))[norm_ratio<tol_grad]
        # disturbance[2][norm_ratio<tol_grad] = n_next[2][norm_ratio<tol_grad] - F.max_pool2d(F.leaky_relu(Mn_next[1]), (2, 2)).view(x.size(0), -1)[norm_ratio<tol_grad]
        # disturbance[3][norm_ratio<tol_grad] = n_next[3][norm_ratio<tol_grad] - F.leaky_relu(Mn_next[2])[norm_ratio<tol_grad]

        # # For those where it was not, input random small disturbance
        # disturbance[1][norm_ratio>=tol_grad] = 0.05*torch.randn_like(disturbance[1][norm_ratio>=tol_grad])
        # disturbance[2][norm_ratio>=tol_grad] = 0.05*torch.randn_like(disturbance[2][norm_ratio>=tol_grad])
        # disturbance[3][norm_ratio>=tol_grad] = 0.05*torch.randn_like(disturbance[3][norm_ratio>=tol_grad])


        if debug:
            print(" d[0]: ", disturbance[0].abs().max().data.cpu().numpy() ," d[1]: ", disturbance[1].abs().max().data.cpu().numpy()," d[2]: ", disturbance[2].abs().max().data.cpu().numpy()," d[3]: ", disturbance[3].abs().max().data.cpu().numpy())
            print(solver)
            # breakpoint()
            # time.sleep(0.000001)

        for i in range(len(n)):
            model.d[i] = disturbance[i]

        if False and solver!="solved":
            print(solver)
            print("Iter:", iter, ", Loss max: ", loss.max().data.cpu().numpy(), ", Loss min:", loss.min().data.cpu().numpy(), ", Output Loss", criterion(Mn_next[3], y_true).max().data.cpu().numpy())
            print(" d[0]: ", disturbance[0].abs().max().data.cpu().numpy() ," d[1]: ", disturbance[1].abs().max().data.cpu().numpy()," d[2]: ", disturbance[2].abs().max().data.cpu().numpy()," d[3]: ", disturbance[3].abs().max().data.cpu().numpy())
            # breakpoint()

    for p in model.parameters():
        p.requires_grad = True

def DistortNeuronsConjugateGradientLineSearchV2(model, x, y_true, lamb, mu, optimizer=None):
    '''
    Descriptions:
        Conjugate Gradient
        Line search using Armijo Rule
        No projection of n[0]  but included penalization norm(n[0]-0.5) to incentivize n[0] to stay in (0,1)
    '''
    # Parameters
    num_iters = 200
    eps_input = 0.3 # value to clamp input disturbance
    eps_layers = 0.5 # value to clamp layer disturbance
    lamb_input = 120
    lamb_d0 = 150
    lamb_layers = 160 # how many times regularization in inside layers
    tol_grad=1e-3 # value gradient problem considered solved
    debug = True# set to true to activate break points and prints
    #Code
    model.eval()
    device = model.parameters().__next__().device
    criterion = nn.CrossEntropyLoss(reduction="none")

    for p in model.parameters():
        p.requires_grad = False

    layers = list(model.children())

    # Initializing the nodes, the descent direction and the next node
    n = [None] * (len(model.n)-1)
    direct = [None] * (len(model.n)-1)
    n_next = [None] * (len(model.n)-1)
    grad_prev = [None] * (len(model.n)-1)
    for i in range(len(n)):
        n[i] = torch.zeros(model.n[i].size()).to(device)
        direct[i] = torch.zeros(model.n[i].size()).to(device)
        n_next[i] = torch.zeros(model.n[i].size()).to(device)
        grad_prev[i] = torch.zeros(model.n[i].size()).to(device)

    # Initializing the same list but after matrix multiplication so operations can be reused
    Mn = [None] * (len(model.n)-1)
    Mdirect = [None] * (len(model.n)-1)
    Mn_next = [None] * (len(model.n)-1)


    norm_grad=torch.zeros((x.size(0)),device=device)
    alpha=mu*torch.ones((x.size(0),1),device=device)
    loss_next=torch.zeros((x.size(0)),device=device)

    # Defining some function
    crit=torch.nn.MSELoss(reduction='none')
    def rho(z,w):
        return crit(z.view(z.size(0), -1),w.view(w.size(0), -1)).sum(1)
    def reg(z):
        return (z.view(z.size(0), -1)**2).sum(1)
    def batch_dot_prod(z,w):
        return (z.view(x.size(0), -1)*w.view(x.size(0), -1)).sum(1)

    def compute_loss(n,Mn,serching_step=torch.tensor([True]*x.size(0))):
        # Calculating the loss
        loss =  lamb_d0*rho(n[0][serching_step],x[serching_step])+lamb_input*reg(n[0][serching_step]-0.5)
        loss += lamb_layers*rho(n[1][serching_step],F.max_pool2d(F.leaky_relu(Mn[0]), (2, 2))[serching_step])
        loss += lamb_layers*rho(n[2][serching_step],F.max_pool2d(F.leaky_relu(Mn[1]), (2, 2)).view(x.size(0), -1)[serching_step])
        loss += lamb_layers*rho(n[3][serching_step],F.leaky_relu(Mn[2][serching_step]))
        loss += -criterion(Mn[3][serching_step], y_true[serching_step])
        return loss


    with torch.no_grad():
        x=layers[0](x)
        n[0]=x+eps_input*(2*torch.rand(x.size())-1).to(device)
        n[0]=torch.clamp(n[0],0.,1.)
        Mn[0]=layers[1](n[0])
        n[1]=F.max_pool2d(F.leaky_relu(Mn[0]), (2, 2))+eps_layers*(2*torch.rand(model.n[1].size())-1).to(device)
        Mn[1]=layers[2](n[1])
        n[2]=F.max_pool2d(F.leaky_relu(Mn[1]), (2, 2)).view(x.size(0), -1)+eps_layers*(2*torch.rand(model.n[2].size())-1).to(device)
        Mn[2]=layers[3](n[2])
        n[3]=F.leaky_relu(Mn[2])+eps_layers*(2*torch.rand(model.n[3].size())-1).to(device)
        Mn[3]=layers[4](n[3])


    # Initialize some parameters that need the size of Mn[i] to be initialized
    for i in range(len(n)):
        Mdirect[i]=torch.zeros_like(Mn[i])
        Mn_next[i]=torch.zeros_like(Mn[i])



    solver = 'running'
    iter=0
    while iter<= num_iters and solver=='running':
        iter+=1

        for i in range(len(n)):
            n[i].requires_grad_(True)
            Mn[i]=layers[i+1](n[i])



        loss=compute_loss(n,Mn)

        loss.backward(gradient=torch.ones_like(y_true, dtype=torch.float))

        with torch.no_grad():
            norm_grad_prev=norm_grad.clone()
            norm_grad.zero_()

            # First part: Finding conjugated direction for descent
            for i in range(len(n)):
                norm_grad+=batch_dot_prod(n[i].grad,n[i].grad)
            if iter == 1:
                norm_grad_init=norm_grad.clone()
                loss_init=loss.detach().clone()
                for i in range(len(n)):
                    direct[i]=-n[i].grad.clone() # Update the search direction with the new step to go
                    grad_prev[i]=n[i].grad.clone()
                grad_prod_direct=-norm_grad
                beta=torch.zeros((x.size(0),1),device=device)
            else:
                # Calculating beta
                beta=(norm_grad/norm_grad_prev).view(x.size(0),-1)
                for i in range(len(n)):
                    beta-=(batch_dot_prod(n[i].grad,grad_prev[i])/norm_grad_prev).view(x.size(0),-1)
                    grad_prev[i]=n[i].grad.clone()
                beta.relu_()
                grad_prod_direct.zero_()
                for i in range(len(n)):
                    direct[i].view(x.size(0),-1).mul_(beta) # Multiply the previous step direction of each batch by the value of beta that is equivalent. Equivalent to direct[i]*=beta.view([-1]+[1]*(direct[i].ndim-1)).expand_as(direct[i])
                    direct[i]+=-n[i].grad # Update the search direction with the new step to go
                    grad_prod_direct+=batch_dot_prod(n[i].grad,direct[i]) # This value is used to check whether the new direction is a descent direction and used in the selection of the optimal descent step


                # Checking if it is a descent direction and correcting it in the negative case
                isdescent=grad_prod_direct>0
                if torch.any(isdescent):
                    for i in range(len(n)):
                        direct[i][isdescent]=-n[i].grad[isdescent].clone()
                    grad_prod_direct[isdescent]=-norm_grad[isdescent]
                    if debug:
                        beta[isdescent]=0


            # Second part: Computing optimal step size using Armijo Rule

            # Maximum value that step can take

            loss_next.zero_()
            if iter ==1:
                for i in range(len(n)):
                    Mdirect[i]=layers[i+1](direct[i])
                    Mn_next[i]=torch.zeros_like(Mn[i])
            else:
                alpha*=10
                for i in range(len(n)):
                    Mdirect[i]=layers[i+1](direct[i])
                    Mn_next[i].zero_()

            serching_step=torch.tensor([True]*x.size(0))
            while torch.any(serching_step) and alpha.min()>=mu*1e-15:
                # Update the new value of the neurons with the step direction alpha.

                for i in range(len(n)):
                    n_next[i][serching_step]=n[i][serching_step]+direct[i].view(x.size(0),-1)[serching_step].mul(alpha[serching_step]).view_as(n[i][serching_step])
                    Mn_next[i][serching_step]=Mn[i][serching_step]+Mdirect[i].view(x.size(0),-1)[serching_step].mul(alpha[serching_step]).view_as(Mn[i][serching_step])

                # Calculating the next loss. This step could be optimize by avoiding to compute the loss for the directions already found
                loss_next[serching_step]=compute_loss(n_next,Mn_next,serching_step)
                # Checking whether the Armijo criteria is satisfied
                serching_step[(loss-loss_next)>=-0*alpha.squeeze()*grad_prod_direct]=False

                # For the elements in the batch where it is not, ie, serching_step=True, update the step
                alpha[serching_step]*=0.3


            for i in range(len(n)):
                n[i].grad.zero_()
                n[i]=n_next[i].clone()
                Mn[i]=Mn_next[i]
            # breakpoint()

            if iter>=10:
                norm_ratio=norm_grad/norm_grad_init
                if norm_ratio.max()<tol_grad:
                    solver='solved'
                elif iter>=20 and (norm_ratio.min()>=1 or (loss_init-loss).max()<0):
                    solver='failed'
                elif iter>=40 and norm_ratio.median()<tol_grad:
                    solver='partially solved'
                # elif norm_grad_prev.max()<norm_grad.max():
                    # solver='failed'

            if debug:
                print("Iter:", iter, ", Loss max: ", loss.max().data.cpu().numpy(), ", Loss min:", loss.min().data.cpu().numpy(), ", Output Loss", criterion(Mn_next[3], y_true).max().data.cpu().numpy())
                # print("      Grad n[0]: ", n[0].grad.abs().mean().data.cpu().numpy()," Grad n[1]: ", n[1].grad.abs().mean().data.cpu().numpy()," Grad n[2]: ", n[2].grad.abs().mean().data.cpu().numpy()," Grad n[3]: ", n[3].grad.abs().mean().data.cpu().numpy())
                print("      Beta:", beta.median().data.cpu().numpy(),"norm_grad:", (norm_grad/norm_grad_init).max().data.cpu().numpy(), "alpha: ", alpha.min().data.cpu().numpy())



    if torch.any(torch.isnan(loss)) or torch.any(torch.isnan(n[0])) or torch.any(torch.isnan(n[1])) or torch.any(torch.isnan(n[2])):
        raise ValueError('Diverged')
    # if torch.any(loss.abs()>1e5) or torch.any(n[0].abs()>1e3) or torch.any(n[1].abs()>1e3) or torch.any(n[2].abs()>1e3) or torch.any(n[3].abs()>1e3):
        # breakpoint()

    with torch.no_grad():

        disturbance = [None] * (len(model.n)-1)
        # for i in range(1,len(n)):
        #     disturbance[i] = torch.zeros_like(model.n[i])

        # Input disturbance is always considered as solved. Worst case it is just noise

        disturbance[0]=x-n[0]
        disturbance[1]=n[1]-F.max_pool2d(F.leaky_relu(Mn[0]), (2, 2))
        disturbance[2]=n[2]-F.max_pool2d(F.leaky_relu(Mn[1]), (2, 2)).view(x.size(0), -1)
        disturbance[3]=n[3]-F.leaky_relu(Mn[2])


        if debug:
            print("      n[0]", (n[0]-0.5).abs().max().data.cpu().numpy(), " d[0]: ", disturbance[0].abs().max().data.cpu().numpy() ," d[1]: ", disturbance[1].abs().max().data.cpu().numpy()," d[2]: ", disturbance[2].abs().max().data.cpu().numpy()," d[3]: ", disturbance[3].abs().max().data.cpu().numpy())
            print(solver)
            # breakpoint()
            # time.sleep(0.000001)

        for i in range(len(n)):
            model.d[i] = disturbance[i]

        if False and solver!="solved":
            print(solver)
            print("Iter:", iter, ", Loss max: ", loss.max().data.cpu().numpy(), ", Loss min:", loss.min().data.cpu().numpy(), ", Output Loss", criterion(Mn_next[3], y_true).max().data.cpu().numpy())
            print("      n[0]", (n[0]-0.5).abs().max().data.cpu().numpy(), " d[0]: ", disturbance[0].abs().max().data.cpu().numpy() ," d[1]: ", disturbance[1].abs().max().data.cpu().numpy()," d[2]: ", disturbance[2].abs().max().data.cpu().numpy()," d[3]: ", disturbance[3].abs().max().data.cpu().numpy())
            # breakpoint()

    for p in model.parameters():
        p.requires_grad = True




def DistortNeuronsGradientDescentLineSearch(model, x, y_true, lamb, mu, optimizer=None):
    '''
    Descriptions:
        Gradient Descent
        Line search using Armijo Rule
        Projection of n[0] into correct set (abs(n[0]-x)<0.3, 0<=n[0]<=1)
    '''
    # Parameters
    num_iters = 200
    eps_input = 0.3 # value to clamp input disturbance
    eps_layers = 2 # value to clamp layer disturbance
    lamb_layers = 20 # how many times regularization in inside layers
    eps_init_layers= 2# math.sqrt(1/(lamb*lamb_layers)) # "std" for the initialization of disturbances
    tol_grad=1e-3 # value gradient problem considered solved
    #lamb_reg = 0.01 # regularization of the intermediate layers
    debug = True# set to true to activate break points and prints
    #Code
    model.eval()
    device = model.parameters().__next__().device
    criterion = nn.CrossEntropyLoss(reduction="none")

    for p in model.parameters():
        p.requires_grad = False

    layers = list(model.children())

    # Initializing the nodes, the descent direction and the next node
    n = [None] * (len(model.n)-1)
    n_next = [None] * (len(model.n)-1)
    for i in range(len(n)):
        n[i] = torch.zeros(model.n[i].size()).to(device)
        n_next[i] = torch.zeros(model.n[i].size()).to(device)

    # Initializing the same list but after matrix multiplication so operations can be reused
    Mn = [None] * (len(model.n)-1)
    Mgrad = [None] * (len(model.n)-1)
    Mn_next = [None] * (len(model.n)-1)

    aux = [None] * (len(model.n)-2)
    alpha=mu*torch.ones((x.size(0),1),device=device)
    loss_next=torch.zeros((x.size(0)),device=device)
    norm_grad=torch.zeros((x.size(0)),device=device)



    # Defining some function
    crit=torch.nn.MSELoss(reduction='none')
    def rho(z,w):
        return crit(z.view(z.size(0), -1),w.view(w.size(0), -1)).sum(1)
    def reg(z):
        return (z.view(x.size(0), -1)**2).sum(1)
    def batch_dot_prod(z,w):
        return (z.view(x.size(0), -1)*w.view(x.size(0), -1)).sum(1)

    def compute_loss(n,Mn,serching_step=torch.tensor([True]*x.size(0))):
        # Calculating the loss
        loss =  rho(n[0][serching_step],x[serching_step])
        loss += lamb_layers*rho(n[1][serching_step],F.max_pool2d(F.leaky_relu(Mn[0]), (2, 2))[serching_step])
        loss += lamb_layers*rho(n[2][serching_step],F.max_pool2d(F.leaky_relu(Mn[1]), (2, 2)).view(x.size(0), -1)[serching_step])
        loss += lamb_layers*rho(n[3][serching_step],F.leaky_relu(Mn[2][serching_step]))
        loss += -criterion(Mn[3][serching_step], y_true[serching_step])/lamb
        return loss


    with torch.no_grad():
        x=layers[0](x)
        n[0]=x+eps_input*(2*torch.rand(x.size())-1).to(device)
        n[0]=torch.clamp(n[0],0.,1.)
    n[0].requires_grad_(True)
    Mn[0]=layers[1](n[0])
    with torch.no_grad():
        aux[0]=F.max_pool2d(F.leaky_relu(Mn[0]), (2, 2))
        n[1]=aux[0]+eps_init_layers*(2*torch.rand(model.n[1].size())-1).to(device)
        # n[1]=aux[0]+eps_init_layers*torch.randn(model.n[1].size()).to(device)
    n[1].requires_grad_(True)
    Mn[1]=layers[2](n[1])
    with torch.no_grad():
        aux[1]=F.max_pool2d(F.leaky_relu(Mn[1]), (2, 2)).view(x.size(0), -1)
        n[2]=aux[1]+eps_init_layers*(2*torch.rand(model.n[2].size())-1).to(device)
        # n[2]=aux[1]+eps_init_layers*torch.randn(model.n[2].size()).to(device)
    n[2].requires_grad_(True)
    Mn[2]=layers[3](n[2])
    with torch.no_grad():
        aux[2]=F.leaky_relu(Mn[2])
        n[3]=aux[2]+eps_init_layers*(2*torch.rand(model.n[3].size())-1).to(device)
        n[3]=aux[2]+eps_init_layers*torch.randn(model.n[3].size()).to(device)
    n[3].requires_grad_(True)
    Mn[3]=layers[4](n[3])


    solver = 'running'
    iter=0
    while iter<= num_iters and solver=='running':
        iter+=1



        # for i in range(len(n)):
        #     n[i].requires_grad_(True)
        #     Mn[i]=layers[i+1](n[i])



        loss=compute_loss(n,Mn)

        loss.backward(gradient=torch.ones_like(y_true, dtype=torch.float))

        with torch.no_grad():
            norm_grad.zero_()
            # First part: Computing Descent Direction
            for i in range(len(n)):
                norm_grad+=batch_dot_prod(n[i].grad,n[i].grad)
            if iter == 1:
                norm_grad_init=norm_grad.clone()

            # Second part: Computing optimal step size using Armijo Rule

            # Maximum value that step can take
            alpha*=10
            loss_next.zero_()

            if iter ==1:
                for i in range(len(n)):
                    Mgrad[i]=layers[i+1](n[i].grad)
                    Mn_next[i]=torch.zeros_like(Mn[i])
            else:
                for i in range(len(n)):
                    Mgrad[i]=layers[i+1](n[i].grad)
                    Mn_next[i].zero_()


            serching_step=torch.tensor([True]*x.size(0))
            while torch.any(serching_step) and alpha.min()>=mu*1e-15:
                # Update the new value of the neurons with the step direction alpha.

                # n_next[0][serching_step]=n[0][serching_step]-n[0].grad.view(x.size(0),-1)[serching_step].mul(alpha[serching_step]).view_as(n[0][serching_step])
                # n_next[0][serching_step]=x[serching_step]+torch.clamp(n_next[0][serching_step] - x[serching_step],-eps_input,eps_input)
                # n_next[0][serching_step]=torch.clamp(n_next[0][serching_step],0.,1.)
                # Mn_next[0][serching_step]=layers[1](n_next[0][serching_step])

                for i in range(len(n)):
                    n_next[i][serching_step]=n[i][serching_step]-n[i].grad.view(x.size(0),-1)[serching_step].mul(alpha[serching_step]).view_as(n[i][serching_step])
                    Mn_next[i][serching_step]=Mn[i][serching_step]-Mgrad[i].view(x.size(0),-1)[serching_step].mul(alpha[serching_step]).view_as(Mn[i][serching_step])


                # Calculating the next loss. This step could be optimize by avoiding to compute the loss for the directions already found
                loss_next[serching_step]=compute_loss(n_next,Mn_next,serching_step)
                # Checking whether the Armijo criteria is satisfied
                serching_step[(loss-loss_next)>0*1e-10*alpha.squeeze()*norm_grad]=False

                # For the elements in the batch where it is not, ie, serching_step=True, update the step
                alpha[serching_step]*=0.3


            for i in range(len(n)):
                n[i]=n_next[i].clone()

            # breakpoint()

            if iter>=10:
                norm_ratio=norm_grad/norm_grad_init
                if norm_ratio.max()<tol_grad:
                    solver='solved'
                elif iter>=40 and norm_ratio.median()<tol_grad:
                    solver='partially solved'
                # elif norm_grad_prev.max()<norm_grad.max():
                    # solver='failed'

            if debug:
                print("Iter:", iter, ", Loss max: ", loss.max().data.cpu().numpy(), ", Loss min:", loss.min().data.cpu().numpy(), ", Output Loss", criterion(Mn_next[3], y_true).max().data.cpu().numpy())
                # print("      Grad n[0]: ", n[0].grad.abs().mean().data.cpu().numpy()," Grad n[1]: ", n[1].grad.abs().mean().data.cpu().numpy()," Grad n[2]: ", n[2].grad.abs().mean().data.cpu().numpy()," Grad n[3]: ", n[3].grad.abs().mean().data.cpu().numpy())
                print("      norm_grad:", (norm_grad/norm_grad_init).max().data.cpu().numpy(), "alpha: ", alpha.min().data.cpu().numpy())

        with torch.no_grad():
            n[0]=x+torch.clamp(n[0] - x,-eps_input,eps_input)
            n[0]=torch.clamp(n[0],0.,1.)
        n[0].requires_grad_(True)
        Mn[0]=layers[1](n[0])
        with torch.no_grad():
            aux[0]=F.max_pool2d(F.leaky_relu(Mn[0]), (2, 2))
            n[1]=aux[0]+torch.clamp(n[1] - aux[0],-eps_layers,eps_layers)
        n[1].requires_grad_(True)
        Mn[1]=layers[2](n[1])
        with torch.no_grad():
            aux[1]=F.max_pool2d(F.leaky_relu(Mn[1]), (2, 2)).view(x.size(0), -1)
            n[2]=aux[1]+torch.clamp(n[2] - aux[1],-eps_layers,eps_layers)
        n[2].requires_grad_(True)
        Mn[2]=layers[3](n[2])
        with torch.no_grad():
            aux[2]=F.leaky_relu(Mn[2])
            n[3]=aux[2]+torch.clamp(n[3] - aux[2],-eps_layers,eps_layers)
        n[3].requires_grad_(True)
        Mn[3]=layers[4](n[3])


    if torch.any(torch.isnan(loss)) or torch.any(torch.isnan(n[0])) or torch.any(torch.isnan(n[1])) or torch.any(torch.isnan(n[2])):
        raise ValueError('Diverged')
    # if torch.any(loss.abs()>1e5) or torch.any(n[0].abs()>1e3) or torch.any(n[1].abs()>1e3) or torch.any(n[2].abs()>1e3) or torch.any(n[3].abs()>1e3):
        # breakpoint()

    with torch.no_grad():

        disturbance = [None] * (len(model.n)-1)
        for i in range(1,len(n)):
            disturbance[i] = torch.zeros_like(model.n[i])

        # Input disturbance is always considered as solved. Worst case it is just noise


        disturbance[0] = (n[0] - x)

        # For those where it was solved, input the solved solution
        disturbance[1] = (n[1] - aux[0])
        disturbance[2] = (n[2] - aux[1])
        disturbance[3] = (n[3] - aux[2])


        # aux[0]=F.max_pool2d(F.leaky_relu(Mn_next[0]), (2, 2))
        # disturbance[1]=torch.clamp(n[1] - aux[0],-eps_layers,eps_layers)

        # n[1]=aux[0]+disturbance[1]
        # aux[1]=F.max_pool2d(F.leaky_relu(layers[2](n[1])), (2, 2)).view(x.size(0), -1)
        # disturbance[2]=torch.clamp(n[2] - aux[1],-eps_layers,eps_layers)

        # n[2]=aux[1]+disturbance[2]
        # aux[2]=F.leaky_relu(layers[3](n[2]))
        # disturbance[3] = torch.clamp(n[3] - aux[2],-eps_layers,eps_layers)


        # # For those where it was solved, input the solved solution
        # disturbance[1][norm_ratio<tol_grad] = n_next[1][norm_ratio<tol_grad] - F.max_pool2d(F.leaky_relu(Mn_next[0]), (2, 2))[norm_ratio<tol_grad]
        # disturbance[2][norm_ratio<tol_grad] = n_next[2][norm_ratio<tol_grad] - F.max_pool2d(F.leaky_relu(Mn_next[1]), (2, 2)).view(x.size(0), -1)[norm_ratio<tol_grad]
        # disturbance[3][norm_ratio<tol_grad] = n_next[3][norm_ratio<tol_grad] - F.leaky_relu(Mn_next[2])[norm_ratio<tol_grad]

        # # For those where it was not, input random small disturbance
        # disturbance[1][norm_ratio>=tol_grad] = 0.05*torch.randn_like(disturbance[1][norm_ratio>=tol_grad])
        # disturbance[2][norm_ratio>=tol_grad] = 0.05*torch.randn_like(disturbance[2][norm_ratio>=tol_grad])
        # disturbance[3][norm_ratio>=tol_grad] = 0.05*torch.randn_like(disturbance[3][norm_ratio>=tol_grad])


        if debug:
            print(" d[0]: ", disturbance[0].abs().max().data.cpu().numpy() ," d[1]: ", disturbance[1].abs().max().data.cpu().numpy()," d[2]: ", disturbance[2].abs().max().data.cpu().numpy()," d[3]: ", disturbance[3].abs().max().data.cpu().numpy())
            print(solver)
            # breakpoint()
            # time.sleep(0.000001)

        for i in range(len(n)):
            model.d[i] = disturbance[i]

        if False and solver!="solved":
            print(solver)
            print("Iter:", iter, ", Loss max: ", loss.max().data.cpu().numpy(), ", Loss min:", loss.min().data.cpu().numpy(), ", Output Loss", criterion(Mn_next[3], y_true).max().data.cpu().numpy())
            print(" d[0]: ", disturbance[0].abs().max().data.cpu().numpy() ," d[1]: ", disturbance[1].abs().max().data.cpu().numpy()," d[2]: ", disturbance[2].abs().max().data.cpu().numpy()," d[3]: ", disturbance[3].abs().max().data.cpu().numpy())
            # breakpoint()

    for p in model.parameters():
        p.requires_grad = True


def DistortNeuronsGradientDescentCoordinate(model, x, y_true, lamb, mu, optimizer=None):
    '''
    Descriptions:
        Coordinate Descent, alternate between optimizing n[0] and projecting it and the optimizing n[1], n[2] and n[3]
        Gradient Descent
        Line search using Armijo Rule
        Projection of n[0] into correct set (abs(n[0]-x)<0.3, 0<=n[0]<=1)
    '''
    # Parameters
    num_iters = 200
    eps_input = 0.3 # value to clamp input disturbance
    eps_layers = 2 # value to clamp layer disturbance
    lamb_layers = 1. # how many times regularization in inside layers
    tol_grad=1e-3 # value gradient problem considered solved
    #lamb_reg = 0.01 # regularization of the intermediate layers
    debug = True# set to true to activate break points and prints
    #Code
    model.eval()
    device = model.parameters().__next__().device
    criterion = nn.CrossEntropyLoss(reduction="none")

    for p in model.parameters():
        p.requires_grad = False

    layers = list(model.children())

    # Initializing the nodes, the descent direction and the next node
    n = [None] * (len(model.n)-1)
    n_next = [None] * (len(model.n)-1)
    d = [None] * (len(model.n)-1)
    for i in range(len(n)):
        n[i] = torch.zeros(model.n[i].size()).to(device)
        n_next[i] = torch.zeros(model.n[i].size()).to(device)
        d[i] = torch.zeros_like(model.n[i])


    # Initializing the same list but after matrix multiplication so operations can be reused
    Mn = [None] * (len(model.n)-1)
    Mgrad = [None] * (len(model.n)-1)
    Mn_next = [None] * (len(model.n)-1)

    alpha=mu*torch.ones((x.size(0),1),device=device)
    loss_next=torch.zeros((x.size(0)),device=device)
    norm_grad_input=torch.zeros((x.size(0)),device=device)
    norm_grad_hidden=torch.zeros((x.size(0)),device=device)



    # Defining some function
    crit=torch.nn.MSELoss(reduction='none')
    def rho(z,w):
        return crit(z.view(z.size(0), -1),w.view(w.size(0), -1)).sum(1)

    def batch_dot_prod(z,w):
        return (z.view(x.size(0), -1)*w.view(x.size(0), -1)).sum(1)

    def compute_loss_input(n,Mn,serching_step=torch.tensor([True]*x.size(0))):
        # Calculating the loss
        loss =  rho(n[0][serching_step],x[serching_step])
        loss += lamb_layers*rho(n[1][serching_step],F.max_pool2d(F.leaky_relu(Mn[0]), (2, 2))[serching_step])
        return loss

    def compute_loss_hidden(n,Mn,serching_step=torch.tensor([True]*x.size(0))):
        # Calculating the loss
        loss =  lamb_layers*rho(n[1][serching_step],F.max_pool2d(F.leaky_relu(Mn[0]), (2, 2))[serching_step])
        loss += lamb_layers*rho(n[2][serching_step],F.max_pool2d(F.leaky_relu(Mn[1]), (2, 2)).view(x.size(0), -1)[serching_step])
        loss += lamb_layers*rho(n[3][serching_step],F.leaky_relu(Mn[2][serching_step]))
        loss += -criterion(Mn[3][serching_step], y_true[serching_step])/lamb
        return loss

    with torch.no_grad():
        x=layers[0](x)
        n[0]=x+0.1*eps_input*(2*torch.rand(x.size())-1).to(device)
        n[0]=torch.clamp(n[0],0.1,0.9)
        Mn[0]=layers[1](n[0])
        n[1]=F.max_pool2d(F.leaky_relu(Mn[0]), (2, 2))+0.1*eps_layers*(2*torch.rand(model.n[1].size())-1).to(device)
        Mn[1]=layers[2](n[1])
        n[2]=F.max_pool2d(F.leaky_relu(Mn[1]), (2, 2)).view(x.size(0), -1)+0.1*eps_layers*(2*torch.rand(model.n[2].size())-1).to(device)
        Mn[2]=layers[3](n[2])
        n[3]=F.leaky_relu(Mn[2])+0.1*eps_layers*(2*torch.rand(model.n[3].size())-1).to(device)
        Mn[3]=layers[4](n[3])


    # Initialize some parameters that need the size of Mn[i] to be initialized
    for i in range(len(n)):
        Mgrad[i]=torch.zeros_like(Mn[i])
        Mn_next[i]=torch.zeros_like(Mn[i])


    solver = 'running'
    iter=0
    while iter<= num_iters and solver=='running':
        iter+=1

        # This algorithm computes the steps on a coordinate descent base. It starts with n[0]:

        # Part 1: Optimizing the input
        # 1.1: Computing Descent Direction for n[0]
        n[0].requires_grad_(True)
        Mn[0]=layers[1](n[0])


        loss=compute_loss_input(n,Mn)
        loss.backward(gradient=torch.ones_like(y_true, dtype=torch.float))

        n[0].requires_grad_(False)
        Mn[0].detach_()

        norm_grad_input.zero_()
        norm_grad_input=batch_dot_prod(n[0].grad,n[0].grad)
        if iter == 1:
            norm_grad_input_init=norm_grad_input.clone()



        # 1.2: Computing optimal step size using Armijo Rule

        alpha[:]=mu
        loss_next.zero_()
        Mgrad[0]=layers[1](n[0].grad)
        Mn_next[0].zero_()

        serching_step=torch.tensor([True]*x.size(0))
        while torch.any(serching_step) and alpha.min()>=mu*1e-15:
            # Update the new value of the neurons with the step direction alpha.

            n_next[0][serching_step]=n[0][serching_step]-n[0].grad.view(x.size(0),-1)[serching_step].mul(alpha[serching_step]).view_as(n[0][serching_step])
            Mn_next[0][serching_step]=Mn[0][serching_step]-Mgrad[0].view(x.size(0),-1)[serching_step].mul(alpha[serching_step]).view_as(Mn[0][serching_step])


            # Calculating the next loss. This step could be optimize by avoiding to compute the loss for the directions already found
            loss_next[serching_step]=compute_loss_input(n_next,Mn_next,serching_step)
            # Checking whether the Armijo criteria is satisfied
            # maybe we should look into armijo for another criteria, taking more care whether it is the full loss or just the last aspect that we care about
            serching_step[(loss-loss_next)>1e-5*alpha.squeeze()*norm_grad_hidden]=False

            # For the elements in the batch where it is not, ie, serching_step=True, update the step
            alpha[serching_step]*=0.3

        # 1.3: Updating the value of n[0]
        n[0].grad.zero_()
        n[0]=n_next[0].clone()
        n[0]=x+torch.clamp(n[0] - x,-eps_input,eps_input)
        n[0]=torch.clamp(n[0],0.,1.)
        Mn[0]=Mn_next[0]

        # Part 2: Optimizing the hidden layers


        # 2.1: Computing Descent Direction for n[0]
        for i in range(1,len(n)):
            n[i].requires_grad_(True)
            Mn[i]=layers[i+1](n[i])

        loss=compute_loss_hidden(n,Mn)

        loss.backward(gradient=torch.ones_like(y_true, dtype=torch.float))

        for i in range(1,len(n)):
            n[i].requires_grad_(False)
            Mn[i]=layers[i+1](n[i]).detach_()


        norm_grad_hidden.zero_()
        for i in range(1,len(n)):
            norm_grad_hidden+=batch_dot_prod(n[i].grad,n[i].grad)
        if iter == 1:
            norm_grad_hidden_init=norm_grad_hidden.clone()



        # 2.2: Computing optimal step size using Armijo Rule
        alpha[:]=mu
        loss_next.zero_()
        for i in range(1,len(n)):
            Mgrad[i]=layers[i+1](n[i].grad)
            Mn_next[i].zero_()

        serching_step=torch.tensor([True]*x.size(0))
        while torch.any(serching_step) and alpha.min()>=mu*1e-15:
            # Update the new value of the neurons with the step direction alpha.


            for i in range(1,len(n)):
                n_next[i][serching_step]=n[i][serching_step]-n[i].grad.view(x.size(0),-1)[serching_step].mul(alpha[serching_step]).view_as(n[i][serching_step])
                Mn_next[i][serching_step]=Mn[i][serching_step]-Mgrad[i].view(x.size(0),-1)[serching_step].mul(alpha[serching_step]).view_as(Mn[i][serching_step])


            # Calculating the next loss. This step could be optimize by avoiding to compute the loss for the directions already found
            loss_next[serching_step]=compute_loss_hidden(n_next,Mn_next,serching_step)
            # Checking whether the Armijo criteria is satisfied
            # maybe we should look into armijo for another criteria, taking more care whether it is the full loss or just the last aspect that we care about
            serching_step[(loss-loss_next)>1e-5*alpha.squeeze()*norm_grad_hidden]=False

            # For the elements in the batch where it is not, ie, serching_step=True, update the step
            alpha[serching_step]*=0.3

        # 2.3: Updating the value of n[1...]
        for i in range(1,len(n)):
            n[i].grad.zero_()
            n[i]=n_next[i].clone()
            Mn[i]=Mn_next[i]


        norm_ratio=torch.cat([(norm_grad_input/norm_grad_input_init).unsqueeze(1),(norm_grad_hidden/norm_grad_hidden_init).unsqueeze(1)],dim=1).max(1)[0]
        if iter>=10:
            if norm_ratio.max()<tol_grad:
                solver='solved'
            elif iter>=40 and norm_ratio.median()<tol_grad:
                solver='partially solved'
            elif torch.all(torch.isnan(loss_next)):
                solver='hitting all constraints'
            # elif norm_grad_prev.max()<norm_grad.max():
                # solver='failed'

        if debug:
            print("Iter:", iter, ", Loss max: ", loss.max().data.cpu().numpy(), ", Loss min:", loss.min().data.cpu().numpy(), ", Output Loss", criterion(Mn_next[3], y_true).max().data.cpu().numpy())
            # print("      Grad n[0]: ", n[0].grad.abs().mean().data.cpu().numpy()," Grad n[1]: ", n[1].grad.abs().mean().data.cpu().numpy()," Grad n[2]: ", n[2].grad.abs().mean().data.cpu().numpy()," Grad n[3]: ", n[3].grad.abs().mean().data.cpu().numpy())
            print("      norm_grad:", (norm_ratio).max().data.cpu().numpy(), "alpha: ", alpha.min().data.cpu().numpy())



    if torch.any(torch.isnan(loss)) or torch.any(torch.isnan(n[0])) or torch.any(torch.isnan(n[1])) or torch.any(torch.isnan(n[2])):
        raise ValueError('Diverged')

    with torch.no_grad():

        d[0]=x-n[0]
        d[1]=n[1]-F.max_pool2d(F.leaky_relu(Mn[0]), (2, 2))
        d[2]=n[2]-F.max_pool2d(F.leaky_relu(Mn[1]), (2, 2)).view(x.size(0), -1)
        d[3]=n[3]-F.leaky_relu(Mn[2])



        if debug:
            print(" d[0]: ", d[0].abs().max().data.cpu().numpy() ," d[1]: ", d[1].abs().max().data.cpu().numpy()," d[2]: ", d[2].abs().max().data.cpu().numpy()," d[3]: ", d[3].abs().max().data.cpu().numpy())
            print(solver)
            # breakpoint()
            # time.sleep(0.000001)

        for i in range(len(n)):
            model.d[i] = d[i].view_as(n[i])

        if False and solver!="solved":
            print(solver)
            print("Iter:", iter, ", Loss max: ", loss.max().data.cpu().numpy(), ", Loss min:", loss.min().data.cpu().numpy(), ", Output Loss", criterion(Mn_next[3], y_true).max().data.cpu().numpy())
            print(" d[0]: ", d[0].abs().max().data.cpu().numpy() ," d[1]: ", d[1].abs().max().data.cpu().numpy()," d[2]: ", d[2].abs().max().data.cpu().numpy()," d[3]: ", d[3].abs().max().data.cpu().numpy())
            # breakpoint()

    for p in model.parameters():
        p.requires_grad = True



def DistortNeuronsGradientDescent(model, x, y_true, lamb, mu, optimizer=None):
    '''
    Descriptions:
        Gradient Descent
        No line search
        Projection of n[0] into correct set (abs(n[0]-x)<0.3, 0<=n[0]<=1)
    '''
    model.eval()
    num_iters = 10
    eps_input = 0.3 # value to clamp input disturbance
    eps_layers = 10 # value to clamp layers disturbance
    device = model.parameters().__next__().device
    criterion = nn.CrossEntropyLoss(reduction="none")

    for p in model.parameters():
        p.requires_grad = False

    layers = list(model.children())

    # Initializing the Nodes
    n = [None] * (len(model.n)-1)
    for i in range(len(n)):
        n[i] = torch.empty(model.n[i].size()).to(device)

    # Initializing the auxiliary variables
    aux = [None] * (len(model.n)-2)
    for i in range(len(aux)):
        aux[i] = torch.empty(model.n[i+1].size()).to(device)

    # Initializing the SGD momentum
    momentum = [None] * (len(model.n)-1)
    for i in range(len(n)):
        momentum[i] = torch.zeros(model.n[i].size()).to(device)

    with torch.no_grad():
        x=layers[0](x)

    # Initializing the value of the nodes

    with torch.no_grad():
        n[0]=x+eps_input*(2*torch.rand(x.size())-1).to(device)
    # verify device and verify requires_grad
    n[0].requires_grad_(True)
    aux[0]=F.max_pool2d(F.leaky_relu(layers[1](n[0])), (2, 2))
    with torch.no_grad():
        n[1]=aux[0]+eps_layers*(2*torch.rand(model.n[1].size())-1).to(device)
    n[1].requires_grad_(True)
    aux[1]=F.max_pool2d(F.leaky_relu(layers[2](n[1])), (2, 2)).view(x.size(0), -1)
    with torch.no_grad():
        n[2]=aux[1]+eps_layers*(2*torch.rand(model.n[2].size())-1).to(device)
    n[2].requires_grad_(True)
    aux[2]=F.leaky_relu(layers[3](n[2]))
    with torch.no_grad():
        n[3]=aux[2]+eps_layers*(2*torch.rand(model.n[3].size())-1).to(device)
    n[3].requires_grad_(True)


    def rho(z):
        # return torch.sum(z**2,1)
        return torch.norm(z,p=1,dim=1)

    # The idea of the next following steps is to partially normalize the gradients
    # such that mu is more a less constant in all epochs and batches. I will be calculate on the first batch
    norm_grad_init_inv = [None] * (len(model.n)-1)


    iter=0
    norm_grad=float("inf")
    while iter<= num_iters: #and norm_grad<=0.001:
        iter+=1

        # Calculating the loss
        loss = lamb * rho((n[0] - x).view(x.size(0), -1))
        loss += lamb * rho((n[1] - aux[0]).view(x.size(0), -1))
        loss += lamb * rho(n[2] - aux[1])
        loss += lamb * rho(n[3] - aux[2])
        loss += -criterion(layers[-1](n[3]), y_true)


        if torch.any(torch.isnan(loss)) or torch.any(torch.isnan(n[0])) or torch.any(torch.isnan(n[1])) or torch.any(torch.isnan(n[2])):
            breakpoint()
        if torch.any(loss.abs()>1e5) or torch.any(n[0].abs()>1e3) or torch.any(n[1].abs()>1e3) or torch.any(n[2].abs()>1e3) or torch.any(n[3].abs()>1e3):
            breakpoint()

        loss.backward(gradient=torch.ones_like(y_true, dtype=torch.float))


        norm_grad=float("-inf")
        with torch.no_grad():
            if iter==1:
                for i in range(len(n)):
                    norm_grad_init_inv[i] = 1/torch.norm(n[i].grad)
            for i in range(len(n)):
                norm_grad=max(norm_grad,torch.norm(norm_grad_init_inv[i]*n[i].grad))
                momentum[i].mul_(0.01).add_(-mu*n[i].grad)
                n[i].add_(momentum[i])
                n[i].grad.zero_()


        # Simulating the system
        with torch.no_grad():
            n[0]=x+torch.clamp(n[0] - x,-eps_input,eps_input)
            n[0]=torch.clamp(n[0],-1,1)
        n[0].requires_grad_(True)
        aux[0]=F.max_pool2d(F.leaky_relu(layers[1](n[0])), (2, 2))
        with torch.no_grad():
            n[1]=aux[0]+torch.clamp(n[1] - aux[0],-eps_layers,eps_layers)
        n[1].requires_grad_(True)
        aux[1]=F.max_pool2d(F.leaky_relu(layers[2](n[1])), (2, 2)).view(x.size(0), -1)
        with torch.no_grad():
            n[2]=aux[1]+torch.clamp(n[2] - aux[1],-eps_layers,eps_layers)
        n[2].requires_grad_(True)
        aux[2]=F.leaky_relu(layers[3](n[2]))
        with torch.no_grad():
            n[3]=aux[2]+torch.clamp(n[3] - aux[2],-eps_layers,eps_layers)
        n[3].requires_grad_(True)


    with torch.no_grad():
        # print("Norm grad: ",norm_grad, " Loss: ", torch.max(loss))
        model.d[0] = n[0] - x
        model.d[1] = n[1] - aux[0]
        model.d[2] = n[2] - aux[1]
        model.d[3] = n[3] - aux[2]

    for p in model.parameters():
        p.requires_grad = True



