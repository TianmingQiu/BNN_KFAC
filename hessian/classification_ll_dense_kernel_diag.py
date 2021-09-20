from re import X
import sys
import os
from utils import calculateDominance
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

from numpy.core.function_base import add_newdoc
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

# Standard imports
import numpy as np
from tqdm import tqdm
import seaborn as sns
from matplotlib import pyplot as plt
from PIL import Image, ImageOps  

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# From the repository
from models.wrapper import BaseNet
from models.curvatures import BlockDiagonal, KFAC, EFB, INF
from models.utilities import calibration_curve
from models import plot

def get_near_psd(A, epsilon):
    C = (A + A.T)/2
    eigval, eigvec = torch.linalg.eig(C.to(torch.double))
    eigval[eigval.real < epsilon] = epsilon
    return eigvec @ (torch.diag(eigval)) @ eigvec.t()


def gradient(y, x, grad_outputs=None):
    """Compute dy/dx @ grad_outputs"""
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs = grad_outputs, create_graph=True, retain_graph=True, allow_unused=True)[0]
    return grad

def jacobian(y, x, device):
    '''
    Compute dy/dx = dy/dx @ grad_outputs; 
    y: output, batch_size * class_number
    x: parameter
    '''
    jac = torch.zeros(y.shape[1], torch.flatten(x).shape[0]).to(device)
    for i in range(y.shape[1]):
        grad_outputs = torch.zeros_like(y)
        grad_outputs[:,i] = 1
        jac[i,:] = torch.flatten(gradient(y, x, grad_outputs))
    return jac

def plot_tensors(tensor):
    if not tensor.ndim == 2:
        raise Exception("assumes a 2D tensor")
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1,1,1)
    ax.imshow(tensor.cpu().numpy())
    ax.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])   

device = "cuda" if torch.cuda.is_available() else "cpu"
parent = os.path.dirname(current)
path = parent + "/data"
# load and normalize MNIST
new_mirror = 'https://ossci-datasets.s3.amazonaws.com/mnist'
datasets.MNIST.resources = [
    ('/'.join([new_mirror, url.split('/')[-1]]), md5)
    for url, md5 in datasets.MNIST.resources
]

train_set = datasets.MNIST(root=path,
                                        train=True,
                                        transform=transforms.ToTensor(),
                                        download=True)
train_loader = DataLoader(train_set, batch_size=32)

# And some for evaluating/testing
test_set = datasets.MNIST(root=path,
                                        train=False,
                                        transform=transforms.ToTensor(),
                                        download=True)
test_loader = DataLoader(test_set, batch_size=1)

# Train the model
net = BaseNet(lr=1e-3, epoch=3, batch_size=32, device=device)
#net.load(models_dir + '/theta_best.dat')
criterion = nn.CrossEntropyLoss().to(device)
net.train(train_loader, criterion)
sgd_predictions, sgd_labels = net.eval(test_loader)

print(f"MAP Accuracy: {100 * np.mean(np.argmax(sgd_predictions.cpu().numpy(), axis=1) == sgd_labels.numpy()):.2f}%")



# update likelihood FIM
H = None
for images, labels in tqdm(test_loader):
    logits = net.model(images.to(device))
    dist = torch.distributions.Categorical(logits=logits)
    # A rank-1 Kronecker factored FiM approximation.
    labels = dist.sample()
    loss = criterion(logits, labels)
    net.model.zero_grad()
    loss.backward()
            
    grads = []
    for layer in list(net.model.modules())[1:]:
        for p in layer.parameters():    
            J_p = torch.flatten(p.grad.view(-1)).unsqueeze(0)
            grads.append(J_p)
    J_loss = torch.cat(grads, dim=1)
    H_loss = J_loss.t() @ J_loss
    H_loss.requires_grad = False
    H = H_loss if H == None else H + H_loss

H = H/len(test_loader)    

calculateDominance(H)