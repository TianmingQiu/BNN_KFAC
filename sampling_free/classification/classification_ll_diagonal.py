from re import X
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

from numpy.core.function_base import add_newdoc
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(os.path.dirname(current))
sys.path.append(parent)

from inspect import getsourcefile
current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)
import utils

# Standard imports
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from PIL import Image, ImageOps  

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# From the repository
from models.curvatures import Diagonal, BlockDiagonal, KFAC, EFB, INF
from models.utilities import *
from models.plot import *
from models.wrapper import *

    
# file path
parent = os.path.dirname(os.path.dirname(current))
data_path = parent + "/data/"
model_path = parent + "/theta/"
result_path = parent + "/results/Hessian/"

# choose the device
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
if device == 'cuda':
    torch.cuda.manual_seed(42) 

# load and normalize MNIST
new_mirror = 'https://ossci-datasets.s3.amazonaws.com/mnist'
datasets.MNIST.resources = [
    ('/'.join([new_mirror, url.split('/')[-1]]), md5)
    for url, md5 in datasets.MNIST.resources
]

train_set = datasets.MNIST(root=data_path,
                                        train=True,
                                        transform=transforms.ToTensor(),
                                        download=True)
train_loader = DataLoader(train_set, batch_size=32)

# And some for evaluating/testing
test_set = datasets.MNIST(root=data_path,
                                        train=False,
                                        transform=transforms.ToTensor(),
                                        download=True)
test_loader = DataLoader(test_set, batch_size=256)


N = 200
std = 0.2

# Train the model
net = BaseNet_15k()
net.weight_init_uniform(std)
if device == 'cuda': 
    net.to(torch.device('cuda'))
get_nb_parameters(net)
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# train(net, device, train_loader, criterion, optimizer, epochs=10)
# save(net, model_path + 'BaseNet_750.dat')
load(net, model_path + 'BaseNet_15k.dat')

# run on the testset
sgd_predictions, sgd_labels = eval(net, device, test_loader)
acc = accuracy(sgd_predictions, sgd_labels)

# compute the diagonal FiM
diag = Diagonal(net)

for images, labels in tqdm(train_loader):
    logits = net(images.to(device))
    dist = torch.distributions.Categorical(logits=logits)
    # A rank-1 Kronecker factored FiM approximation.
    labels = dist.sample()
    loss = criterion(logits, labels)
    net.zero_grad()
    loss.backward()
    diag.update(batch_size=images.size(0))


# inversion of H
estimator = diag
estimator.invert(std**2, N)

h = []
for i,layer in enumerate(list(estimator.model.modules())[1:]):
    if layer in estimator.state:
        H_i = estimator.inv_state[layer]
        h.append(torch.flatten(H_i))
H = torch.cat(h, dim=0)
H_inv = torch.diag(H).cpu()

targets = torch.Tensor()
diag_prediction = torch.Tensor().to(device)
diag_entropy_lst = []

for images,labels in tqdm(test_loader):
    # prediction mean, equals to the MAP output 
    pred_mean = torch.nn.functional.softmax(net(images.to(device)) ,dim=1)        
    # compute prediction variance  
    grad_outputs = torch.zeros_like(pred_mean)
    idx  = np.argmax(pred_mean.cpu().detach().numpy(), axis=1)
    grad_outputs[:,idx] = 1
    g = []
    for p in net.parameters():    
        g.append(torch.flatten(utils.gradient(pred_mean, p, grad_outputs=grad_outputs)))
    J = torch.cat(g, dim=0).unsqueeze(0).cpu()
    pred_std = torch.abs(J * H_inv * J).sum().item()
    const = 2*np.e*np.pi 
    entropy = 0.5 * np.log2(const * pred_std)
    # ground truth
    targets = torch.cat([targets, labels])  
    # prediction, mean value of the gaussian distribution
    diag_prediction = torch.cat([diag_prediction, pred_mean]) 
    diag_entropy_lst.append(entropy)
diag_uncertainty = np.array(diag_entropy_lst)
print(f"Diagonal Accuracy: {100 * np.mean(np.argmax(diag_prediction.cpu().detach().numpy(), axis=1) == targets.numpy()):.2f}%")
print(f"Mean Diagonal Entropy: {diag_uncertainty.mean()}")
# kfac entropy: -0.64

res_entropy_lst = []
for i in tqdm(range(10000)):
    noise = torch.randn_like(images)
    pred_mean = torch.nn.functional.softmax(net(noise.to(device)) ,dim=1)        
    # compute prediction variance  
    grad_outputs = torch.zeros_like(pred_mean)
    idx  = np.argmax(pred_mean.cpu().detach().numpy(), axis=1)
    grad_outputs[:,idx] = 1
    g = []
    for p in net.parameters():    
        g.append(torch.flatten(gradient(pred_mean, p, grad_outputs=grad_outputs)))
    J = torch.cat(g, dim=0).unsqueeze(0)
    pred_std = torch.abs(J * H_inv * J).sum().item()
    del J
    const = 2*np.e*np.pi 
    entropy = 0.5 * np.log2(const * pred_std)
    res_entropy_lst.append(entropy) 
    res_uncertainty = np.array(res_entropy_lst)
print(f"Mean Noise Entropy: {res_uncertainty.mean()}")
# noise entropy: 2.86
