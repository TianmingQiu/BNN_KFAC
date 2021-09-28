from re import X
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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
from models.curvatures import BlockDiagonal, KFAC, EFB, INF
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

std = 0.2

# Train the model
net = BaseNet_750()
net.weight_init_gaussian(std)
if device == 'cuda': 
    net.to(torch.device('cuda'))
get_nb_parameters(net)
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
train(net, device, train_loader, criterion, optimizer, epochs=10)
# save(net, model_path + 'BaseNet_750.dat')
# load(net, model_path + 'BaseNet_750.dat')

# run on the testset
sgd_predictions, sgd_labels = eval(net, device, test_loader)
acc = accuracy(sgd_predictions, sgd_labels)

# update likelihood FIM
H = None
for images, labels in tqdm(train_loader):
    logits = net(images.to(device))
    dist = torch.distributions.Categorical(logits=logits)
    labels = dist.sample()
    loss = criterion(logits, labels)
    net.zero_grad()
    loss.backward()
    grads = []
    for layer in list(net.modules())[1:]:
        for p in layer.parameters():    
            J_p = torch.flatten(p.grad.view(-1)).unsqueeze(0)
            grads.append(J_p)
    J_loss = torch.cat(grads, dim=1)
    H_loss = J_loss.t() @ J_loss
    H_loss.requires_grad = False
    H = H_loss if H == None else H + H_loss
         
H = H.cpu()/len(train_loader) 

diag = torch.diag((std**2) * torch.ones(H.shape[0]))
H_inv = torch.pinverse(H + diag)

H_diag = torch.diag(H)
H_inv_diag = torch.diag(torch.reciprocal(H_diag + (std**2) * torch.ones(H.shape[0])))

mean_dense = torch.diag(H_inv).abs().sum().item()
mean_diag = H_inv_diag.abs().sum().item()
H_inv_diag_norm = H_inv_diag * mean_dense / mean_diag

min = H_inv.abs().min().item()
max = H_inv.abs().max().item()
scale = max - min 

image_inv = utils.tensor_to_image(H_inv.abs(), scale=scale)
image_inv.save(result_path+'images/H_inv_750_dense.png')

image_inv_diag = utils.tensor_to_image(H_inv_diag.abs(), scale=scale)
image_inv_diag.save(result_path+'images/H_inv_750_diag.png')

image_error = utils.tensor_to_image(torch.abs(H_inv-H_inv_diag_norm), scale=scale)
image_error.save(result_path+'images/error_750.png')

'''

torch.save(H, result_path+'tensor/H_dense_750.pt')
torch.save(H_inv, result_path+'tensor/H_inv_dense_750.pt')
torch.save(H_inv_diag, result_path+'tensor/H_inv_diag_750.pt')
torch.save(H_inv_diag_norm, result_path+'tensor/H_inv_diag_norm_750.pt')

H = torch.load(result_path+'tensor/H_dense_15k.pt')
H_inv = torch.load(result_path+'tensor/H_inv_dense_15k.pt')
H_inv_diag = torch.load(result_path+'tensor/H_inv_diag_15k.pt')
H_inv_diag_norm = torch.load(result_path+'tensor/H_inv_diag_norm_15k.pt')
'''