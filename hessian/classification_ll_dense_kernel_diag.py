from re import X
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

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
from models.wrapper import BaseNet
from models.curvatures import BlockDiagonal, KFAC, EFB, INF
from models.utilities import calibration_curve
from models import plot

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

utils.calculateDominance(H)
torch.cuda.empty_cache()