from re import X
import sys
import os
import time
import datetime
import itertools

from numpy.core.function_base import add_newdoc
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(os.path.dirname(current))
sys.path.append(parent)

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import init
import torch.utils.data as Data
from torch import nn, optim
from torch.utils.data import random_split, DataLoader, RandomSampler
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import hamiltorch
import tqdm

# file path
data_path = parent + "/data/"
model_path = parent + "/theta/"
result_path = parent + "/results/Regression/"


def load_agw_1d(x, y, get_feats=False):
    def features(x):
        return np.hstack([x[:, None] / 2.0, (x[:, None] / 2.0) ** 2])
    f = features(x)
    '''
    x_means, x_stds = x.mean(axis=0), x.std(axis=0)
    y_means, y_stds = y.mean(axis=0), y.std(axis=0)
    f_means, f_stds = f.mean(axis=0), f.std(axis=0)

    X = ((x - x_means) / x_stds).astype(np.float32)
    Y = ((y - y_means) / y_stds).astype(np.float32)
    F = ((f - f_means) / f_stds).astype(np.float32)
    '''

    X = x.astype(np.float32)
    Y = y.astype(np.float32)
    F = f.astype(np.float32)

    if get_feats:
        return F, Y

    return X, Y

class Net(torch.nn.Module):
    def __init__(self, input_dim, output_dim, n_hid):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, n_hid)   # hidden layer
        self.fc2 = torch.nn.Linear(n_hid, n_hid)   # hidden layer
        self.fc3 = torch.nn.Linear(n_hid, output_dim)   # output layer

    def forward(self, x):
        x = F.silu(self.fc1(x))  # activation function for hidden layer
        x = F.silu(self.fc2(x)) 
        x = self.fc3(x)  # linear output
        return x

    def weight_init_gaussian(self, std):
        for layer in self.modules():   
            if layer.__class__.__name__ in ['Linear', 'Conv2d']:
                init.normal_(layer.weight, 0, std)
                # bias.data should be 0
                layer.bias.data.fill_(0)
            elif layer.__class__.__name__ == 'MultiheadAttention':
                raise NotImplementedError

    def weight_init_uniform(self, lim):
        for layer in self.modules():   
            if layer.__class__.__name__ in ['Linear', 'Conv2d']:
                init.uniform_(layer.weight, -lim, lim)
                # bias.data should be 0
                layer.bias.data.fill_(0)
            elif layer.__class__.__name__ == 'MultiheadAttention':
                raise NotImplementedError

def get_nb_parameters(model):
    print('Total params: %.2f' % (np.sum(p.numel() for p in model.parameters())))


## code begins here

N = 30
sigma = 3
lim = 0.2
x = np.sort(np.random.uniform(-4,4,N))
y = np.power(x,3) + sigma * np.random.rand(x.size) # noisy y data (tensor), shape=(20, 1)

X, Y = load_agw_1d(x, y, get_feats=False)
X = torch.Tensor(X).view(-1,1)
Y = torch.Tensor(Y).view(-1,1)

net = Net(input_dim=1, output_dim=1, n_hid=30)     
net.weight_init_uniform(lim)
get_nb_parameters(net)

X_test = torch.linspace(-6,6,100).view(-1,1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

step_size = 0.0005
num_samples = 10000
L = 30
burn = -1
store_on_GPU = False
debug = False
model_loss = 'regression'
mass = 1.0

# Effect of tau
# Set to tau = 1000. to see a function that is less bendy (weights restricted to small bends)
# Set to tau = 1. for more flexible

tau = 1.0 # Prior Precision
tau_out = 110.4439498986428 # Output Precision
r = 0 # Random seed


tau_list = []
for w in net.parameters():
    tau_list.append(tau) # set the prior precision to be the same for each set of weights
tau_list = torch.tensor(tau_list).to(device)

# Set initial weights
params_init = hamiltorch.util.flatten(net).to(device).clone()
# Set the Inverse of the Mass matrix
inv_mass = torch.ones(params_init.shape) / mass

print(params_init.shape)
integrator = hamiltorch.Integrator.EXPLICIT
sampler = hamiltorch.Sampler.HMC

hamiltorch.set_random_seed(r)
params_hmc_f = hamiltorch.sample_model(net, X.to(device), Y.to(device), params_init=params_init,
                                       model_loss=model_loss, num_samples=num_samples,
                                       burn = burn, inv_mass=inv_mass.to(device),step_size=step_size,
                                       num_steps_per_sample=L,tau_out=tau_out, tau_list=tau_list,
                                       debug=debug, store_on_GPU=store_on_GPU,
                                       sampler = sampler)

# At the moment, params_hmc_f is on the CPU so we move to GPU

params_hmc_gpu = [ll.to(device) for ll in params_hmc_f[1:]]


# Let's predict over the entire test range [-2,2]
pred_list, log_probs_f = hamiltorch.predict_model(net, x = X_test.to(device),
                                                  y = X_test.to(device), samples=params_hmc_gpu,
                                                  model_loss=model_loss, tau_out=tau_out,
                                                  tau_list=tau_list)
# Let's evaluate the performance over the training data
pred_list_tr, log_probs_split_tr = hamiltorch.predict_model(net, x = X.to(device), y=Y.to(device),
                                                            samples=params_hmc_gpu, model_loss=model_loss,
                                                            tau_out=tau_out, tau_list=tau_list)
ll_full = torch.zeros(pred_list_tr.shape[0])
ll_full[0] = - 0.5 * tau_out * ((pred_list_tr[0].cpu() - Y) ** 2).sum(0)
for i in range(pred_list_tr.shape[0]):
    ll_full[i] = - 0.5 * tau_out * ((pred_list_tr[:i].mean(0).cpu() - Y) ** 2).sum(0)

fs = 16

m = pred_list[200:].mean(0).to('cpu')
s = pred_list[200:].std(0).to('cpu')
s_al = (pred_list[200:].var(0).to('cpu') + tau_out ** -1) ** 0.5

f, ax = plt.subplots(1, 1, figsize=(8, 4))

# Get upper and lower confidence bounds
lower_1, upper_1 = (m - s*2).flatten(), (m + s*2).flatten()
lower_2, upper_2 = (m - 2*s*2).flatten(), (m + 2*s*2).flatten()
lower_3, upper_3 = (m - 3*s*2).flatten(), (m + 3*s*2).flatten()

# view data
plt.figure(figsize=(6,5))
plt.fill_between(X_test.flatten().numpy(), lower_1.numpy(), upper_1.numpy(), color='burlywood', alpha=.6, label='+/- 1 std')
plt.fill_between(X_test.flatten().numpy(), lower_2.numpy(), upper_2.numpy(), color='burlywood', alpha=.5, label='+/- 2 std')
plt.fill_between(X_test.flatten().numpy(), lower_3.numpy(), upper_3.numpy(), color='burlywood', alpha=.4, label='+/- 3 std')
plt.plot(X_test.numpy(), np.power(X_test.numpy(),3), c='black', label='ground truth', linewidth = 2)
plt.plot(X_test.numpy(), m.numpy(), c='cornflowerblue', label='mean pred', linewidth = 2)
plt.scatter(X.numpy(), Y.numpy(), s=20, color = "black")
plt.xlabel('$x$', fontsize=15)
plt.ylabel('$y$', fontsize=15)
plt.legend()
plt.xlim([-6, 6])
#plt.ylim([-400, 400])
plt.gca().yaxis.grid(alpha=0.3)
plt.gca().xaxis.grid(alpha=0.3)
plt.tick_params(labelsize=10) 
plt.savefig(result_path+'hmc.png', format='png', bbox_inches = 'tight')
#plt.savefig(result_path+'hmc.eps', format='eps', bbox_inches = 'tight')


