from re import X
import sys
import os

from numpy.core.function_base import add_newdoc
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(os.path.dirname(current))
sys.path.append(parent)


# From the repository
from models.curvatures import BlockDiagonal, Diagonal, KFAC, EFB, INF
from models.utilities import calibration_curve
from models import plot


import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import init
import torch.utils.data as Data

import matplotlib.pyplot as plt
import numpy as np
import hamiltorch

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

# file path
data_path = parent + "/data/"
model_path = parent + "/theta/"
result_path = parent + "/results/Regression/"

torch.manual_seed(2)    # reproducible
device = 'cpu'
# initialize data
lim = 0.2
N = 80
sigma = 3

x = torch.FloatTensor(N, 1).uniform_(-4, 4).sort(dim=0).values # random x data (tensor), shape=(20, 1)
y = x.pow(3) + sigma * torch.rand(x.size()) # noisy y data (tensor), shape=(20, 1)

X = torch.Tensor(x).view(-1,1)
Y = torch.Tensor(y).view(-1,1)

x_ = torch.linspace(-6,6,500)
y_ = x_.pow(3)  
X_test = torch.Tensor(x_).view(-1,1)

# define the network
net = Net(input_dim=1, output_dim=1, n_hid=4)     
net.weight_init_uniform(lim)
get_nb_parameters(net)

step_size = 0.0005
num_samples = 1000
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
lower, upper = (m - s*2).flatten(), (m + s*2).flatten()
# + aleotoric
lower_al, upper_al = (m - s_al*2).flatten(), (m + s_al*2).flatten()

# Plot training data as black stars
ax.plot(X.numpy(), Y.numpy(), 'k*', rasterized=True)
# Plot predictive means as blue line
ax.plot(X_test.numpy(), m.numpy(), 'b', rasterized=True)
# Shade between the lower and upper confidence bounds
ax.fill_between(X_test.flatten().numpy(), lower.numpy(), upper.numpy(), alpha=0.5, rasterized=True)
ax.fill_between(X_test.flatten().numpy(), lower_al.numpy(), upper_al.numpy(), alpha=0.2, rasterized=True)
ax.set_ylim([-2, 2])
ax.set_xlim([-2, 2])
plt.grid()
ax.legend(['Observed Data', 'Mean', 'Epistemic', 'Aleatoric'], fontsize = fs)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.tick_params(axis='both', which='minor', labelsize=14)

bbox = {'facecolor': 'white', 'alpha': 0.8, 'pad': 1, 'boxstyle': 'round', 'edgecolor':'black'}
plt.text(1., -1.5, 'Acceptance Rate: 58 %', bbox=bbox, fontsize=16, horizontalalignment='center')


plt.tight_layout()
# plt.savefig('plots/full_hmc.pdf', rasterized=True)
    
plt.show()