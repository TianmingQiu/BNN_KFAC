from re import X
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)


# From the repository
from models.wrapper import BaseNet
from models.curvatures import BlockDiagonal, Diagonal, KFAC, EFB, INF
from models.utilities import calibration_curve
from models import plot


import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data

import matplotlib.pyplot as plt
import numpy as np
import imageio


#torch.manual_seed()    # reproducible
x = torch.FloatTensor(30, 1).uniform_(-4, 4).sort(dim=0).values # random x data (tensor), shape=(30, 1)
y = x.pow(3) + 0.2*torch.rand(x.size())  # noisy y data (tensor), shape=(30, 1)

# torch can only train on Variable, so convert them to Variable
x, y = Variable(x), Variable(y)

# this is one way to define a network
class Net(torch.nn.Module):
    def __init__(self, input_dim, output_dim, n_hid):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, n_hid)   # hidden layer
        self.fc2 = torch.nn.Linear(n_hid, n_hid)   # hidden layer
        self.fc3 = torch.nn.Linear(n_hid, n_hid)   # hidden layer
        self.fc4 = torch.nn.Linear(n_hid, output_dim)   # output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))      # activation function for hidden layer
        x = F.relu(self.fc2(x)) 
        x = F.relu(self.fc3(x)) 
        x = self.fc4(x)             # linear output
        return x

net = Net(input_dim=1, output_dim=1, n_hid=10)     # define the network
optimizer = torch.optim.SGD(net.parameters(), lr=1e-3)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss


# train the network
for t in range(10000):
    prediction = net.forward(x)     # input x and predict based on x
    loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)
    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients  
    
kfac = KFAC(net)
prediction = net.forward(x)     # input x and predict based on x
loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)
optimizer.zero_grad()   # clear gradients for next train
loss.backward()         # backpropagation, compute gradients
kfac.update(batch_size=1)

estimator = kfac
add = 2
multiply = 100
estimator.invert(add, multiply)


x_ = torch.unsqueeze(torch.linspace(-6, 6), dim=1)  # x data (tensor), shape=(100, 1)
y_ = x_.pow(3)      
x_ = Variable(x_)
y_ = Variable(y_)


pred_lst = []
for i in range(100):
    estimator.sample_and_replace()
    pred_lst.append(net.forward(x_).data.numpy().squeeze(1))

pred = np.array(pred_lst).T
pred_mean = pred.mean(axis=1)
pred_std = pred.std(axis=1)


# view data
plt.figure(figsize=(10,10))
plt.scatter(x.data.numpy(), y.data.numpy(), s=80, color = "black")
plt.plot(x_.data.numpy(), pred_mean.T, c='royalblue', label='mean pred')
plt.fill_between(x_.data.numpy().squeeze(1), pred_mean.T - 3 * pred_std, pred_mean.T + 3 * pred_std,
                    color='cornflowerblue', alpha=.5, label='+/- 3 std')
plt.plot(x_.data.numpy(), y_.data.numpy(), c='grey', label='truth')
plt.legend()
plt.tight_layout()
plt.show()   