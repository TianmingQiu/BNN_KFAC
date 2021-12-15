from utils.classification_ll import gradient
from layers.functions.detection import Detect
from data.kitti import KittiDetection
from models.utilities import calibration_curve
from numpy.core.fromnumeric import diagonal
from models.curvatures import BlockDiagonal, EFB, INF, KFAC, Diagonal
from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# DEVICE = torch.device('cuda:7')
DEVICE_LIST = [0]

import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
from scipy.linalg import block_diag
import tqdm
import pickle
from matplotlib import pyplot as plt
from data import KITTI_CLASSES as labels
import random

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

PATH_TO_WEIGHTS = 'weights/KITTI_30000.pth'

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='KITTI', choices=['VOC', 'COCO', 'KITTI'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default=COCO_ROOT, # osp.join('data/coco/') OR osp.join("data/VOCdevkit/")
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=4, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default= PATH_TO_WEIGHTS, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
try:
    args = parser.parse_args()
except:
    class ARGS:
        batch_size = 4
        resume = PATH_TO_WEIGHTS
        start_iter = 0
        num_workers = 4
        cuda = True
        lr = 1e-4
        momentum = 0.9
        weight_decay = 5e-4
        gamma = 0.1
        save_folder = 'weights/'
    args = ARGS()

torch.set_default_tensor_type('torch.cuda.FloatTensor')

def add_gaussian_noise(image,std,mean=0):
    image = image + np.random.normal(mean,std,size=image.shape)
    image[image<0] = 0
    image[image>255] = 255
    return image.astype('uint8')

def add_salt_and_pepper(image,scale):
    # (375, 1242, 3)
    scale /= 1000
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if random.random() < scale:
                # salt and pepper
                image[i,j,:] = random.randint(0,1) * 255
    return image.astype('uint8')

def crop_image(image, coords):
    for a,b,c,d in coords:
        image[a:b,c:d] = 0
    return image

def blur_image(image, coords,dividend = 2.0):
    for a,b,c,d in coords:
        x = image[a:b,c:d]
        x = cv2.resize(x,(int((d-c)/dividend), int((b-a)/dividend)) )
        x = cv2.resize(x,( d-c , b-a) ).astype('uint8')
        image[a:b,c:d] = x
    return image

cfg = kitti_config
dataset = KittiDetection(root='data/kitti/train.txt',
                            transform=SSDAugmentation(cfg['min_dim'],
                                                    MEANS))


ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])            # initialize SSD
try: weight_path = args.resume
except: weight_path = args['resume']
ssd_net.load_weights(weight_path)
ssd_net.cuda()
net = ssd_net

if args.cuda and torch.cuda.is_available():
    # speed up using multiple GPUs
    # net = torch.nn.DataParallel(ssd_net,device_ids=DEVICE_LIST)
    cudnn.benchmark = True
    net.cuda()

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay)
data_loader = data.DataLoader(dataset, args.batch_size,
                                num_workers=args.num_workers,
                                shuffle=True, collate_fn=detection_collate,
                                pin_memory=True)

# create batch iterator
batch_iterator = iter(data_loader)
criterion = MultiBoxLoss(cfg['num_classes'], 0.3, True, 0, True, 3, 0.5,
                            False, args.cuda)
colors = plt.cm.hsv(np.linspace(0, 1, 9)).tolist()

kfac_direc = 'weights/diag_full.obj'
# kfac_direc = None


if kfac_direc:
    print('Loading kfac...')
    filehandler = open(kfac_direc, 'rb')
    diag = pickle.load(filehandler)
    # diag.invert(add=0.1, multiply=1)
    print('Finished!')
else:
    # compute KFAC Fisher Information Matrix
    diag = Diagonal(net)

    for _ in range(args.start_iter, 1871):

        images, targets = next(batch_iterator)
        images = Variable(images.cuda())
        targets = [Variable(ann.cuda(), volatile=True) for ann in targets]

        out = net(images)
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()

        diag.update(batch_size=images.size(0))
        if (_ + 1) % 100 == 0:
            print('Updating iter: ', _ + 1)

    # inversion and sampling
    estimator = diag

    # estimator.invert(add=1, multiply=2)

    # saving kfac
    file_pi = open('weights/diag_full_uninverted.obj', 'wb')
    pickle.dump(estimator, file_pi)

def eval_unvertainty_diag(model, x, H, diag):
    threshold = 0.5
    x = Variable(x.cuda(), requires_grad=True)
    model.softmax = nn.Softmax(dim=-1)
    model.detect = Detect()
    model.phase = 'test'
    model.cuda()

    detections = model.forward(x)
    out = torch.Tensor([[0,0,0,0,0,0]])
    for i in range(detections.size(1)):
        for j in range(detections.size(2) - 1):
            if detections[0,i,j,0] >= threshold:
                # out.append(torch.cat((torch.Tensor([i]), detections[0,i,j,:])))
                out = torch.cat(( out , torch.cat((torch.Tensor([i]), detections[0,i,j,:])).unsqueeze(dim=0) ))

    # For car showing purposes
    # out = torch.cat(( out , torch.cat((torch.Tensor([1]), detections[0,0,1,:])).unsqueeze(dim=0) ))

    # DETECTIONS: SCORE(1), LOC(4)
    # OUT       : CLASS(1), SCORE(1), LOC(4)
    uncertainties = []
    for _ in range(1,out.size(0)):
        uncertainty = []
        for i in range(1,2):

            # retaining graph for multiple backward propagations
            out[_,i].backward(retain_graph = True)
            # left, up, right, down

            # Loading all gradients from layers
            all_grad = torch.Tensor()
            for layer in model.modules():
                if layer.__class__.__name__ in diag.layer_types:
                    if layer.__class__.__name__ in ['Linear', 'Conv2d']:
                        grads = layer.weight.grad.contiguous().view(layer.weight.grad.shape[0], -1)
                        if layer.bias is not None:
                            grads = torch.cat([grads, layer.bias.grad.unsqueeze(dim=1)], dim=1)
                        all_grad = torch.cat((all_grad, torch.flatten(grads)))
                        model.zero_grad()

            J = all_grad.unsqueeze(0)
            pred_std = torch.abs(J * H * J).sum()
            del J, all_grad
            uncertainty.append(pred_std)
        uncertainties.append(torch.FloatTensor(uncertainty))
    uncertainties = torch.stack(uncertainties) if uncertainties else torch.tensor([])

    return out[1:], uncertainties

num_iterations = 20
grand_repo = None
for iteration in tqdm.tqdm(range(num_iterations)):
    testset = KittiDetection(root='data/kitti/train.txt')
    img_id = iteration
    image = testset.pull_image(img_id)
    # img_name = '/root/Documents/BNN_KFAC/data/kitti/testing/image_2/000402.png'
    # image = cv2.imread(img_name, cv2.IMREAD_COLOR)

    repo = None
    noise_level = 15
    for noise in range(noise_level):
    # INTRODUCE NOISE
        image = add_gaussian_noise(image,noise)

        # CROP AND BLUR IMAGE
        # [637.70557, 167.12257, 783.7215 , 230.99509]
        # image = crop_image(image,[[167,230,637,680]])
        # image = blur_image(image,[[167,230,637,783]],4)

        # RESIZE
        x = cv2.resize(image, (300, 300)).astype(np.float32)
        x -= (104.0, 117.0, 123.0)
        x = x.astype(np.float32)
        x = x[:, :, ::-1].copy()
        # plt.imshow(x)
        # plt.axis('off')
        # plt.show()
        x = torch.from_numpy(x).permute(2, 0, 1)
        xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
        if torch.cuda.is_available():
            xx = xx.cuda()

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h = []
        for i,layer in enumerate(list(diag.model.modules())[1:]):
            if layer in diag.state:
                H_i = diag.inv_state[layer]
                h.append(torch.flatten(H_i))
        H = torch.cat(h, dim=0)


        mean_predictions, uncertainty = eval_unvertainty_diag(net, xx, H, diag)
        mean_predictions = mean_predictions.detach()
        uncertainty = uncertainty ** 0.5
        
        if repo == None: repo = uncertainty.cuda()
        else:
            if uncertainty.shape[0] < repo.shape[0]:
                uncertainty = torch.cat((uncertainty.cuda(),torch.zeros(repo.shape[0] - uncertainty.shape[0],1,device='cuda')),dim=0)

            repo = torch.cat((repo,uncertainty[:repo.shape[0]].cuda()),dim=1)


    if repo == None or repo.numel() == 0: continue
    dimension = (repo > 1e-10).all(dim = 1)
    dimension =  dimension.unsqueeze(1).repeat(1,noise_level)
    repo = repo[dimension].view(-1,noise_level)
    if grand_repo == None: grand_repo = repo
    else:
        grand_repo = torch.cat((grand_repo,repo),dim=0)

# (23,5)
# [0.1736, 0.1787, 0.1774, 0.1905, 0.1985, 0.2076, 0.2243, 0.2490, 0.2835,
        # 0.2787, 0.3369, 0.3764, 0.4215, 0.4964, 0.5643]) 200,15

# plt.plot(xrange,grand_repo.T.cpu(),alpha=0.2)
# plt.plot(xrange,grand_repo.mean(dim=0).cpu(),color='k')

grand_repo = grand_repo.cpu()
xrange = [a/1000 for a in range(15)]
# xrange = [a/1 for a in range(15)]
sorted, _ = grand_repo.sort(dim=1)
plt.plot(xrange,sorted.T.cpu(),alpha=0.2)
plt.plot(xrange,sorted.mean(dim=0).cpu(),color='k')
plt.xlabel('Salt and pepper noise density')
plt.ylabel('Uncertainty')
plt.show()