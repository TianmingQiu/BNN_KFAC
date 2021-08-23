from layers.functions.detection import Detect
from data.kitti import KittiDetection
from models.utilities import calibration_curve
from numpy.core.fromnumeric import diagonal
from tqdm.std import tqdm
from models.curvatures import BlockDiagonal, EFB, INF, KFAC
from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
# DEVICE = torch.device('cuda:7')
DEVICE_LIST = [0,1,2,3]

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
args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

# viz = visdom.Visdom()

def torch_kron(A, B):
    return torch.einsum("ab,cd->acbd", A, B).view(A.size(0)*B.size(0),  A.size(1)*B.size(1))

def torch_block_diag(*arrs):
    bad_args = [k for k in range(len(arrs)) if not (isinstance(arrs[k], torch.Tensor) and arrs[k].ndim == 2)]
    if bad_args:
        raise ValueError("arguments in the following positions must be 2-dimension tensor: %s" % bad_args)
    shapes = torch.tensor([a.shape for a in arrs])
    i = []
    v = []
    r, c = 0, 0
    for k, (rr, cc) in enumerate(shapes):
        first_index = torch.arange(r, r + rr, device=arrs[0].device)
        second_index = torch.arange(c, c + cc, device=arrs[0].device)
        index = torch.stack((first_index.tile((cc,1)).transpose(0,1).flatten(), second_index.repeat(rr)), dim=0)
        i += [index]
        v += [arrs[k].flatten()]
        r += rr
        c += cc
    out_shape = torch.sum(shapes, dim=0).tolist()

    if arrs[0].device == "cpu":
        out = torch.sparse.DoubleTensor(torch.cat(i, dim=1), torch.cat(v), out_shape)
    else:
        out = torch.cuda.sparse.DoubleTensor(torch.cat(i, dim=1).to(arrs[0].device), torch.cat(v), out_shape)
    return out

def train_kfac(continue_flag):
    if args.dataset == 'COCO':
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = COCO_ROOT
        cfg = coco
        dataset = COCODetection(root=args.dataset_root,
                                transform=SSDAugmentation(cfg['min_dim'],
                                                          MEANS))
    elif args.dataset == 'VOC':
        if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')
        cfg = voc
        dataset = VOCDetection(root=args.dataset_root,
                               image_sets=[('2007', 'trainval')],
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS))


    elif args.dataset == 'KITTI':
        # if args.dataset_root == COCO_ROOT:
        #     parser.error('Must specify dataset if specifying dataset_root')

        cfg = kitti_config
        dataset = KittiDetection(root='data/kitti/train.txt',
                                 transform=SSDAugmentation(cfg['min_dim'],
                                                          MEANS))
    

    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])            # initialize SSD
    ssd_net.load_state_dict(torch.load(args.resume))
    # ssd_net.cuda()
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


    # compute KFAC Fisher Information Matrix
    kfac = KFAC(net)
    # kfac = nn.DataParallel(kfac)

    for iteration in range(args.start_iter, 10):

        images, targets = next(batch_iterator)
        images = Variable(images.cuda())
        targets = [Variable(ann.cuda(), volatile=True) for ann in targets]

        out = net(images)
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()

        kfac.update(batch_size=images.size(0))

    # inversion and sampling
    estimator = kfac

    # TODO: Parameter Adjustment
    add = 2
    multiply = 100 

    estimator.invert(add, multiply)

    mean_predictions = 0
    samples = 10  # 10 Monte Carlo samples from the weight posterior.
    data_loader = data.DataLoader(dataset, 1,
            num_workers=args.num_workers,
            shuffle=True, collate_fn=detection_collate,
            pin_memory=True)
    batch_iterator = iter(data_loader)

    H = np.array([])
    for i,layer in enumerate(list(net.modules())[2:]):
        if layer in estimator.state:
            Q_i = estimator.inv_state[layer][0].cpu()
            H_i = estimator.inv_state[layer][1].cpu()

            H = np.kron(Q_i,H_i) if H.any() else block_diag(H,np.kron(Q_i,H_i))

    def gradient(y, x, grad_outputs=None):
        """Compute dy/dx @ grad_outputs"""
        if grad_outputs is None:
            grad_outputs = torch.ones_like(y)
        grad = torch.autograd.grad(y, [x], grad_outputs = grad_outputs, create_graph=True, retain_graph=True, allow_unused=True)[0]
        return grad

    sigma = 0
    for iteration in range(1):
        image, target = next(batch_iterator)
        g = []
        pred_j = net(images)
        for p in net.parameters():
            g.append(torch.flatten(gradient(pred_j,p)))
        J = torch.cat(g,dim=0).unsqueeze(0)
        std = (J @ H @ J.t())**0.5 + sigma

    pred_mean = pred_j.data.numpy().squeeze(1)
    pred_std = np.array(std, dtype=float)
        




if __name__ == '__main__':
    train_kfac(continue_flag = True)
