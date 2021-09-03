from layers.functions.detection import Detect
from data.kitti import KittiDetection
from models.utilities import calibration_curve
from numpy.core.fromnumeric import diagonal
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
import tqdm
import pickle

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
kfac_direc = 'weights/kfac.obj'

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
    ssd_net.load_weights(args.resume)
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

    if kfac_direc: 
        print('Loading kfac...')
        filehandler = open(kfac_direc, 'rb') 
        kfac = pickle.load(filehandler)
        print('Finished!')
    else:
        # compute KFAC Fisher Information Matrix
        kfac = KFAC(net)
        # kfac = nn.DataParallel(kfac)

        for _ in range(args.start_iter, 10):

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

        estimator.invert(add=1, multiply=2)

        # saving kfac
        file_pi = open('weights/kfac.obj', 'wb')
        pickle.dump(estimator, file_pi)


    mean_predictions = 0
    samples = 10  # 10 Monte Carlo samples from the weight posterior.
    data_loader = data.DataLoader(dataset, 1,
            num_workers=args.num_workers,
            shuffle=True, collate_fn=detection_collate,
            pin_memory=True)
    batch_iterator = iter(data_loader)

    def eval_fgsm_bnn(model,
                    data,
                    estimator,
                    samples=30,
                    device=torch.device('cuda')):

        model.eval()
        mean_predictions = 0

        samples = tqdm.tqdm(range(samples))
        for _ in samples:
            estimator.sample_and_replace()
            predictions, labels, scores = eval_fgsm(model, data)
            mean_predictions += predictions
        mean_predictions /= len(samples)

        return mean_predictions, labels

    def eval_fgsm(model, x, threshold = 0.2):
        x = x.cuda()
        model.softmax = nn.Softmax(dim=-1)
        model.detect = Detect()
        model.phase = 'test'

        y = model(x)
        detections = y
        out = []
        for i in range(detections.size(1)):
            for j in range(detections.size(2) - 1):
                if detections[0,i,j,0] >= threshold:
                    out.append(torch.cat((torch.Tensor([i]), detections[0,i,j,:])))

        out = torch.stack(out)
        return out[:,2:], out[:,0], out[:,1]


        
    for iteration in range(1):
        images, targets = next(batch_iterator)
        images = Variable(images.cuda())
        targets = [Variable(ann.cuda(), volatile=True) for ann in targets]

        mean_predictions, labels = eval_fgsm_bnn(net, images, kfac)
                




if __name__ == '__main__':
    train_kfac(continue_flag = True)
