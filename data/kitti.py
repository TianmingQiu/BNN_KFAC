import glob
import random
import os
import numpy as np

import torch

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from skimage.transform import resize

import sys
import os.path as osp
import cv2


KITTI_CLASSES = (  # always index 0
'Car',
'Van',
'Truck',
'Pedestrian',
'Person_sitting',
'Cyclist',
'Tram',
'Misc')

KITTI_ROOT = osp.join("data/kitti")


class KittiDetection(Dataset):
    def __init__(self, root, img_size=300,transform=None):
        with open(root, 'r') as file:
            self.img_files = file.readlines()
        self.label_files = [path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt') for path in self.img_files]
        self.img_shape = (img_size, img_size)
        self.max_objects = 50
        self.transform = transform
        self.name = 'kitti'

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)
        return im, gt


    def pull_image(self, index):
        img_name = self.img_files[index].rstrip()
        return cv2.imread(img_name, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        label_path = self.label_files[index % len(self.img_files)].rstrip()
        labels = np.loadtxt(label_path).reshape(-1, 5)
        return label_path, labels


    def pull_item(self, index):
        #---------
        #  Image
        #---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()
        # img = np.array(Image.open(img_path))
        img = cv2.imread(img_path)

        # Handles images with less than three channels
        while len(img.shape) != 3:
            index += 1
            img_path = self.img_files[index % len(self.img_files)].rstrip()
            img = np.array(Image.open(img_path))

        h, w, _ = img.shape
        input_img = img
        # dim_diff = np.abs(h - w)
        # # Upper (left) and lower (right) padding
        # pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # # Determine padding
        # pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # # Add padding
        # input_img = np.pad(img, pad, 'constant', constant_values=128) / 255.
        padded_h, padded_w, _ = input_img.shape
        # # Resize and normalize
        # input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        # # Channels-first
        # input_img = np.transpose(input_img, (2, 0, 1))
        # # As pytorch tensor
        # input_img = torch.from_numpy(input_img).float()

        #---------
        #  Label
        #---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        labels = None
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 5)
            # Extract coordinates for unpadded + unscaled image
            x1 = w * (labels[:, 1] - labels[:, 3]/2)
            y1 = h * (labels[:, 2] - labels[:, 4]/2)
            x2 = w * (labels[:, 1] + labels[:, 3]/2)
            y2 = h * (labels[:, 2] + labels[:, 4]/2)
            # Adjust for added padding
            # x1 += pad[1][0]
            # y1 += pad[0][0]
            # x2 += pad[1][0]
            # y2 += pad[0][0]

            # ORIGIN:
            # # Calculate ratios from coordinates
            # labels[:, 1] = ((x1 + x2) / 2) / padded_w
            # labels[:, 2] = ((y1 + y2) / 2) / padded_h
            # labels[:, 3] *= w / padded_w
            # labels[:, 4] *= h / padded_h

            # CURRENT:
            # Calculate ratios from coordinates
            labels[:, 1] = x1 / padded_w
            labels[:, 2] = y1 / padded_h
            labels[:, 3] = x2 / padded_w
            labels[:, 4] = y2 / padded_h
        # Fill matrix
        filled_labels = np.zeros((self.max_objects, 5))
        if labels is not None:
            filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
        filled_labels = torch.from_numpy(filled_labels)
        target = [np.append(a[1:],a[0]) for a in labels]

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))


        # DEBUG:
        # ORIGIN:  [idx,x1+x2,y1+y2,w,h]        e.g. [7.        , 0.39713135, 0.50079371, 0.06983078, 0.06423046]
        # CURRENT: [xmin,xmax,ymin,ymax,idx]    e.g. [0.47359375000000004, 0.42920833333333336, 0.82246875, 0.6674166666666667, 59]
        # return img_path, input_img, filled_labels

        return input_img, target, h, w
        # +1 for background


    def __len__(self):
        return len(self.img_files)
