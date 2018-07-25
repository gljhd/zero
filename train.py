import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from unet import Unet
from torch import optim
import data

def train_net(net,
              epochs=5,
              lr=0.1,
              batch_size=1):
    dir_img = 'data/train/'
    dir_mask = 'data/train_masks/'
    dir_checkpoint = 'checkpoints/'

    train_pipe = data.data_load(dir_img,dir_mask)

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay = 0.0005)

    for epoch in range(epochs):
        print('starting epoch {}/{}.'.format(epoch+1, epochs))
        imgs_labels = train_pipe.data_gene()
        epoch_loss = 0
        for imgs,labels in imgs_labels:
            imgs = torch.from_numpy(imgs)
            true_masks = torch.from_numpy(true_masks)
            mask_pred = net(imgs)
            loss = loss_cal(true_masks, true_masks)
            epoch_loss += loss.item()




