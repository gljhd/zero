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
    imgs, labels = train_pipe.data_gene()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay = 0.0005)

    for epoch in range(epochs):
        print('starting epoch {}/{}.'.format(epoch+1, epochs))





