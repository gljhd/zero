import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import data
from Unet_model import Unet
import argparse

def train_net(net,
              epochs=5,
              lr=0.1,
              batch_size=1):

    file_path = '/media/wingspan/ssd512/Glj_train/VOCtrainval_2012/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt'
    dir_img = '/media/wingspan/ssd512/Glj_train/VOCtrainval_2012/VOCdevkit/VOC2012/JPEGImages'
    dir_mask = '/media/wingspan/ssd512/Glj_train/VOCtrainval_2012/VOCdevkit/VOC2012/SegmentationClass'
    dir_checkpoint = 'checkpoints/'

    train_pipe = data.data_load(file_path, dir_img, dir_mask)

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay = 0.0005)
    loss_cal = nn.CrossEntropyLoss();

    for epoch in range(epochs):
        print('starting epoch {}/{}.'.format(epoch+1, epochs))
        imgs_labels = train_pipe.data_gene()
        epoch_loss = 0

        for imgs,labels in imgs_labels:
            imgs = torch.from_numpy(imgs)
            true_masks = torch.LongTensor(labels)
            pred_masks = net(imgs)
            loss = loss_cal(pred_masks, true_masks)
            epoch_loss += loss.item()
            #print('{0:.4f} --- loss : {1:.6f}'.format(i*batch_size/N_train, loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Epoch finished ! Loss: {}'.format(epoch_loss))

        torch.save(net.state_dict(),dir_checkpoint+'CP{}.pth'.format(epoch+1))
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', '-e', default = 20, help = 'number of epoches')
    parser.add_argument('--lr', '-l', default = 0.0005, help = 'learning rate')
    parser.add_argument('--model', '-m', default = 'MODEL.pth', help = 'file model')
    args = parser.parse_args()
    return args


    args = parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    net = Unet(n_channels=3, n_classes=21)
    model_path ='MODEL.pth'
    net.load_state_dict(torch.load(model_path))
    try:
        train_net(net = net,
                  epochs=args.epochs,
                  lr = args.lr
                  )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)











