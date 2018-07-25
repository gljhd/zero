import cv2
import numpy as np
import config as cfg


def seg2label(dir):
    img = cv2.imread(dir)
    w = img.shape[0]
    h = img.shape[1]
    img = cv2.resize(img,(neww,newh),interpolation = cv2.INTER_NEAREST)
    labels = np.zeros((neww, newh), dtype=np.int32)
    for i in range(20):
        labels += ((img[:,:,2] == cfg.palette[i][0]) * (img[:,:,1] == cfg.palette[i][1]) * (img[:,:,0] == cfg.palette[i][2]))*(i+1)
    '''
    for i in range(w):
        for j in range(h):
            one_hot[i,j,labels[i,j]] =1
    cv2.imshow('Image1', one_hot[:,:,15]*255)
    cv2.imshow('Image2', one_hot[:, :, 2] * 255)
    cv2.waitKey(0)
    '''
    return labels
class data_load(object):
    def __init__(self,train_dir):
        self.train_dir = train_dir
        self.train_list = []
        self.batch_size = cfg.batch_size
    def data_gene(self):
        imgs = []
        labels = []
        batches = 0
        for i in self.train_list:
            img = cv2.imread(i)
            label = seg2label(i)
            img,label = self.resize_img_lab(img,label)
            imgs.append(img)
            labels.append(label)
            batches += 1
            if batches == self.batch_size:
                yield np.array(imgs), np.array(labels)
                imgs = []
                labels = []
                batches = 0
    def resize_img_lab(self,img,label):
        img = cv2.resize(img, (cfg.width, cfg.height))
        label = cv2.resize(label, (cfg.width, cfg.height))
        return img, np.ceil(label)




