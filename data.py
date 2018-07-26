import cv2
import numpy as np
import config as cfg
import os

def seg2label(dir):
    img = cv2.imread(dir)
    h = img.shape[0]
    w = img.shape[1]
    labels = np.zeros((h, w), dtype=np.int32)
    for i in range(20):
        labels += ((img[:, :, 2] == cfg.palette[i][0]) * (img[:,:,1] == cfg.palette[i][1]) * (img[:,:,0] == cfg.palette[i][2]))*(i+1)
    '''
    one_hot = np.zeros((neww, newh, 21))
    for i in range(neww):
        for j in range(newh):
            one_hot[i, j, labels[i, j]] =1
    cv2.imshow('Image1', one_hot[:, :, 15] * 255)
    cv2.imshow('Image2', one_hot[:, :, 2] * 255)
    cv2.waitKey(0)
    '''
    return labels
class data_load(object):
    def __init__(self, file_list, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.batch_size = cfg.batch_size
        self.train_list = self.get_list(file_list)
    def get_list(self, file_list):
        lists = []
        with open(file_list) as f:
            for l in f.readlines():
                lists.append(l.strip('\n'))
        return lists
    def data_gene(self):
        imgs = []
        labels = []
        batches = 0
        for i in self.train_list:
            img = cv2.imread(os.path.join(self.img_dir, (i+'.jpg')))
            label = seg2label(os.path.join(self.mask_dir, (i+'.png')))
            img, label = self.resize_img_lab(img,label)
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
        img = img.transpose(2, 0, 1).astype(np.float32)
        label = cv2.resize(label, (cfg.width, cfg.height), interpolation = cv2.INTER_NEAREST)
        return img, np.ceil(label)
def main():
    dir = '/media/wingspan/ssd512/Glj_train/VOCtrainval_2012/VOCdevkit/VOC2012/SegmentationClass/2007_000129.png'
    seg2label(dir)

if __name__ == '__main__':
    main()




