import cv2
import numpy as np
from config import palette


def seg2label(dir):
    img = cv2.imread(dir)
    w = img.shape[0]
    h = img.shape[1]
    labels = np.zeros((w, h), dtype=np.int32)
    for i in range(20):
        labels += ((img[:,:,2] == palette[i][0]) * (img[:,:,1] == palette[i][1]) * (img[:,:,0] == palette[i][2]))*(i+1)
    return labels
path = '/media/wingspan/ssd512/Glj_train/VOCtrainval_2012/VOCdevkit/VOC2012/SegmentationClass/2007_000039.png'
seg2label(path)

