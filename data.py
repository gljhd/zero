import cv2
import numpy as np
          #background;    airplane;   bicycle;    bird;   boat;   bottle;
palette = [[128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128],
                #bus;   car;    cat;    chair;  cow;    Dining table;
                [0, 128, 128],[128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                #dog;   horse;  motor bike;     persion;    potted plant;   sheep;
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
                # sofa;   train;  monitor
                [0, 192, 0], [128, 192, 0], [0, 64, 128]]
img = cv2.imread('/media/wingspan/ssd512/Glj_train/VOCtrainval_2012/VOCdevkit/VOC2012/SegmentationClass/2007_000042.png')
def seg2label(dir):
    #img = cv2.imread(dir)
    w = img.shape[0]
    h = img.shape[1]
    labels = np.zeros((w, h), dtype=np.int32)
    print((img[1,1,1]==0))
    a = int(True)
    for i in range(20):
        labels += ((img[:,:,2] == palette[i][0]) * (img[:,:,1] == palette[i][1]) * (img[:,:,0] == palette[i][2]))*(i+1)
    one_hot = np.zeros((w,h,21))
    for i in range(w):
        for j in range(h):
            one_hot[i,j,labels[i,j]] = 1
    return one_hot
labels = seg2label(img)
cv2.imshow('Image',labels[:,:,19]*255)
cv2.waitKey(0)
