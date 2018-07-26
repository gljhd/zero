import random
def test(a):
    epoches =3
    for epoch in range(epoches):
        b = []
        n = 0
        random.shuffle(a)
        for i in a:
            b.append(i)
            n += 1;
            if n == 2:
                yield b
                b = []
                n = 0
class inconv(nn.Module):
    def __int__(self, in_ch, out_ch):
        super(inconv, self).__int__()
        self.conv = double_conv(in_ch, out_ch)
    def forward(self, x):
        x = self.conv(x)
        return x
if __name__ == '__main__':
    file = '/media/wingspan/ssd512/Glj_train/VOCtrainval_2012/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt'
    a = []
    with open(file) as f:
        for i in f.readlines():
            a.append(i)
    print(len(a))
    print(a[0])
