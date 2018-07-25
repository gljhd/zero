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

if __name__ == '__main__':
    a = [1, 2, 3, 4, 5, 6, 7]
    for i in test(a):
        print(i)