def test():
    a = []
    b = 0
    for i in range(5):
        a.append(i)
        b += 1
        if b== 2:
            yield a
            b = 0
            a = []
c = test()
for i in c:
    print(i)


