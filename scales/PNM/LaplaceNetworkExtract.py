import numpy as np


def AddBox(image, point1, point2, value):
    dim = len(p1)
    p1 = np.array(point1)
    p2 = np.array(point2)
    for i in range(dim):
        if p1[i] > p2[i]:
            p1[i], p2[i] = p2[i], p1[i].copy()
    image[p1[0] : ]


lx = 100
ly = 100

fluid = 255

image = np.zeros((lx, ly))
