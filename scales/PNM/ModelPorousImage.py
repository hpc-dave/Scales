
import numpy as np


def AddBox(image, point1, point2, value):
    p1 = np.array(point1, dtype=int)
    p2 = np.array(point2, dtype=int)
    dim = len(image.shape)
    for i in range(dim):
        if p1[i] > p2[i]:
            p1[i], p2[i] = p2[i], p1[i].copy()

    if dim == 2:
        image[p1[0]:p2[0], p1[1]:p2[1]] = value
    elif dim == 3:
        image[p1[0]:p2[0], p1[1]:p2[1], p1[2]:p2[2]] = value

    return image


def AddBall(image, center, radius, value):
    dim = len(image.shape)
    pc = np.array(center)
    shape = np.array(image.shape)
    shape = np.append(shape, dim)

    prel = np.zeros(shape=shape, dtype=float)
    for d in range(dim):
        shape_l = np.ones_like(shape[:-1])
        shape_l[d] = shape[d]
        l_v = np.reshape((np.array(range(shape[d])) - pc[d])**2, shape_l)
        shape_l = shape[:-1].copy()
        shape_l[d] = 1
        prel[..., d] = np.tile(l_v, shape_l)

    mask = np.sqrt(np.sum(prel, axis=-1)) <= radius
    image[mask] = value
    return image


def TwoThroatsOnePore(image):
    center = (np.asarray(image.shape) / 2).astype(int)
    radius = int(np.min(center)*0.5)
    AddBall(image, center=np.floor(center), radius=radius, value=255)
    p0 = center * 0.8
    p1 = center * 1.2
    p0[0] = 0
    p1[0] = image.shape[0]
    AddBox(image, point1=p0, point2=p1, value=255)


def FourThroatsOnePore(image):
    center = (np.asarray(image.shape) / 2).astype(int)
    radius = int(np.min(center)*0.5)
    p1 = center.copy()
    p2 = center.copy()
    p1[0], p2[0] = 0, image.shape[0]
    p1[1] = center[1]-radius*2/3
    p2[1] = p1[1] + radius/2
    AddBall(image, center=center, radius=radius, value=255)
    AddBox(image, np.floor(p1), np.floor(p2), value=255)
    p1[1] = center[1]+radius*2/3
    p2[1] = p1[1] - radius/2
    AddBox(image, np.floor(p1), np.floor(p2), value=255)


def TwoTouchingPoresInSolid(image):
    center = (np.asarray(image.shape) / 2).astype(int)
    radius = int(np.min(center)*0.3)
    bcenter = center.copy()
    bcenter[0] = center[0] - radius
    AddBall(image, center=np.floor(bcenter), radius=radius, value=255)
    bcenter[0] = center[0] + radius
    AddBall(image, center=np.floor(bcenter), radius=radius, value=255)
