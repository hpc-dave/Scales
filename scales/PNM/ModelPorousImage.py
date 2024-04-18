
import numpy as np
import random


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


def TwoTouchingPores(image, value=255):
    center = (np.asarray(image.shape) / 2).astype(int)
    radius = int(np.min(center)*0.3)
    bcenter = center.copy()
    bcenter[0] = center[0] - radius
    AddBall(image, center=np.floor(bcenter), radius=radius, value=value)
    bcenter[0] = center[0] + radius
    AddBall(image, center=np.floor(bcenter), radius=radius, value=value)


def ArrayOfBalls(image, shape, shrink_factor: float = 1, value: float = 0):
    dim = len(image.shape)
    if len(shape) < dim:
        shape = list(shape)
        for i in range(len(shape), len(image)):
            shape.append(0)
        shape = tuple(shape)

    if dim != len(shape):
        raise Exception('image dimension and array dimension are not compatible')

    num_pores = np.prod(shape)

    cshape = [dim]
    cshape.extend(shape)
    coords = np.full(cshape, fill_value=-1)
    radius = float(image.shape[0])
    for d in range(dim):
        dn = float(image.shape[d]) / shape[d]
        offset = dn * 0.5
        coord = np.asarray([int(offset+dn*i) for i in range(shape[d])])
        coord = coord.reshape([shape[n] if n == d else 1 for n in range(dim)])
        coord = np.tile(coord, reps=[shape[n] if n != d else 1 for n in range(dim)])
        coords[d, ...] = coord
        radius = np.min([radius, dn*0.5*shrink_factor])

    coords = np.transpose(coords.reshape((dim, num_pores)))
    for i in range(num_pores):
        AddBall(image=image, center=coords[i, :], radius=radius, value=value)


def RandomUniformBalls(image, target_porosity: float, radius: float, value=0, seed=None):
    random.seed(a=seed)
    shape = image.shape
    dim = len(image.shape)

    porosity = float(image[image == value].size) / image.size
    while porosity < target_porosity:
        coords = [random.randrange(0, image.shape[d]-1) for d in range(dim)]
        AddBall(image=image, center=coords, radius=radius, value=value)
        porosity = float(image[image == value].size) / image.size
