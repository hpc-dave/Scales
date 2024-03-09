import numpy as np
from numpy import matlib
from matplotlib import pyplot as plt
from matplotlib import cm
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
from scipy.sparse.linalg import gmres
from pyamg.aggregation import smoothed_aggregation_solver

def PlotPField(P):
    plt.style.use('_mpl-gallery')
    X = np.array(range(P.shape[0]))
    Y = np.array(range(P.shape[1]))
    X,Y = np.meshgrid(X, Y)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X, Y, P)
    plt.show()

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


def ComputePotentialField(image):
    ###################
    # construct matrix
    ###################
    dim = len(image.shape)
    num_rows = image.size
    num_nb = dim * 2

    # assigning a unique index to each cell
    indices = np.array(range(num_rows), dtype=int)
    indices = np.reshape(indices, image.shape)

    # preparing a list of neighbors, first index designates the number faces
    # following order
    # 0 -> west, 1->east, 2->south, 3->north, 4->bottom, 5->top
    W, E, S, N, B, T = range(6)
    shape_l = np.append(np.array(num_nb), (indices.shape))
    nb = np.full(shape_l, -1, dtype=int)
    nb[W, 1:  , ...]    = indices[ :-1, ...]                    # noqa: E203, E201, E221, E501
    nb[E,  :-1, ...]    = indices[1:  , ...]                    # noqa: E203, E201, E221, E501
    nb[S,  :  , 1:  , ...] = indices[ :  ,  :-1, ...]           # noqa: E203, E201, E221, E501
    nb[N,  :  ,  :-1, ...] = indices[ :  , 1:  , ...]           # noqa: E203, E201, E221, E501
    if dim > 2:
        nb[B, :, :, 1:  ] = indices[ :, :, 0:-1]                # noqa: E203, E201, E221, E501, E202
        nb[T, :, :,  :-1] = indices[ :, :, 1:  ]                # noqa: E203, E201, E221, E501, E202

    # determine;/ those neighbors which are associated with a phase change
    pc = np.full(shape_l, False)
    pc[E, :-1, ...] = pc[W, 1:, ...] = image[:-1, ...] != image[1:, ...]
    pc[N, :, :-1, ...] = pc[S, :, 1:, ...] = image[:, :-1, ...] != image[:, 1:, ...]
    if dim > 2:
        pc[T, :, :, :-1] = pc[B, :, :, 1:] = image[:, :, :-1] != image[:, :, 1: ]

    # all faces with phase changes are now removed from the neighbor list (set to -1), so we can form the
    # adjacency matrix
    nb[pc] = -1
    shape_l[1:] = 1
    rows = np.tile(indices, shape_l)
    rows = rows.flatten()
    nb = nb.flatten()
    filter = nb > -1
    rows = rows[filter]
    nb = nb[filter]

    A = sparse.coo_matrix((np.ones(rows.shape), (rows, nb)), shape=(num_rows, num_rows))
    A = sparse.csgraph.laplacian(A).astype(float)

    # add manipulation for dirichlet condition at faces
    Dbc = np.squeeze(np.sum(pc.astype(int), axis=0)).flatten()
    A += sparse.diags(Dbc)

    B = np.ones(num_rows, dtype=float)
    B[(image == solid).ravel()] = -1

    # x = linalg.spsolve(A, B)
    M = smoothed_aggregation_solver(A).aspreconditioner(cycle='V')
    x, _ = gmres(A, B, atol=1e-8, M=M)

    sol = np.reshape(x, image.shape)

    return sol


ldim = (1500, 1500)

solid = 0
fluid = 255

image = np.full(shape=ldim, fill_value=solid, dtype='uint8')

radius = ldim[0] * 0.1

center = np.array(ldim)
dimarr = center.copy()
center[0], center[1] = dimarr * 0.25
AddBall(image, center=np.floor(center), radius=radius, value=fluid)
center[0], center[1] = dimarr[0] * 0.75, dimarr[1] * 0.25
AddBall(image, center=np.floor(center), radius=radius, value=fluid)
center[0], center[1] = dimarr[0] * 0.25, dimarr[1] * 0.75
AddBall(image, center=np.floor(center), radius=radius, value=fluid)
center[0], center[1] = dimarr[0] * 0.75, dimarr[1] * 0.75
AddBall(image, center=np.floor(center), radius=radius, value=fluid)

p1 = dimarr.copy()
p2 = dimarr.copy()
# p1[0], p1[1], p2[0], p2[1] = 0, 1, dimarr[0], dimarr[1]-1
# AddBox(image, np.floor(p1), np.floor(p2), value=fluid)
p1[0], p1[1], p2[0], p2[1] = 0, dimarr[1]*0.2, dimarr[0], dimarr[1]*0.3
AddBox(image, np.floor(p1), np.floor(p2), 255)
p1[0], p1[1], p2[0], p2[1] = 0, dimarr[1]*0.7, dimarr[0], dimarr[1]*0.8
AddBox(image, np.floor(p1), np.floor(p2), 255)
p1[0], p1[1], p2[0], p2[1] = dimarr[0]*0.2, 0, dimarr[0]*0.3, dimarr[1]
AddBox(image, np.floor(p1), np.floor(p2), 255)
p1[0], p1[1], p2[0], p2[1] = dimarr[0]*0.7, 0, dimarr[0]*0.8, dimarr[1]
AddBox(image, np.floor(p1), np.floor(p2), 255)

P = ComputePotentialField(image)

# Extract network from solution
# define tolerance to avoid FPA noise
tol = (np.max(P) - np.min(P)) * 1e-10

# find changes in derivative in x, y (and z)
dim = len(image.shape)
network = np.full(image.shape, False)
network_x = ((P[1:-1, ...] - P[:-2, ...]) > tol) & ((P[1:-1, ...] - P[2:, ...]) > tol)
network_y = ((P[:, 1:-1, ...] - P[:, :-2, ...]) > tol) & ((P[:, 1:-1, ...] - P[:, 2:, ...]) > tol)
network[1:-1, ...] = network_x
network[:, 1:-1, ...] |= network_y
if dim > 2:
    network_z = ((P[:, :, 1:-1] - P[:, :, :-2]) > tol) & ((P[:, :, 1:-1] - P[:, :, 2:]) > tol)
    network[:, :, 1:-1 ] |= network_z

# network_EN = ((P[1:-1, 1:-1, ...] - P[:-2, :-2, ...]) > tol) & ((P[1:-1, 1:-1, ...] - P[2:,2:, ...]) > tol)
# network_ES = ((P[1:-1, 1:-1, ...] - P[:-2, 2:, ...]) > tol) & ((P[1:-1, 1:-1, ...] - P[2:,:-2, ...]) > tol)
# network[1:-1, 1:-1, ...] = network_EN
# network[1:-1, 1:-1, ...] |= network_ES

# network by nodes
# grad_shape = np.array(image.shape, dtype=int)-1
# gradn = np.full(grad_shape, False)
# grad_EN = (P[1:, 1:, ...] - P[:-1, :-1, ...])
# grad_ES = (P[1:, :-1, ...] - P[:-1, 1:, ...])
# grad_x = (P[:-1, :-1, ...] + P[:-1, 1:, ...] - P[1:, :-1, ...] - P[1:, 1:, ...]) * 0.5
# find pores at the crossings
map = network.astype(int)
cross = np.zeros_like(map)
cross[:-1, ...] += map[1:, ...]
cross[1:, ...] += map[:-1, ...]
cross[:, :-1, ...] += map[:, 1:, ...]
cross[:, 1:, ...] += map[:, :-1, ...]
if dim > 2:
    cross[:, :, :-1] += map[:, :, 1:]
    cross[:, :, 1:]  += map[:, :, :-1]

cross = cross > 2


imsol = P.copy()
imsol = imsol - np.min(imsol)
imsol = imsol / np.max(imsol) * 255
plt.imshow(imsol)

print('finished')
