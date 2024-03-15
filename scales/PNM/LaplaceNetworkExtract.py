import collections
import numpy as np
from numpy import matlib
from matplotlib import pyplot as plt
from matplotlib import cm
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
from scipy.sparse.linalg import gmres
from pyamg.aggregation import smoothed_aggregation_solver
import skimage


def PlotPField(P):
    plt.style.use('_mpl-gallery')
    X = np.array(range(P.shape[0]))
    Y = np.array(range(P.shape[1]))
    X,Y = np.meshgrid(X, Y)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X, Y, P)
    plt.show()


def PlotNetwork(network, pores=None, throats=None):
    plt.style.use('_mpl-gallery')
    X, Y, Z = network

    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(X, Y, Z)

    if pores is not None:
        X, Y, Z = pores
        ax.scatter(X, Y, Z, 'o')

    if throats is not None:
        X, Y, Z = throats
        ax.scatter(X, Y, Z, '$|-|$')


def DetermineEigenFeatures(hess):
    k = skimage.feature.hessian_matrix_eigvals(hess)
    shape = np.append(np.array((4)), k.shape)
    D = np.zeros(shape, dtype=float)
    D[0,...] = k[0,...]
    D[1,...] = np.sqrt(k[0,:]**2 + k[1,:]**2)
    D[2,...] = (k[0,:]**2 + k[1,:]**2)**2
    D[3,...] = np.abs(k[0,:]-k[1,:]) * np.abs(k[0,:]+k[1,:])

    theta = np.zeros_like(k)
    theta[0,...] = hess[1]
    theta[1,...] = k[0,...] - hess[0]

    return k, D, theta


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


def ComputePotentialField(image, source: float = 1, image_edge='NoFlux'):
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

    # determine those neighbors which are associated with a phase change
    pc = np.full(shape_l, False)
    pc[E, :-1, ...] = pc[W, 1:, ...] = image[:-1, ...] != image[1:, ...]
    pc[N, :, :-1, ...] = pc[S, :, 1:, ...] = image[:, :-1, ...] != image[:, 1:, ...]
    if dim > 2:
        pc[T, :, :, :-1] = pc[B, :, :, 1:] = image[:, :, :-1] != image[:, :, 1: ]

    # treat image edges as solid phase
    if image_edge == 'Dirichlet':
        pc[E, -1, ...] = True
        pc[W, 0, ...] = True
        pc[N, :, -1, ...] = True
        pc[S, :, 0, ...] = True
        if dim > 2:
            pc[T, :, :, -1] = True
            pc[B, :, :, 0] = True

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

    B = np.full(num_rows, fill_value=source, dtype=float)
    B[(image == solid).ravel()] = -source

    # x = linalg.spsolve(A, B)
    M = smoothed_aggregation_solver(A).aspreconditioner(cycle='V')
    x, _ = gmres(A, B, atol=1e-8, M=M)

    sol = np.reshape(x, image.shape)

    return sol


def LocalExtrema(image, mode: str = 'max', allow_borders: bool = True, indices: bool = False):
    def IsLarger(a, b):
        return a > b

    def IsSmaller(a, b):
        return a < b

    if mode == 'min':
        comp = IsSmaller
    elif mode == 'max':
        comp = IsLarger
    else:
        raise Exception('mode does not exist: ' + mode)

    dim = len(image.shape)
    extrema = np.full(image.shape, fill_value=True)
    extrema[:-1, ...] &= comp(image[:-1, ...], image[1:, ...])  # EAST
    extrema[1:, ...] &= comp(image[1:, ...], image[:-1, ...])   # WEST
    if dim > 1:
        extrema[ : , :-1, ...] &= comp(image[:  , :-1, ...], image[:, 1:, ...])    # NORTH
        extrema[ : , 1: , ...] &= comp(image[:  , 1: , ...], image[:, :-1, ...])   # SOUTH
        extrema[:-1, :-1, ...] &= comp(image[:-1, :-1, ...], image[1:, 1:, ...])   # NORTH-EAST
        extrema[1: , :-1, ...] &= comp(image[1: , :-1, ...], image[:-1, 1:, ...])  # SOUTH-EAST
        extrema[:-1, 1: , ...] &= comp(image[:-1, 1: , ...], image[1:, :-1, ...])  # NORTH-WEST
        extrema[1: , 1: , ...] &= comp(image[1: , 1: , ...], image[:-1, :-1, ...]) # SOUTH-WEST
    if dim > 2:
        for i in [(1,image.shape[2]), (0,-1)]:
            j = (1, image.shape[2]) if i[0] == 0 else (0, -1)
            extrema[..., i[0]:i[1]] &= comp(image[..., i[0]:i[1]], image[..., j[0]:j[1]])  # CENTER
            extrema[:-1, :, i[0]:i[1]] &= comp(image[:-1, :, i[0]:i[1]], image[1:, :, j[0]:j[1]])  # EAST
            extrema[1:, :, i[0]:i[1]] &= comp(image[1:, :, i[0]:i[1]], image[:-1, :, j[0]:j[1]])   # WEST
            extrema[ : , :-1, i[0]:i[1]] &= comp(image[:  , :-1, i[0]:i[1]], image[:, 1:, j[0]:j[1]])    # NORTH
            extrema[ : , 1: , i[0]:i[1]] &= comp(image[:  , 1: , i[0]:i[1]], image[:, :-1,j[0]:j[1]])   # SOUTH
            extrema[:-1, :-1, i[0]:i[1]] &= comp(image[:-1, :-1, i[0]:i[1]], image[1:, 1:, j[0]:j[1]])   # NORTH-EAST
            extrema[1: , :-1, i[0]:i[1]] &= comp(image[1: , :-1, i[0]:i[1]], image[:-1, 1:, j[0]:j[1]])  # SOUTH-EAST
            extrema[:-1, 1: , i[0]:i[1]] &= comp(image[:-1, 1: , i[0]:i[1]], image[1:, :-1, j[0]:j[1]])  # NORTH-WEST
            extrema[1: , 1: , i[0]:i[1]] &= comp(image[1: , 1: , i[0]:i[1]], image[:-1, :-1, j[0]:j[1]]) # SOUTH-WEST

    if not allow_borders:
        extrema[0, ...] = False
        extrema[image.shape[0]-1, ...] = False
        if dim > 1:
            extrema[:, 0, ...] = False
            extrema[:, image.shape[1]-1, ...] = False
        if dim > 2:
            extrema[:,:, 0] = False
            extrema[:,:, image.shape[2]-1] = False

    return np.nonzero(extrema) if indices else extrema


def FindPoresAndFaces(P: np.array, network: np.array, indices: bool = True):
    r"""
    Determines location of pores and faces

    Parameters
    ----------
    P: np.array
        potential field of size M, N[, P]
    network: np.array
        network in the from of a field, where pathways/ridges are assigned
        a nonzero value
    indices: bool
        determines, if either indices or fields are returned by this function

    Returns
    -------
    Depending on the value of 'indices', either two tuples of indices
    are returned or two marked boolean fields
    """
    P_manip = P.copy()
    P_manip[~network] = 0

    P_manip = np.abs(P_manip)
    pores = skimage.morphology.local_maxima(P_manip,
                                            allow_borders=False,
                                            indices=False)

    P_manip *= -1
    P_manip[~network] = np.min(P_manip)
    faces = skimage.morphology.local_maxima(P_manip,
                                            allow_borders=False,
                                            indices=False)

    if indices:
        return np.nonzero(pores), np.nonzero(faces)
    else:
        return pores, faces


def BreadthFirstSearch(A: sparse.csr_array, pores, network_coords, start: int, away_from: int = -1):
    # https://stackoverflow.com/questions/47896461/get-shortest-path-to-a-cell-in-a-2d-array-in-python
    queue = collections.deque([start])
    seen = set([start])
    p_ref = network_coords[away_from if away_from > -1 else start, :]
    min_dist = np.sum((p_ref - network_coords[start, :])**2)
    while queue:
        row = queue.popleft()
        coord = network_coords[row, :]
        if tuple(coord) in pores:
            return row
        for pos in range(A.indptr[row], A.indptr[row+1]):
            next_id = A.indices[pos]
            if (next_id not in seen) and (np.sum((network_coords[next_id] - p_ref)**2) > min_dist):
                queue.append(next_id)
                seen.add(next_id)
    # if we end up here, no pore was found and this is either a dead end or a boundary
    return -1


def EstablishConnections(P: np.array, network: np.array, pores: np.array, faces: np.array):

    # adjacency matrix for the network voxels
    dim = len(P.shape)
    l_face = np.transpose(np.array(faces))
    l_pore = np.transpose(np.array(pores))
    num_faces = l_face.shape[0]
    num_pores = l_pore.shape[0]

    # Compute distance map of all points and remove those who are more than
    # one diagonal away. Note that a diagonally located point is
    # sqrt(dim* (1^2)) distant. We can save the (expensive) sqrt operation by
    # comparing it to the squared distance. Here, a padding is added to account
    # for floating point errors
    net_coords = np.transpose(np.array(np.nonzero(network)))
    dist_map = net_coords[..., np.newaxis]
    # dist_map = np.swapaxes(dist_map, axis1=0, axis2=1)
    dist_map = dist_map - np.swapaxes(dist_map, axis1=0, axis2=2)
    dist_map = np.sum(dist_map**2, axis=1)
    dist_map[dist_map > (dim+0.5)] = 0
    dist_map = sparse.coo_array(dist_map)

    # here we need to convert the reduced graph (only actual network-pixel/voxel are used)
    # to an image wide graph, so we can relate the actual positions to each other
    graph = dist_map.astype(bool).astype(int)
    # pore_ids = indices[pores]
    # face_ids = indices[faces]
    # net_ids = indices[network][..., np.newaxis]

    # row = (graph * np.tile(net_ids, reps=(1, net_ids.size))).data
    # net_ids = np.transpose(net_ids)
    # col = (graph * np.tile(net_ids, reps=(net_ids.size, 1))).data
    # v = np.full((col.size), fill_value=True)
    # graph = sparse.coo_array(row, col, v).tocsr()

    # net_coords = dict(zip(indices[network].tolist(), np.array(net_coords).tolist()))
    pore_coords = dict(zip([tuple(i) for i in l_pore], range(num_pores)))
    # map_coords_to_face = dict(zip([tuple(i) for i in faces], range(num_faces)))
    map_coords_to_row = dict(zip([tuple(i) for i in net_coords], range(net_coords.shape[0])))
    row_to_pore = dict(zip([ map_coords_to_row[tuple(l_pore[i,:].tolist())] for i in range(num_pores)], range(num_pores)))
    row_to_pore[-1] = -1  # default for invalid values
    conns = np.full((num_faces, 2), fill_value=-1)
    graph = graph.tocsr()
    for i in range(num_faces):
        f_coord = l_face[i, :]
        row_start = map_coords_to_row[tuple(f_coord)]
        p1 = BreadthFirstSearch(A=graph, pores=pore_coords, network_coords=net_coords, start=row_start)
        if p1 == -1:
            conns[i, :] = -1
            continue
        p2 = BreadthFirstSearch(A=graph, pores=pore_coords, network_coords=net_coords, start=row_start, away_from=p1)
        if p2 == -1:
            p1, p2 = p2, p1
        conns[i, :] = [row_to_pore[p1], row_to_pore[p2]]

    return conns









    # determine those neighbors which are associated with a phase change
    # pc = np.full(shape_l, False)
    # pc[E, :-1, ...] = pc[W, 1:, ...] = image[:-1, ...] != image[1:, ...]
    # pc[N, :, :-1, ...] = pc[S, :, 1:, ...] = image[:, :-1, ...] != image[:, 1:, ...]
    # if dim > 2:
    #     pc[T, :, :, :-1] = pc[B, :, :, 1:] = image[:, :, :-1] != image[:, :, 1: ]

    # def FillStencil(v, P, ind, dim):
    #     if dim == 2:
    #         for i in range(-1, 2):
    #                 for j in range(-1, 2):
    #                     v[i, j] = P[ind[0]+i, ind[1]+j]
    #     else:
    #         for i in range(-1, 2):
    #                 for j in range(-1, 2):
    #                     for k in range(-1, 2):
    #                         v[i, j, k] = P[ind[0] + i, ind[1] + j, ind[2] + k]

    # dim = len(np.array.shape)
    # num_faces = faces.shape[0]
    # conns = np.full((num_faces, 2), fill_value=-1, dtype=int)

    # # look up table for the pores, exploiting efficient hashing
    # t_pores = dict.fromkeys([tuple(i) for i in pores.tolist()])

    # Pabs = np.abs(P)
    # Pabs[~network] = 0

    # stencil_shape=list(3)
    # for i in range(1,dim):
    #     stencil_shape.append(3)
    # stencil_shape = tuple(stencil_shape)

    # for i in range(num_faces):

    #     v = np.array(stencil_shape, dtype=P.dtype)
    #     ind = faces[i, :]
    #     FillStencil(v, Pabs, faces[i,:], dim)
    #     p0 = np.argmax(v)
    #     v[p0] = 0
    #     p1 = np.argmax(v)
    #     # test if the distance between the points is less than
    #     # one cell diagonal
    #     count, tries = 0, 2 if dim == 2 else 8
    #     while (np.sum((p1-p0)**2) < 2) & (count < tries):
    #         v[p1] = 0
    #         p1=np.argmax(v)
    #         count += 1
    #     if count == tries:





        # tracer_lo = faces[i,:].copy()
        # tracer_up = tracer_lo.copy()
        # v_search = np.full((dim), fill_value=-1, dtype='int8')

        # p_old = tracer_lo.copy()
        # while True:
        #     tracer_lo += v_search
        #     mask = stencil.copy()
        #     ind = tuple((np.ones((dim), dtype=v_search.dtype) - v_search ).tolist())
        #     mask[ind] = False
        #     p_lo = tracer_lo - 1
        #     p_up = tracer_lo + 2
        #     v = np.array(stencil_shape, dtype=network.dtype)
        #     if dim == 2:
        #         for i in range(0, 3):
        #             for j in range(0, 3):
        #                 v[i, j] = network[p_lo[0]+i, p_lo[1]+j]
        #     else:
        #         for i in range(0, 3):
        #             for j in range(0, 3):
        #                 for k in range(0, 3):
        #                     v[i, j, k] = network[p_lo[0]+i, p_lo[1]+j, p_lo[2]+k]
        #     candidates = np.nonzero(v & mask)


def GetNetworkFeatures(image: np.array,
                       P: np.array,
                       threshold=None,
                       include_solid: bool = False):
    r"""
    Extracts network features from image and potential field

    Parameters
    ----------
    image : numpy.array
        A 2D or 3D image of the porous media, on which the potential field was computed
    P : np.array
        Potential field determined on the base of the provided image
    threshold
        a threshold to divide between solid and fluid phases in the image, if set to None
        the average will be taken
    include_solid : bool
        if set to true, a second network inside the solid will be extracted, otherwise only
        the fluid phase is extracted

    Returns
    ------
    network
        a set of coordinates of the pixels/voxel belonging to the extracted skeleton
    pores
        a tuple of the coordinates and labels for each phase
    throats
        a tuple of the coordinates of each throat and phase

    Notes
    -----
    Here, following steps are conducted:
        - skeletonize the original image according to Lee94
        - invert the image and conduct the same if the solid is included
        - remove all potential field values outside of this skeleton
        - define pores as maxima of the absolute potential field along the skeleton
        - define throats as minima of the potential field along the skeleton
        - label them according to their respective image values

    References
    ----------
    Lee94 T.-C. Lee, R.L. Kashyap and C.-N. Chu, Building skeleton models via 3-D medial surface/axis thinning algorithms. Computer Vision, Graphics, and Image Processing, 56(6):462-478, 1994.
    """

    # prepare the threshold
    if threshold is None:
        threshold = (np.max(image)-np.min(image))/2

    # some checks
    if image.shape != P.shape:
        raise Exception('image and potential field do not\
                        have identical dimensions!')
    if len(image.shape) < 2:
        raise Exception('the images dimension is less than 2, aborting')
    if len(image.shape) > 3:
        raise Exception('the images dimension is larger than 3, aborting')
    if not np.all(P[image > threshold] > 0):
        raise Warning('the sign of the potential field values\
                      does not align with the specified image phases')

    # think about ridges maybe instead: https://scikit-image.org/docs/stable/auto_examples/edges/plot_ridge_filter.html
    # could be computationally preferable to Lee, since Lee uses an iterative algorithm
    # skeletonize the image according to Lee
    network = skimage.morphology.skeletonize(image, method='lee').astype(bool)
    if include_solid:
        network |= skimage.morphology.skeletonize(np.invert(image), method='lee').astype(bool)

    # finding pores and faces along the skeleton, depending on the potential field
    # note how we assign the faces as throats right now, that should be changed soon
    pores, faces = FindPoresAndFaces(P, network)

    throats = EstablishConnections(network=network, P=P, pores=pores, faces=faces)

    # assign labels
    p_phase0 = np.where(image[pores] < threshold)[0]
    p_phase1 = np.where(image[pores] > threshold)[0]
    t_phase0 = np.where(image[pores] < threshold)[0]
    t_phase1 = np.where(image[pores] > threshold)[0]

    # connecting networks
    # try strongest gradient method starting from solid: skimage.filters.rank.gradient
    # also this one here might work: skimage.segmentation.inverse_gaussian_gradient

    return np.nonzero(network), (pores, p_phase0, p_phase1), (throats, t_phase0, t_phase1)


A = np.zeros((5,5))
A[0,1] = 1
A[0,4] = 2
A[1,2] = 5
A[2,3] = 3
A[3,4] = 4
A[4,0] = 1
A = sparse.csr_array(A)
for ind in range(A.indptr[0], A.indptr[1]):
    print(A.indices[ind])

ldim = (30, 40)

solid = 0
fluid = 255

image = np.full(shape=ldim, fill_value=solid, dtype='uint8')

radius = ldim[0] * 0.1

center = np.array(ldim)
dimarr = center.copy()

# center[0], center[1], center[2] = dimarr * 0.35
# center[1] , center[2] = dimarr[1]*0.5, dimarr[2]*0.5
center[0], center[1]= dimarr * 0.35
center[1] = dimarr[1]*0.5
AddBall(image, center=np.floor(center), radius=ldim[0]*0.15, value=fluid)
center[0] = dimarr[0] * 0.65
AddBall(image, center=np.floor(center), radius=ldim[0]*0.15, value=fluid)

# center[0], center[1] = dimarr * 0.5
# AddBall(image, center=np.floor(center), radius=ldim[0]*0.3, value=fluid)

# center[0], center[1] = dimarr * 0.25
# AddBall(image, center=np.floor(center), radius=radius, value=fluid)
# center[0], center[1] = dimarr[0] * 0.75, dimarr[1] * 0.25
# AddBall(image, center=np.floor(center), radius=radius, value=fluid)
# center[0], center[1] = dimarr[0] * 0.25, dimarr[1] * 0.75
# AddBall(image, center=np.floor(center), radius=radius, value=fluid)
# center[0], center[1] = dimarr[0] * 0.75, dimarr[1] * 0.75
# AddBall(image, center=np.floor(center), radius=radius, value=fluid)

p1 = dimarr.copy()
p2 = dimarr.copy()
# p1[0], p1[1], p2[0], p2[1] = 0, 1, dimarr[0], dimarr[1]-1
# AddBox(image, np.floor(p1), np.floor(p2), value=fluid)
# p1[0], p1[1], p2[0], p2[1] = 0, dimarr[1]*0.2, dimarr[0], dimarr[1]*0.3
# AddBox(image, np.floor(p1), np.floor(p2), 255)
# p1[0], p1[1], p2[0], p2[1] = 0, dimarr[1]*0.7, dimarr[0], dimarr[1]*0.8
# AddBox(image, np.floor(p1), np.floor(p2), 255)
# p1[0], p1[1], p2[0], p2[1] = dimarr[0]*0.2, 0, dimarr[0]*0.3, dimarr[1]
# AddBox(image, np.floor(p1), np.floor(p2), 255)
# p1[0], p1[1], p2[0], p2[1] = dimarr[0]*0.7, 0, dimarr[0]*0.8, dimarr[1]
# AddBox(image, np.floor(p1), np.floor(p2), 255)

# p1[0], p1[1], p2[0], p2[1] = 1, 1, dimarr[0]-2, dimarr[1]-2
# AddBox(image, np.floor(p1), np.floor(p2), value=fluid)

P = ComputePotentialField(image, source=1e-0)

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

network, pores, throats = GetNetworkFeatures(image, P, include_solid=False)

PlotNetwork(network, pores[0], throats[0])

print('finished')
