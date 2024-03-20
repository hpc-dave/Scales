import collections
import numpy as np
from matplotlib import pyplot as plt
from scipy import sparse
from scipy import ndimage
from scipy import spatial
from scipy.sparse.linalg import gmres as solver
from pyamg.aggregation import smoothed_aggregation_solver
import skimage
import ModelPorousImage as mpim


def PlotPField(P):
    plt.style.use('_mpl-gallery')
    X = np.array(range(P.shape[0]))
    Y = np.array(range(P.shape[1]))
    X, Y = np.meshgrid(X, Y)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X, Y, P)
    plt.show()


def PlotNetwork(network, pores=None, constr=None, P=None):
    is_tuple = type(network) is tuple
    if is_tuple:
        dim = len(network)
    else:
        dim = len(network.shape)

    if dim == 2:
        fig, ax = plt.subplots()
        if P is not None:
            image = P.copy()
            if not is_tuple:
                n = np.nonzero(network)
            else:
                n = network
            image[n] = 0
            ax.imshow(image)
        elif not is_tuple:
            ax.imshow(network)
        if pores is not None:
            ax.scatter(pores[1], pores[0])
            for i in range(len(pores[0])):
                ax.annotate(f'{i}', (pores[1][i], pores[0][i]))
    else:
        plt.style.use('_mpl-gallery')
        X, Y, Z = network

        ax = plt.figure().add_subplot(projection='3d')
        ax.plot(X, Y, Z)

        if pores is not None:
            X, Y, Z = pores
            ax.scatter(X, Y, Z, 'o')

        if constr is not None:
            X, Y, Z = constr
            ax.scatter(X, Y, Z, '$|-|$')


def PlotNetworkSpider(pores, conns):
    ax = plt.figure().add_subplot(projection='3d')
    x = pores[0]
    y = pores[1]
    if len(pores) == 3:
        z = pores[2]
    else:
        z = np.zeros_like(x)

    for i in range(np.max(conns.shape)):
        p0, p1 = conns[i, 0], conns[i, 1]
        if p0 == -1 or p1 == -1:
            continue

        c_x = [x[p0], x[p1]]
        c_y = [y[p0], y[p1]]
        c_z = [z[p0], z[p1]]

        ax.plot(c_x, c_y, c_z)

    ax.scatter(x, y, z)


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
    r"""
    Computes a potential field on an image of a porous media

    Parameters
    ----------
    image: array_like
        The (binary) image file of size [Nx, Ny [,Nz]] from which the image should be extracted.
        It does not to be strictly speaking binary, as long as all points
        within the same phase are labeled the same.
    source: float
        Source term to be applied during the computation.
    image_edge: str
        Indicator, how the image edge should be treated. By default, a
        no-flux boundary is applied. Alternatively, a Dirichlet boundary
        may be imposed

    Returns
    -------
    Array of type float and size [Nx, Ny [,Nz]] with the potential values

    Notes
    -----
    Here, following equation is solved to arrive at the potential field $\phi$:
    .. math:
        \nabla^2 \phi = \beta_i
    where the sign of source term $\beta$ takes differs between the phases
    $\Omega_0$ and $\Omega_1:
    .. math:
        \beta_i
        \begin{cases}
            \beta & for i \in \Omega_0 \\
            -\beta & for i \in \Omega_1
        \end{cases}
    At the boundary \Gamma between the phases, a Dirichlet condition is enforced:
    .. math:
        \phi_\Gamma = 0
    For images, those cell faces between points of different phase values are considered
    the location of the boundary condition.
    At the image boundary, the user may choose to either enforce a No-flux or Dirichlet
    boundary condition.

    The resulting matrix system is solved by an iterative solving algorithm (e.g. GMRES or BiCGStab)
    with an algebraic multigrid preconditioner.
    """
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
        pc[T, :, :, :-1] = pc[B, :, :, 1:] = image[:, :, :-1] != image[:, :, 1:]

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
    x, _ = solver(A, B, atol=1e-8, M=M)

    sol = np.reshape(x, image.shape)

    return sol


def Skeletonize(image, include_solid: bool = True, pad_width=3):
    im_loc = np.pad(image, pad_width=pad_width, mode='edge')
    # think about ridges maybe instead: https://scikit-image.org/docs/stable/auto_examples/edges/plot_ridge_filter.html
    # potential candidate: skimage.filters.sato(P, sigmas=[5])
    # could be computationally preferable to Lee, since Lee uses an iterative algorithm
    # skeletonize the image according to Lee
    network = skimage.morphology.skeletonize(im_loc, method='lee').astype(bool)
    if include_solid:
        network |= skimage.morphology.skeletonize(np.invert(im_loc), method='lee').astype(bool)
    dim = len(image.shape)
    if pad_width == 0:
        return network
    else:
        if dim == 2:
            network = network[pad_width:-pad_width, pad_width:-pad_width]
        else:
            network = network[pad_width:-pad_width, pad_width:-pad_width, pad_width:-pad_width]
        return network


def FindPoresAndConstrictions(P: np.array, network: np.array, indices: bool = True, include_solid: bool = True):
    r"""
    Determines location of pores and constrictions

    Parameters
    ----------
    P: np.array
        potential field of size M, N[, P]
    network: np.array
        network in the from of a field, where pathways/ridges are assigned
        a nonzero value
    indices: bool
        determines, if either indices or fields are returned by this function
    include_solid: bool
        applies the identification procedure to the solid phase, too

    Returns
    -------
    Depending on the value of 'indices', either two tuples of indices
    are returned or two marked boolean fields

    Notes
    -----
    The position of pores and constrictions are given by local maxima and minima
    on the potential field weighted network skeleton. Theoretically, the network
    for fluid and solid phase can be extracted simultaneously. However, the network
    can be problematic at very narrow crevices, depending on the network generation
    algorithm. To increase the robustness of this step, determining fluid and solid
    pores is conducted separately.
    """
    pores = np.full_like(P, dtype=bool, fill_value=False)
    constr = np.full_like(P, dtype=bool, fill_value=False)
    for i in range(1+include_solid):
        if i == 0:
            mask = P > 0
        else:
            mask = P < 0
        P_manip = P.copy()
        P_manip[~(mask & network)] = 0

        P_manip = np.abs(P_manip)
        pores |= skimage.morphology.local_maxima(P_manip,
                                                 allow_borders=False,
                                                 indices=False)

        P_manip *= -1
        P_manip[~(mask & network)] = np.min(P_manip)
        constr |= skimage.morphology.local_maxima(P_manip,
                                                  allow_borders=False,
                                                  indices=False)

    if indices:
        return np.nonzero(pores), np.nonzero(constr)
    else:
        return pores, constr


def FindConnectingPore_BFS(P, A: sparse.csr_array, pores, network_coords, start: int, extent, away_from: int = -1):
    r"""
    Finds the connecting pore via a breadth-first-search approach

    Parameters
    ----------
    P: array_like
        potential field for determining the appropriate phase and exclude network connections due to discretization
        issues
    A: csr_array
        adjacency matrix of discretized network. This should ONLY contain the nearest neighbors/surrounding cells
    pores
        tuple of coordinates of the determined pores
    network_coords
        look up table to get the coordinates of each discretized network path point
    start: int
        start point of the search in terms of a row in A, usually a constriction
    extent
        extent of the network, including the outermost points, e.g. shape - 1
    away_from: int
        reference point from which the search should lead away, used to force the search into opposite direction
        of previously found pore

    Returns
    -------
    tuple of determined row with a pore and identifier, if at boundary
        If no connecting pore was found (e.g. in the case of a dead end or boundary), -1
        is returned as row value

    Notes
    -----
    Took the search algorithm from https://stackoverflow.com/questions/47896461/get-shortest-path-to-a-cell-in-a-2d-array-in-python  # noqa E501
    """
    queue = collections.deque([start])
    seen = set([start])
    phase = P[tuple(network_coords[start])] > 0
    if away_from == -1:
        p_ref = network_coords[start]
        min_dist = 0
    else:
        p_ref = network_coords[away_from]
        min_dist = 3    # note that here we use the squared distance, so we can save a sqrt call

    is_boundary = False
    while queue:
        row = queue.popleft()
        coord = network_coords[row, :]
        # label as boundary pore
        if np.any((coord < 2) | ((extent - coord) < 2)):
            is_boundary = True
        # provide found pore
        if tuple(coord) in pores:
            return row, False
        # test for next adjacent cells
        for pos in range(A.indptr[row], A.indptr[row+1]):
            # here we test each potential connection, if it
            # a) is already in the cue (seen-array)
            # b) is in the correct phase
            # c) far away from the excluded point
            next_id = A.indices[pos]
            next_coords = network_coords[next_id]
            phase_next = P[tuple(next_coords)] > 0
            if (next_id not in seen) and (phase_next == phase) and (np.sum((next_coords - p_ref)**2) > min_dist):
                queue.append(next_id)
                seen.add(next_id)
    # if we end up here, no pore was found and this is either a dead end or a boundary
    return -1, is_boundary


def EstablishConnections(P: np.array, network: np.array, pores: np.array, constr: np.array):
    r"""
    Determines the connectivity between pores by performing a breadth-frist search
    along the network, starting from the constrictions

    Parameters
    ----------
    P: array_like
        Potential field of dimension [M, N [,P]]
    network: array_like
        network in the porous media in the form of a mask, where cells belonging to the network
        are labeled > 0
    pores: tuple
        coordinates of pores of form (array[Np], array[Np], [array[Np]])
    constr: tuple
        coordinates of constrictions of form (array[Nc], array[Nc], [array[Nc]])

    Returns
    -------
    A list of connections of form [Nc, 2] and mask of form [Nc], which of those are dead ends

    Notes
    -----
    Np and Nc denote the number of pores and constrictions.
    The general algorithm works along following steps:
        1) assign each network point a unique ID
        2) compute a distance map between each network point
        3) remove all the values which are not surrounding
           the respective point, effectively computing
           an adjacency matrix of the network
        4) start from each constriction and search twice with
           a breadth-first seach algorithm, to find
           the connecting pores
        5) label those constrictions, which are connected to only
           one pore as dead ends
    """
    # adjacency matrix for the network voxels
    dim = len(P.shape)
    l_constr = np.transpose(np.array(constr))
    l_pore = np.transpose(np.array(pores))
    num_constr = l_constr.shape[0]
    num_pores = l_pore.shape[0]

    # Compute distance map of all points and remove those who are more than
    # one diagonal away. Note that a diagonally located point is
    # sqrt(dim* (1^2)) distant. We can save the (expensive) sqrt operation by
    # comparing it to the squared distance. Here, a padding is added to account
    # for floating point errors
    net_coords = np.transpose(np.array(np.nonzero(network)))
    dist_map = net_coords[..., np.newaxis]
    dist_map = dist_map - np.swapaxes(dist_map, axis1=0, axis2=2)
    dist_map = np.sum(dist_map**2, axis=1)
    dist_map[dist_map > (dim+0.5)] = 0
    dist_map = sparse.coo_array(dist_map)

    # Create the graph of the network (only the reduced network, ignoring empty pixels/voxels)
    # Here, we basically loop through all constrictions and use them as seeds to go along the
    # provided network. Searching is done by applying a breadth-first approach to find the closest
    # pixel occupied by a pore
    graph = dist_map.astype(bool).astype(int).tocsr()
    pore_coords = dict(zip([tuple(i) for i in l_pore], range(num_pores)))
    map_coords_to_row = dict(zip([tuple(i) for i in net_coords], range(net_coords.shape[0])))
    row_to_pore = dict(zip([map_coords_to_row[tuple(l_pore[i, :].tolist())] for i in range(num_pores)], range(num_pores)))  # noqa E501
    row_to_pore[-1] = -1  # default for invalid values
    conns = np.full((num_constr, 2), fill_value=-1)
    extent = (np.asarray(P.shape) - np.ones(len(P.shape))).astype(int)
    dead_end = np.full(num_constr, fill_value=False)
    for i in range(num_constr):
        f_coord = l_constr[i, :]
        row_start = map_coords_to_row[tuple(f_coord)]
        p1, at_boundary = FindConnectingPore_BFS(P=P, A=graph, pores=pore_coords, network_coords=net_coords,
                                                 extent=extent, start=row_start)
        if p1 == -1 and not at_boundary:
            dead_end[i] = True
            conns[i, :] = -1
            continue
        p2, at_boundary = FindConnectingPore_BFS(P=P, A=graph, pores=pore_coords, network_coords=net_coords,
                                                 extent=extent, start=row_start, away_from=p1)
        if p2 == -1:
            dead_end[i] = not at_boundary
            p1, p2 = p2, p1
        conns[i, :] = [row_to_pore[p1], row_to_pore[p2]]

    return conns, dead_end


def profile_line(image, src, dst, spacing: int = 2, order=1):
    r"""
    Steps along a line and interpolates the values

    Parameters
    ----------
    image: array_like
        Image of dimension D to get the value from
    src: array_like
        coordinates of starting point with D entries
    dst: array_like
        coordinates of end point with D entries
    spacing: int
        step width for the interpolation
    order:
        order of the spline interpolation

    Returns
    -------
    array with profile values and coordinates of the points

    Notes
    -----
    https://stackoverflow.com/questions/55651307/how-to-extract-line-profile-ray-trace-line-through-3d-matrix-ndarray-3-dim-b
    I believe that can be done more elegant, but premature optimization is the source of
    all bugs, so let's leave it for now like this. Note, that map_coordinates uses
    spline interpolation. Check if that is really necessary or if we can be satisfied just with
    the closest discrete value
    """
    n_p = int(np.ceil(spatial.distance.euclidean(src, dst) / spacing))
    coords = []
    for s, e in zip(src, dst):
        coords.append(np.linspace(s, e, n_p, endpoint=False))
    return ndimage.map_coordinates(image, coords, order=order), coords


def ConnectTwoNetworks(P, pores: tuple, network=None):
    r"""
    Finds connections between two networks, with respect to the potential field

    Parameters
    ----------
    P: array_like
        potential field of shape [Nx, Ny [,Nz]]
    pores: tuple
        tuple of coordinates of each pore
    network: array_like
        if provided, it has to be a boolean array of shape [Nx, Ny [,Nz]] and the newly found
        connections will be imprinted on it

    Returns
    -------
    tuple of connectivity array and coordinates of newly found constrictions

    Notes
    -----
    The connectivity between two pores in different phases does not follow the same logic as the
    network estraction within a single phase. Mainly because no such a connection is not
    dictated by a constriction. Here, the connection is established via a direct line between
    two pores. Connections are established, if along the connecting axis from solid ($\beta < 0 $)
    to fluid phase (\beta > 0) the potential field is monotonously growing or monotonously decaying
    respectively.
    This is determined via the second derivative along the connecting axis.
    """
    num_pores = pores[0].shape[0]
    dim = len(pores)
    # create connection matrix with all pores in different phases
    # then filter according to gradient in lines
    # note, that we only need to look starting from the one phase
    # the other phase we can skip
    A = np.full((num_pores, num_pores), fill_value=False)
    mask = P[pores] < 0
    A[mask, :] = ~mask
    A = sparse.csr_array(A)

    # loop through each pore (row) and test if any of the other pores (col)
    # satisfy the connection criteria:
    #  - dP/dl > 0  --> slope needs to go up
    # todo: A vicinity criterial may allow significant filtering
    #       do some profiling and check!
    conn = np.full((A.getnnz(), 2), fill_value=-1, dtype=int)
    constr = np.full((A.getnnz(), dim), fill_value=-1, dtype=int)
    p_loc = np.transpose(np.asarray(pores))
    for row in range(num_pores):
        for pos in range(A.indptr[row], A.indptr[row + 1]):
            col = A.indices[pos]
            profile, coords = profile_line(P, p_loc[row, :], p_loc[col, :])
            if np.all(profile[1:] > profile[:-1]):
                # make sure, the second derivative is >0 for P<0 and <0 for P>0
                d2 = profile[:-2] - 2 * profile[1:-1] + profile[2:]
                p_constr = np.argmax(profile > 0)
                valid = True
                if p_constr > 3:
                    valid &= np.all(d2[:p_constr-2] > 0)
                if p_constr < d2.size:
                    valid &= np.all(d2[p_constr:] < 0)
                if not valid:
                    continue
                coords = np.transpose(np.asarray(coords))
                coords = np.array(coords)[p_constr, :]
                conn[pos, :] = [row, col]
                constr[pos, :] = coords

    # remove all those connections, which are still labeled -1
    mask = conn[:, 0] != -1
    conn = conn[mask, :]
    constr = constr[mask, :]

    # add connections to network
    if network is not None:
        for n in range(conn.shape[0]):
            p0, p1 = [], []
            ind_p0, ind_p1 = conn[n, 0], conn[n, 1]
            for v in pores:
                p0.append(v[ind_p0])
                p1.append(v[ind_p1])
            coords = skimage.draw.line_nd(p0, p1)
            network[coords] = True

    return conn, constr


def RemoveDeadEnds(conns, constr, dead_ends):
    r"""
    removes connections and constrictions according to the dead ends

    Parameters
    ----------
    conns
        (N,2) array of connected pores
    constr
        (N, 2[3]) array with positions of the constrictions
    dead_ends
        (N,1) boolean mask with dead end connections as True
    """
    constr = np.transpose(np.asarray(constr))
    return conns[~dead_ends, :], constr[~dead_ends, :]


def DetermineBoundaryPores(P, pores, conns):
    r"""
    Determines, which pores are connected to a boundary, based on the potential field

    Parameters
    ----------
    P: array_like
        potential field
    pores: tuple
        coordinates of the pores
    conns:
        so far known connections between the pores

    Returns
    -------
    array with pore IDs which have been determined as boundary pores

    Notes
    -----
    Currently, this is rather brute force. For each pore the potential along
    a line perpendicular to each bondary is determined and the presence of any
    in between bumps detected by analysis of the second derivative.
    Potential improvements include additional filter criteria or pathfinding
    along potential ridges
    """
    if np.any(conns[:, 1] == -1):
        raise Exception('all boundary pore values need to be marked in the first column')

    dim = len(P.shape)

    # only investigate pores which are not yet labeled as boundary pores
    p_bc = conns[conns[:, 0] != -1, 1]
    mask_bc = np.full_like(p_bc, fill_value=False, dtype=bool)
    # for now brute force
    for i in range(p_bc.size):
        p_id = p_bc[i]
        src = [pores[d][p_id] for d in range(dim)]
        for d in range(dim):
            dst = [s for s in src]  # deep copy without big workarounds
            # low
            dst[d] = 0
            profile, _ = profile_line(P, src=src, dst=dst)
            d2 = profile[:-2] - 2 * profile[1:-1] + profile[2:] if profile.size > 3 else True
            mask_bc[i] |= np.all(d2 > 0) or np.all(d2 < 0)
            # high
            dst[d] = P.shape[d]-1
            profile, _ = profile_line(P, src=src, dst=dst)
            d2 = profile[:-2] - 2 * profile[1:-1] + profile[2:] if profile.size > 3 else True
            mask_bc[i] |= np.all(d2 > 0) or np.all(d2 < 0)
    return p_bc[mask_bc]


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
    Lee94 T.-C. Lee, R.L. Kashyap and C.-N. Chu, Building skeleton models via 3-D medial surface/axis thinning algorithms. Computer Vision, Graphics, and Image Processing, 56(6):462-478, 1994.  # noqa E501
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
    # potential candidate: skimage.filters.sato(P, sigmas=[5])
    # could be computationally preferable to Lee, since Lee uses an iterative algorithm
    # skeletonize the image according to Lee
    # network = skimage.morphology.skeletonize(image, method='lee').astype(bool)
    # if include_solid:
    #     network |= skimage.morphology.skeletonize(np.invert(image), method='lee').astype(bool)
    network = Skeletonize(image, include_solid=include_solid, pad_width=0)

    # finding pores and constrictions along the skeleton, depending on the potential field
    pores, constr = FindPoresAndConstrictions(P, network)

    # determine, which pores are actually connected
    conns, dead_ends = EstablishConnections(network=network, P=P, pores=pores, constr=constr)
    conns, constr = RemoveDeadEnds(conns=conns, constr=constr, dead_ends=dead_ends)

    # see here, if multiple peaks as described by Gostick 2017 are an issue with this method and
    # employ merging algorithm. Should be quite simple with distance map of the pores. Make sure,
    # that the connections and constrictions are updated accordingly and

    # in the case the solid is included, connect the two networks
    if include_solid:
        conn, cstr = ConnectTwoNetworks(network=network, P=P, pores=pores)
        conns = np.append(conns, conn, axis=0)
        constr = np.append(constr, cstr, axis=0)

    # Determine boundary pores
    p_bc = DetermineBoundaryPores(P=P, pores=pores, conns=conns)
    if (p_bc is not None) and p_bc.size > 0:
        conn = np.full((p_bc.size, 2), fill_value=-1, dtype=int)
        conn[:, 1] = p_bc
        conns = np.append(conns, conn, axis=0)

    # we may have to look for duplicates, which have to be merged, or at least mark them for now
    # so they can be processed later

    return np.nonzero(network), pores, conns


# TODOS
# - determine boundary pores correctly
# - assign properties to pores and throats
# - compare with SNOW2 and maximum ball

ldim = (40, 40)

solid = 0
fluid = 255

image = np.full(shape=ldim, fill_value=solid, dtype='uint8')

# mpim.TwoTouchingPoresInSolid(image=image)
# mpim.TwoThroatsOnePore(image=image)
mpim.FourThroatsOnePore(image=image)

# radius = ldim[0] * 0.1

# center = np.array(ldim)
# dimarr = center.copy()

# center[0], center[1], center[2] = dimarr * 0.35
# center[1] , center[2] = dimarr[1]*0.5, dimarr[2]*0.5
# center[0], center[1] = dimarr * 0.35
# center[1] = dimarr[1]*0.5
# AddBall(image, center=np.floor(center), radius=ldim[0]*0.15, value=fluid)
# center[0] = dimarr[0] * 0.65
# AddBall(image, center=np.floor(center), radius=ldim[0]*0.15, value=fluid)

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

# p1 = dimarr.copy()
# p2 = dimarr.copy()
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

network, pores, conns = GetNetworkFeatures(image, P, include_solid=True)

PlotNetworkSpider(pores=pores, conns=conns)
PlotNetwork(network, pores=pores, P=P)

print('finished')
