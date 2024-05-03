import numpy as np
import scipy
import scipy.sparse
import inspect


def GetLineInfo():
    r"""
    Provides information of calling point in the form: <path/to/file>: l. <line number> in <function name>
    """
    return f"{inspect.stack()[1][1]}: l.{inspect.stack()[1][2]} in {inspect.stack()[1][3]}"


def construct_grad(network: any, num_components=1, include=None):
    """
    Constructs the gradient matrix

    Args:
        network (OpenPNM network): network with geometric information
        bc (dict, optional): boundary conditions, if throats are the boundary
        format (string): identifier for the target format

    Returns:
        Gradient matrix

    Notes:
        The direction of the gradient is given by the connections specified in the network,
        mores specifically from conns[:, 0] to conns[:, 1]
    """

    conns = network['throat.conns']
    p_coord = network['pore.coords']
    dist = np.sqrt(np.sum((p_coord[conns[:, 0], :] - p_coord[conns[:, 1], :])**2, axis=1))
    weights = 1./dist
    weights = np.append(weights, -weights)
    if num_components == 1:
        return np.transpose(network.create_incidence_matrix(weights=weights, fmt='csr'))
    else:
        if include is None:
            include = range(num_components)
        num_included = len(include)

        im = np.transpose(network.create_incidence_matrix(weights=weights, fmt='coo'))
        data = np.zeros((im.data.size, num_included), dtype=float)
        rows = np.zeros((im.data.size, num_included), dtype=float)
        cols = np.zeros((im.data.size, num_included), dtype=float)

        pos = 0
        for n in include:
            rows[:, pos] = im.row * num_components + n
            cols[:, pos] = im.col * num_components + n
            data[:, pos] = im.data
            pos += 1

        rows = np.ndarray.flatten(rows)
        cols = np.ndarray.flatten(cols)
        data = np.ndarray.flatten(data)
        mat_shape = (network.Nt * num_components, network.Np * num_components)
        grad = scipy.sparse.coo_matrix((data, (rows, cols)), shape=mat_shape)
        return scipy.sparse.csr_matrix(grad)


def construct_div(network: any, weights=None, custom_weights: bool = False, num_components: int = 1):
    """
    Constructs divergence matrix

    Args:
        network (OpenPNM network): network with geometric information
        weights (array_like of size Nt or 2*Nt): reused weights for optimization, e.g. throat area
        format (string): identifier for the target format

    Returns:
        Divergence matrix
    """
    _weights = None
    if custom_weights:
        if weights is None:
            raise ('custom weights were specified, but none were provided')
        _weights = np.flatten(weights)
        if _weights.shape[0] < network.Nt*num_components*2:
            _weights = np.append(-_weights, _weights)
    else:
        _weights = np.ones(shape=(network.Nt)) if weights is None else np.ndarray.flatten(weights)
        if _weights.shape[0] == network.Nt:
            _weights = np.append(-_weights, _weights)

    if num_components == 1:
        div_mat = network.create_incidence_matrix(weights=_weights, fmt='coo')
    else:
        ones = np.ones(shape=(network.Nt*2))
        div_mat = network.create_incidence_matrix(weights=ones, fmt='coo')
        data = np.zeros((div_mat.data.size, num_components), dtype=float)
        rows = np.zeros((div_mat.data.size, num_components), dtype=float)
        cols = np.zeros((div_mat.data.size, num_components), dtype=float)
        for n in range(num_components):
            rows[:, n] = div_mat.row * num_components + n
            cols[:, n] = div_mat.col * num_components + n
            if custom_weights:
                beg, end = n * network.Nt * 2, (n + 1) * network.Nt * 2 - 1
                data[:, n] = _weights[beg: end]
            else:
                data[:, n] = _weights
        rows = np.ndarray.flatten(rows)
        cols = np.ndarray.flatten(cols)
        data = np.ndarray.flatten(data)
        mat_shape = (network.Np * num_components, network.Nt * num_components)
        div_mat = scipy.sparse.coo_matrix((data, (rows, cols)), shape=mat_shape)

    # converting to CSR format for improved computation
    div_mat = scipy.sparse.csr_matrix(div_mat)

    def div(*args):
        fluxes = args[-1]
        for i in range(len(args)-1):
            if isinstance(args[i], list) and len(args[i]) == num_components:
                fluxes.multiply(np.tile(np.asarray(args[i]), network.Nt))
            elif isinstance(args[i], np.ndarray):
                fluxes = fluxes.multiply(args[i].reshape(args[i].size, 1))
            else:
                fluxes = fluxes.multiply(args[i])
        return div_mat * fluxes

    return div


def construct_upwind(network, fluxes, num_components: int = 1, include=None):
    r"""
    Constructs a [Nt, Np] matrix representing a directed network based on the upwind
    fluxes

    Parameters
    ----------
    network : any
        OpenPNM network instance
    fluxes : any
        fluxes which determine the upwind direction, see below for more details
    num_components : int
        number of components the matrix is constructed for
    include : list
        a list of integers to specify for which components the matrix should be constructed,
        for components which are not listed here, the rows will be 0. If 'None' is provided,
        all components will be selected

    Returns
    -------
    A [Nt, Np] sized CSR-matrix representing a directed network

    Notes
    -----
    The direction of the fluxes is directly linked with the storage of the connections
    inside the OpenPNM network. For more details, refer to the 'create_incidence_matrix' method
    of the network module.
    The resulting matrix IS NOT SCALED with the fluxes and can also be used for determining
    upwind interpolated values.
    The provided fluxes can either be:
        int/float - single value
        list/numpy.ndarray - with size num_components applies the values to each component separately
        numpy.ndarray - with size Nt applies the fluxes to each component by throat: great for convection
        numpy.ndarray - with size Nt * num_components is the most specific application for complex
                        multicomponent coupling, where fluxes can be opposed to each other within
                        the same throat
    """

    if num_components == 1:
        weights = np.append(-(fluxes < 0).astype(float), fluxes > 0)
        return np.transpose(network.create_incidence_matrix(weights=weights, fmt='csr'))
    else:
        if include is None:
            include = range(num_components)
        num_included = len(include)

        im = np.transpose(network.create_incidence_matrix(fmt='coo'))

        data = np.zeros((im.data.size, num_included), dtype=float)
        rows = np.zeros((im.data.size, num_included), dtype=int)
        cols = np.zeros((im.data.size, num_included), dtype=int)

        pos = 0
        for n in include:
            rows[:, pos] = im.row * num_components + n
            cols[:, pos] = im.col * num_components + n
            data[:, pos] = im.data
            pos += 1

        if isinstance(fluxes, float) or isinstance(fluxes, int):
            # single provided value
            _fluxes = np.zeros((network.Nt)) + fluxes
            weights = np.append(_fluxes < 0, _fluxes > 0)
            pos = 0
            for n in include:
                data[:, pos] = weights
                pos += 1
        elif (isinstance(fluxes, list) and len(fluxes) == num_components)\
                or (isinstance(fluxes, np.ndarray) and fluxes.size == num_components):
            # a list of values for each component
            _fluxes = np.zeros((network.Nt))
            pos = 0
            for n in include:
                _fluxes[:] = fluxes[n]
                weights = np.append(_fluxes < 0, _fluxes > 0)
                data[:, pos] = weights
                pos += 1
        elif fluxes.size == network.Nt:
            # fluxes for each throat, e.g. for single component or same convective fluxes
            # for each component
            weights = np.append(fluxes < 0, fluxes > 0)
            pos = 0
            for n in include:
                data[:, pos] = weights.reshape((network.Nt*2))
                pos += 1
        elif (len(fluxes.shape)) == 2\
            and (fluxes.shape[0] == network.Nt)\
                and (fluxes.shape[1] == num_components):
            # each throat has different fluxes for each component
            pos = 0
            for n in include:
                weights = np.append(fluxes[:, n] < 0, fluxes[:, n] > 0)
                data[:, pos] = weights.reshape((network.Nt*2))
                pos += 1
        else:
            raise ('fluxes have incompatible dimension')

        rows = np.ndarray.flatten(rows)
        cols = np.ndarray.flatten(cols)
        data = np.ndarray.flatten(data)
        mat_shape = (network.Nt * num_components, network.Np * num_components)
        upwind = scipy.sparse.coo_matrix((data, (rows, cols)), shape=mat_shape)
        return scipy.sparse.csr_matrix(upwind)


def construct_ddt(network, dt: float, num_components: int = 1, weight='pore.volume'):
    if dt <= 0.:
        raise (f'timestep is invalid, following constraints were violated: {dt} !> 0')
    if num_components < 1:
        raise (f'number of components has to be positive, following value was provided: {num_components}')

    Nc = num_components
    dVdt = network[weight]/dt
    dVdt = dVdt.reshape((dVdt.shape[0], 1))
    if Nc > 1:
        dVdt = np.tile(A=dVdt, reps=Nc)
    ddt = scipy.sparse.spdiags(data=[dVdt.ravel()], diags=[0])
    return ddt


def ApplyBC(network, bc, A, rhs=None):
    num_pores = network.Np
    num_rows = A.shape[0]
    if rhs is None:
        rhs = np.zeros((num_pores, 1), dtype=float)

    if (num_rows % num_pores) != 0:
        raise (f'the number of matrix rows now not consistent with the number of pores,\
               mod returned {num_rows % num_pores}')
    if num_rows != rhs.shape[0]:
        raise ('Dimension of rhs and matrix inconsistent!')

    num_components = int(num_rows / num_pores)
    if num_components > 1:
        raise ('Error, cannot deal with more than one component at the moment!')

    for label, param in bc.items():
        bc_pores = network.pores(label)
        row_aff = bc_pores * num_components
        if 'prescribed' in param:
            rhs[row_aff] = param['prescribed']
            A[row_aff, :] = 0.
            A[row_aff, row_aff] = 1.
        elif 'rate' in param:
            rhs[row_aff] = -param['rate']
        else:
            raise (f'Provided boundary condition cannot be applied for {label}')

    A.eliminate_zeros()
    return A, rhs


def EnforcePrescribed(network, bc, A, x, b, type='Jacobian'):
    if len(bc) == 0:
        print(f'{GetLineInfo()}: No boundary conditions were provided, consider removing function altogether!')
        return A, b

    num_pores = network.Np
    num_rows = A.shape[0]
    num_components = int(num_rows/num_pores)
    if (num_rows % num_pores) != 0:
        raise (f'the number of matrix rows now not consistent with the number of pores,\
               mod returned {num_rows % num_pores}')
    if num_rows != b.shape[0]:
        raise ('Dimension of rhs and matrix inconsistent!')

    if isinstance(bc, dict) and isinstance(list(bc.keys())[0], int):
        bc = list(bc)
    elif not isinstance(bc, list):
        bc = [bc]

    for n_c, boundary in enumerate(bc):
        for label, param in boundary.items():
            bc_pores = network.pores(label)
            row_aff = bc_pores * num_components + n_c
            if 'prescribed' in param:
                if type == 'Jacobian':
                    b[row_aff] = x[row_aff] - param['prescribed']
                else:
                    b[row_aff] = param['prescribed']
                if scipy.sparse.isspmatrix_csr(A):
                    # optimization for csr matrix (avoid changing the sparsity structure)
                    # benefits are memory and speedwise
                    for r in row_aff:
                        ptr = (A.indptr[r], A.indptr[r+1])
                        A.data[ptr[0]:ptr[1]] = 0.
                        pos = np.where(A.indices[ptr[0]: ptr[1]] == r)[0]
                        A.data[ptr[0] + pos[0]] = 1.
                else:
                    A[row_aff, :] = 0.
                    A[row_aff, row_aff] = 1.

    A.eliminate_zeros()
    return A, b
