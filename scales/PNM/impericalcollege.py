import logging
import numpy as np
from openpnm.topotools import trim
from openpnm.network import Network
from pathlib import Path
from pandas import read_table, DataFrame

logger = logging.getLogger(__name__)


def network_from_statoil(path, prefix, scale: float = 1):
    r"""
    Load data from the \'dat\' files located in specified folder.

    Parameters
    ----------
    path : str
        The full path to the folder containing the set of \'dat\' files.
    prefix : str
        The file name prefix on each file. The data files are stored
        as \<prefix\>_node1.dat.
    network : Network
        If given then the data will be loaded on it and returned.  If not
        given, a Network will be created and returned.
    scale : float
        Scaling of the imported data, linear with length. Areas and Volumes are adapted
        with **2 and **3 respectively

    Returns
    -------
    Project
        An OpenPNM Project containing a Network holding all the data

    Notes
    -----
    The StatOil format is used by the Maximal Ball network extraction code of
    the Imperial College London group

    This class can be used to load and work with those networks. Numerous
    datasets are available for download from the group's
    `website <http://tinyurl.com/zurko4q>`_.

    The 'Statoil' format consists of 4 different files in a single
    folder. The data is stored in columns with each corresponding to a
    specific property. Headers are not provided in the files, so one must
    refer to various theses and documents to interpret their meaning.

    Note: This is a fixed version of the OpenPNM function, which is
    unfortunately incompatible with the standard way of using openpnm

    """
    net = {}

    # Parse the link1 file
    path = Path(path)
    filename = Path(path.resolve(), prefix+'_link1.dat')
    with open(filename, mode='r') as f:
        link1 = read_table(filepath_or_buffer=f,
                           header=None,
                           skiprows=1,
                           sep=' ',
                           skipinitialspace=True,
                           index_col=0)
    link1.columns = ['throat.pore1', 'throat.pore2', 'throat.radius',
                     'throat.shape_factor', 'throat.total_length']
    # Add link1 props to net
    net['throat.conns'] = np.vstack((link1['throat.pore1']-1,
                                     link1['throat.pore2']-1)).T
    net['throat.conns'] = np.sort(net['throat.conns'], axis=1)
    net['throat.radius'] = np.array(link1['throat.radius']) * scale
    net['throat.shape_factor'] = np.array(link1['throat.shape_factor'])
    net['throat.total_length'] = np.array(link1['throat.total_length']) * scale

    filename = Path(path.resolve(), prefix+'_link2.dat')
    with open(filename, mode='r') as f:
        link2 = read_table(filepath_or_buffer=f,
                           header=None,
                           sep=' ',
                           skipinitialspace=True,
                           index_col=0)
    link2.columns = ['throat.pore1', 'throat.pore2',
                     'throat.pore1_length', 'throat.pore2_length',
                     'throat.length', 'throat.volume',
                     'throat.clay_volume']
    # Add link2 props to net
    cl_t = np.array(link2['throat.length']) * scale
    net['throat.length'] = cl_t 
    net['throat.conduit_lengths.throat'] = cl_t
    net['throat.volume'] = np.array(link2['throat.volume']) * scale**3
    cl_p1 = np.array(link2['throat.pore1_length']) * scale
    net['throat.conduit_lengths.pore1'] = cl_p1
    cl_p2 = np.array(link2['throat.pore2_length']) * scale
    net['throat.conduit_lengths.pore2'] = cl_p2
    net['throat.clay_volume'] = np.array(link2['throat.clay_volume']) * scale**3
    # ---------------------------------------------------------------------
    # Parse the node1 file
    filename = Path(path.resolve(), prefix+'_node1.dat')
    with open(filename, mode='r') as f:
        row_0 = f.readline().split()
        num_lines = int(row_0[0])
        array = np.ndarray([num_lines, 6])
        for i in range(num_lines):
            row = f.readline()\
                   .replace('\t', ' ').replace('\n', ' ').split()
            array[i, :] = row[0:6]
    node1 = DataFrame(array[:, [1, 2, 3, 4]])
    node1.columns = ['pore.x_coord', 'pore.y_coord', 'pore.z_coord',
                     'pore.coordination_number']
    # Add node1 props to net
    net['pore.coords'] = np.vstack((node1['pore.x_coord'],
                                    node1['pore.y_coord'],
                                    node1['pore.z_coord'])).T * scale
    # ---------------------------------------------------------------------
    # Parse the node2 file
    filename = Path(path.resolve(), prefix+'_node2.dat')
    with open(filename, mode='r') as f:
        node2 = read_table(filepath_or_buffer=f,
                           header=None,
                           sep=' ',
                           skipinitialspace=True,
                           index_col=0)
    node2.columns = ['pore.volume', 'pore.radius', 'pore.shape_factor',
                     'pore.clay_volume']
    # Add node2 props to net
    net['pore.volume'] = np.array(node2['pore.volume']) * scale**3
    net['pore.radius'] = np.array(node2['pore.radius']) * scale
    net['pore.shape_factor'] = np.array(node2['pore.shape_factor'])
    net['pore.clay_volume'] = np.array(node2['pore.clay_volume']) * scale**3
    net['throat.cross_sectional_area'] = ((net['throat.radius']**2)
                                          / (4.0*net['throat.shape_factor']))
    net['pore.area'] = ((net['pore.radius']**2)
                        / (4.0*net['pore.shape_factor']))

    network = Network()
    network.update(net)

    # Use OpenPNM Tools to clean up network
    # Trim throats connected to 'inlet' or 'outlet' reservoirs
    trim1 = np.where(np.any(net['throat.conns'] == -1, axis=1))[0]
    # Apply 'outlet' label to these pores
    outlets = network['throat.conns'][trim1, 1]
    network['pore.outlet'] = False
    network['pore.outlet'][outlets] = True
    trim2 = np.where(np.any(net['throat.conns'] == -2, axis=1))[0]
    # Apply 'inlet' label to these pores
    inlets = network['throat.conns'][trim2, 1]
    network['pore.inlet'] = False
    network['pore.inlet'][inlets] = True
    # Now trim the throats
    to_trim = np.hstack([trim1, trim2])
    trim(network=network, throats=to_trim)

    return network


def valvatne_blunt(
    phase,
    pore_viscosity="pore.viscosity",
    throat_viscosity="throat.viscosity",
    pore_shape_factor="pore.shape_factor",
    throat_shape_factor="throat.shape_factor",
    pore_area="pore.area",
    throat_area="throat.cross_sectional_area",
    conduit_lengths="throat.conduit_lengths",
):
    r"""
    Calculates the single phase hydraulic conductance of conduits.

    Function has been adapted for use with the Statoil imported networks
    and makes use of the shape factor in these networks to apply
    Hagen-Poiseuille flow for conduits of different shape classes:
    triangular, square and circular [2].

    Parameters
    ----------
    %(phase)s
    pore_viscosity : str
        %(dict_blurb)s pore viscosity
    throat_viscosity : str
        %(dict_blurb)s throat viscosity
    pore_shape_factor : str
        %(dict_blurb)s pore geometric shape factor
    throat_shape_factor : str
        %(dict_blurb)s throat geometric shape factor
    pore_area : str
        %(dict_blurb)s pore area
        The pore area is calculated using following formula:

        .. math::

            A_P = \frac{R_P^2}{(4 \cdot SF_P)}

        where theoratical value of pore_shape_factor in a circular tube is
        calculated using following formula:

        .. math::

            SF_P = \frac{A_P}{P_P^2} = 1/4π

    throat_area : str
        %(dict_blurb)s throat area.
        The throat area is calculated using following formula:

        .. math::

            T_A = \frac{R_T^2}{(4 \cdot SF_T)}

        where theoratical value of throat shape factor in circular tube is
        calculated using :

        .. math::

            SF_T = \frac{T_A}{T_P^2} = 1/4π

    conduit_lengths : str
        %(dict_blurb)s throat conduit lengths

    Returns
    -------
    %(return_arr)s

    References
    ----------
    [1] Valvatne, Per H., and Martin J. Blunt. "Predictive pore‐scale
    modeling of two‐phase flow in mixed wet media." Water Resources
    Research 40, no. 7 (2004).

    [2] Patzek, T. W., and D. B. Silin (2001), Shape factor and hydraulic
    conductance in noncircular capillaries I. One-phase creeping flow,
    J. Colloid Interface Sci., 236, 295–304.

    Note, this is a fixed version fo the implementation in OpenPNM
    """
    network = phase.network
    conns = network["throat.conns"]
    mu_p = phase[pore_viscosity]
    mu_t = phase[throat_viscosity]

    # Bugfix
    # Lt, L2, L2 = network[conduit_lengths]
    Lt = network[f'{conduit_lengths}.throat']
    L1 = network[f'{conduit_lengths}.pore1']
    L2 = network[f'{conduit_lengths}.pore2']

    Gp = network[pore_shape_factor]
    Gt = network[throat_shape_factor]
    Ap = network[pore_area]
    At = network[throat_area]

    # Throat portions
    tri = Gt <= np.sqrt(3) / 36.0
    circ = Gt >= 0.07
    square = ~(tri | circ)
    ntri = np.sum(tri)
    nsquare = np.sum(square)
    ncirc = np.sum(circ)
    kt = np.ones_like(Gt)
    kt[tri] = 3.0 / 5.0
    kt[square] = 0.5623
    kt[circ] = 0.5

    # Pore portions
    tri = Gp <= np.sqrt(3) / 36.0
    circ = Gp >= 0.07
    square = ~(tri | circ)
    ntri += np.sum(tri)
    nsquare += np.sum(square)
    ncirc += np.sum(circ)
    kp = np.ones_like(Gp)
    kp[tri] = 3.0 / 5.0
    kp[square] = 0.5623
    kp[circ] = 0.5

    # Calculate conductance values
    gp = kp * Ap**2 * Gp / mu_p
    gt = kt * At**2 * Gt / mu_t
    value = L1 / gp[conns[:, 0]] + Lt / gt + L2 / gp[conns[:, 1]]

    return 1 / value
