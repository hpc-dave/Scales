import openpnm as op
import numpy as np
import math
from warnings import warn


def ExtractSubsection(network, begin, end):
    range_beg = np.asarray(begin)
    range_end = np.asarray(end)
    coords = network['pore.coords']
    p_sub = coords >= range_beg
    p_sub &= coords <= range_end
    return np.all(p_sub, axis=1)


def DeletePoresAndLabelBoundary(network, pores_delete, label: str):
    temp_label = 'merge_for_delete'
    p_delete = np.asarray(pores_delete)
    if p_delete.size > 0:
        op.topotools.merge_pores(network=network, pores=p_delete, labels=[temp_label])
        p_merged = network.pores(temp_label)
        p_bc = network.find_neighbor_pores(pores=p_merged)
        network.set_label(pores=p_bc, label=label)
        op.topotools.trim(network=network, pores=p_merged)
        network.set_label(label=temp_label, mode='purge')
    else:
        warn('an empty list was provided for deleting pores, doing nothing now')
    return network


def ExtractPackingAlongAxis(network, axis: int, labels, range):
    if not isinstance(range, list) or len(range) != 2:
        raise TypeError('range needs to be a list of size 2')
    if range[0] > range[1]:
        range[0], range[1] = range[1], range[0]

    dim = network['pore.coords'].shape[1]
    lim_low = np.full((dim), fill_value=math.inf, dtype=float)
    lim_high = np.full((dim), fill_value=-math.inf, dtype=float)
    lim_low[axis], lim_high[axis] = range

    p_section = ExtractSubsection(network, begin=np.full_like(lim_low, fill_value=-math.inf), end=lim_low)
    p_section = np.where(p_section)[0]
    DeletePoresAndLabelBoundary(network=network, pores_delete=p_section, label=labels[0])
    p_section = ExtractSubsection(network, begin=lim_high, end=np.full_like(lim_high, fill_value=math.inf))
    p_section = np.where(p_section)[0]
    DeletePoresAndLabelBoundary(network=network, pores_delete=p_section, label=labels[1])
    return network


def AddPoreTrain(network, coords, pore_labels, throat_labels, properties, label_connection: str):

    num_pores = None
    coord = None
    props = [properties] if isinstance(properties, dict) else properties
    labels_pore = [pore_labels] if isinstance(pore_labels, str) else pore_labels
    labels_throat = [throat_labels] if isinstance(throat_labels, str) else throat_labels
    if isinstance(coords, list):
        if not isinstance(coords[0], list) and not isinstance(coords[0], np.ndarray):
            coord = [coords]
        else:
            coord = coords
        num_pores = len(coord)
    else:
        raise TypeError('The provided coordinates need to be either a single list of coordinates or a list of those')

    if isinstance(props, list):
        if len(props) != num_pores:
            raise ValueError('The number of properties is inconsistent with the number of coordinates')
    else:
        raise TypeError('the properties need to be provided as list')

    label_conn = label_connection
    for n in range(num_pores):
        p_label = network.num_pores()
        label_new = labels_pore[n]
        conns = np.zeros((network.pores(label_conn).size, 2), dtype=int)
        conns[:, 0] = network.pores(label_conn)
        conns[:, 1] = p_label
        op.topotools.extend(network=network, coords=coord[n], conns=conns, labels=[label_new])
        t_label = network.find_neighbor_throats(pores=[p_label])
        if labels_throat is not None and labels_throat[n] is not None:
            network.set_label(label=labels_throat[n], throats=t_label)
        for key, value in props[n].items():
            if 'pore' in key:
                network[key][p_label] = value
            elif 'throat' in key:
                network[key][t_label] = value
        label_conn = label_new


if __name__ == '__main__':
    network = op.network.Cubic((10, 3, 3))
    network.add_model_collection(op.models.collections.geometry.spheres_and_cylinders)
    network.regenerate_models()
    network = ExtractPackingAlongAxis(network=network, axis=0, labels=['lower', 'upper'], range=[1., 5.])

    print('finished')
