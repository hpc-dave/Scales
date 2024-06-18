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


if __name__ == '__main__':
    network = op.network.Cubic((10, 3, 3))
    network.add_model_collection(op.models.collections.geometry.spheres_and_cylinders)
    network.regenerate_models()
    network = ExtractPackingAlongAxis(network=network, axis=0, labels=['lower', 'upper'], range=[1., 5.])

    print('finished')
