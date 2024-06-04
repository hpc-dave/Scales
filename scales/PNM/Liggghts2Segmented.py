import os
import sys
import numpy as np
import skimage
from tqdm import tqdm
import ModelPorousImage as mp

dir = os.path.dirname(os.path.realpath(__file__))
dir = os.path.dirname(dir)
sys.path.append(dir + os.path.sep + 'IO')

import ReadLiggghts as rl   # noqa: E402


def _get_bounding_box_sphere(center, radius, safety: int = 2):
    bb_low = np.asarray(center).copy()-radius
    bb_high = np.asarray(center).copy()+radius
    bb_low = np.asarray(bb_low, dtype=int) - safety
    bb_high = np.asarray(bb_high, dtype=int) + safety
    return bb_low, bb_high


def run(file_in: str, file_out: str, system_size, resolution, tube_wall = None, offset=None):
    x, y, z, r = rl.Read(filename=file_in, extract_columns=['x', 'y', 'z', 'radius'])

    dim = len(system_size)
    dl = [float(e[1])-float(e[0]) for e in system_size]
    dl_min = np.min(dl)
    dn = dl_min/resolution
    res_im = tuple(int(dl[n]/dn) for n in range(dim))

    image = np.zeros(res_im, dtype=bool)
    if offset is None:
        offset = [-system_size[n][0] for n in range(dim)]
    if len(offset) != dim:
        raise ValueError('offset dimensions are not consistent')

    num_obj = x.size
    bb_low = np.zeros((num_obj, dim), dtype=int)
    bb_high = np.zeros_like(bb_low)
    center = np.zeros_like(bb_low)
    radii = np.zeros((num_obj), dtype=int)
    for n in range(num_obj):
        center[n, :] = np.asarray([(x[n]+offset[0])/dn, (y[n]+offset[1])/dn, (z[n]+offset[2])/dn])
        radii[n] = r[n]/dn
        bb_low[n, :], bb_high[n, :] = _get_bounding_box_sphere(center=center[n, :], radius=radii[n])

    for n in tqdm(range(num_obj)):
        mp.AddBall(image=image,
                   center=[(x[n]+offset[0])/dn, (y[n]+offset[1])/dn, (z[n]+offset[2])/dn], radius=r[n]/dn,
                   value=True,
                   bb_l=bb_low[n, :],
                   bb_h=bb_high[n, :])

    if tube_wall is not None:
        dir = int(tube_wall['direction'])
        dim_cross=[d for d in range(dim) if d != dir]
        dir_0 = dim_cross[0]
        dir_1 = dim_cross[1]
        d_cross = np.zeros((image.shape[dir_0], image.shape[dir_1], 2), dtype=float)
        pc = np.asarray([tube_wall['center'][dir_0]/dn, tube_wall['center'][dir_1]/dn], dtype=float)
        radius = float(tube_wall['radius']/dn)

        for i in range(len(dim_cross)):
            shape_l = np.ones_like(d_cross.shape[:-1])
            shape_l[i] = np.asarray(d_cross.shape[i])
            l_v = np.reshape((np.array(range(d_cross.shape[i])) - pc[i])**2, shape_l)
            shape_l = np.asarray(d_cross.shape[:-1])
            shape_l[i] = 1
            d_cross[..., i] = np.tile(l_v, shape_l)

        mask = (np.sum(d_cross, axis=-1)) > radius**2
        mask = np.expand_dims(mask, axis=dir)
        mask = np.tile(mask, [1 if d != dir else image.shape[d] for d in range(dim)])
        image[mask] = True

    skimage.io.imsave(file_out, image)


# filename = '/home/drieder/Data/cylinder1000000.liggghts'
# imagename = '/home/drieder/Data/cylinder1000000.tiff'
# tube = {'direction': 2, 'radius': 0.5, 'center': (0.5, 0.5, 0.)}
# run(file_in=filename, file_out=imagename, system_size=[[-0.5, 0.5], [-0.5, 0.5], [0., 100.]], resolution=100, tube_wall=tube)
