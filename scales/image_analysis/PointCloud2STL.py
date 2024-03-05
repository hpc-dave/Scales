import numpy as np
from . import TiffUtils
from . import Voxel2STL as v2s
import os


def Image2STL(in_name, levels, step_size: int = 4, verbose: bool = True, out_name=None):
    if verbose:
        print(f'Reading in image: {in_name}')
    data = TiffUtils.ReadTiffStack(in_name)
    data = np.swapaxes(data, 0, 1)
    use_generic = False
    use_level = False
    opt = {
                'step_size': step_size,
                'level': level,
                'verbose': verbose
    }
    if hasattr(levels, '__len__'):
        if out_name is None:
            use_generic = True
        elif not hasattr(out_name, '__len__'):
            use_level = True

        for n in range(levels):
            level = levels[n]
            opt['level'] = level
            if use_generic:
                oname = GetGenericSTLOutputName(in_name, level)
            elif use_level:
                oname = f'{out_name}_{level}.stl'
            else:
                oname = out_name if out_name is not None else GetGenericSTLOutputName(in_name, level)
            if verbose:
                print(f'Extraction level {level}')
            phase = np.zeros_like(data)
            phase[data == level] = 1
            v2s.Extract(phase, fname=oname, opt=opt)
    else:
        oname = out_name if out_name != None else GetGenericSTLOutputName(in_name, level)
        v2s.Extract(data, fname=oname, opt=opt)


def GetGenericSTLOutputName(fname, level):
    base, ext = os.path.splitext(fname)
    return f'{base}_{level}.stl'
