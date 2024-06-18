
list_1d_default = ['radius', 'diameter', 'length', 'coords']
list_2d_default = ['area']
list_3d_default = ['volume']


def ScaleNetwork(network, scale: float, list_1D=None, list_2D=None, list_3D=None, mode: str = 'add'):
    list_1d = list_1d_default
    list_2d = list_2d_default
    list_3d = list_3d_default
    if list_1D is not None:
        if mode == 'add':
            list_1d = list_1d.append(list_1D)
        elif mode == 'only':
            list_1d = list_1D
        else:
            raise ValueError(f'Unknown mode: {mode}')
    if list_2D is not None:
        if mode == 'add':
            list_2d = list_2d.append(list_2D)
        elif mode == 'only':
            list_2d = list_2D
        else:
            raise ValueError(f'Unknown mode: {mode}')
    if list_3D is not None:
        if mode == 'add':
            list_3d = list_3d.append(list_3D)
        elif mode == 'only':
            list_3d = list_3D
        else:
            raise ValueError(f'Unknown mode: {mode}')

    for key, value in network.items():
        if any(v in key for v in list_1d):
            value *= scale
        elif any(v in key for v in list_2d):
            value *= scale**2
        elif any(v in key for v in list_3d):
            value *= scale**3
    return network
