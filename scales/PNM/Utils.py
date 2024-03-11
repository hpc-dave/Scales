# PNMUtils
import numpy as np
import openpnm as op
import os
import vtk
import sys
from ..image_analysis import TiffUtils
import imperialcollege as ic


def CheckSystemSettings():
    """
    Quick hack to allow for redirecting with powershell,
    see https://stackoverflow.com/questions/75951299/wrong-encoding-when-redirecting-printed-unicode-characters-on-windows-powershell    # noqa: E501
    for more details
    """
    is_redirected = not sys.stdout.isatty()
    if is_redirected:
        is_power_shell = len(os.getenv('PSModulePath', '').split(os.pathsep)) >= 3
        if is_power_shell:
            sys.stdout.reconfigure(encoding='utf-16')
        # else:
        #     sys.stdout.reconfigure(encoding='utf-8')


def ReadPNExtract(fname: str, silent: bool = True):
    """
    Reads in a file in the statoil format and returns a 2D list of strings.
    The first dimension refers to a line, whereas the second stores each whitespace
    separated string
    input:
    fname(str): path with file stub of statoil format
    silent(bool): suppresses output
    """
    sep = ''
    _, fext = os.path.splitext(fname)
    if (fext == '.csv'):
        sep = ','

    if not silent:
        print('Reading: ' + fname)

    with open(fname, 'r') as f:
        data = f.read()

    # split it into lines
    data = data.splitlines()

    # clean whitespaces and substitute with comma
    if sep == '':
        for i in range(len(data)):
            data[i] = data[i].split()
    else:
        for i in range(len(data)):
            data[i] = data[i].split(sep)

    return data


def WriteToCSV(data, fname: str, silent=True):
    """
    Appends data to CSV-file
    input:
    data: array with data to write, each entry is written in a new line
    fname: file to write to
    silent(bool): suppresses output
    """
    # concatenate the data with comma
    for i in range(len(data)):
        data[i] = ','.join(data[i])

    if not silent:
        print('Writing to: ' + fname)

    with open(fname, 'a') as f:
        for d in data:
            print(d, file=f)


def ConvertPNExtractToCSV(base_fname, silent=True):
    """
    Converts the .dat files in Statoil format to CSV files
    base_fname -- base name of the files, without the appendices for links and nodes
                    e.g. if the files are 'mynetwork_link1.dat', 'mynetwork_node1.dat', ...
                    then 'mynetwork' needs to be provided as string. Note, that the base_fname
                    may also include the whole path to the files, if they are not located in the
                    same folder as the script
    """
    # read in the whole data set
    filenames = [base_fname + '_link1.dat',
                 base_fname + '_link2.dat',
                 base_fname + '_node1.dat',
                 base_fname + '_node2.dat']

    # reading in a writing immediately
    for filename in filenames:
        data = ReadPNExtract(filename, silent)

        # printing to file
        filename_out = filename.rsplit('.', 1)[0] + '.csv'
        WriteToCSV(data, fname=filename_out, silent=silent)


def ReadPoresPNExtract(path: str, ext: str, silent=True):
    """
    reads pore data in
    """
    # reading pore positions
    fname = path + '_node1.' + ext
    data = ReadPNExtract(fname, silent=silent)
    # number of pores is first argument in node1 file
    num_pores = int(data[0][0])
    data = data[1:]
    coords = np.zeros((num_pores, 3))
    for i in range(num_pores):
        coords[i][0] = float(data[i][1])
        coords[i][1] = float(data[i][2])
        coords[i][2] = float(data[i][3])
    # reading pore diameters
    fname = path + '_node2.' + ext
    data = ReadPNExtract(fname, silent=silent)
    dpore = np.zeros(num_pores, dtype=float)
    vpore = np.zeros(num_pores, dtype=float)
    shape_pore = np.zeros(num_pores, dtype=float)
    for i in range(num_pores):
        vpore[i] = float(data[i][1])
        dpore[i] = float(data[i][2])
        shape_pore[i] = float(data[i][3])

    dpore *= 2  # stored are radii, which are converted to diameter here
    return coords, dpore, vpore, shape_pore


def ReadThroatsPNExtract(path: str, ext: str, silent=True):
    fname = path + '_link1.' + ext
    data = ReadPNExtract(fname, silent=silent)
    # number of throats is first argument in link1 file
    num_throats = int(data[0][0])

    data = data[1:]

    conns = np.zeros((num_throats, 2), dtype=int)
    dthroat = np.zeros(num_throats, dtype=float)
    shape_throat = np.zeros(num_throats, dtype=float)
    ONE = int(1)    # pnextract starts counting at 1, which we need to correct here
    for i in range(num_throats):
        conns[i][0] = int(data[i][1]) - ONE
        conns[i][1] = int(data[i][2]) - ONE
        dthroat[i] = float(data[i][3])
        shape_throat[i] = float(data[i][4])
    dthroat *= 2    # same as pores, it stores radii, so we convert it to diameter
    conns[conns[:, 0] < -1, 0] = -1
    conns[conns[:, 1] < -1, 0] = -1
    return conns, dthroat, shape_throat


def FilterPores(network, prop):
    """
    Applies a filter to the pores and returns list with pores for removal
    network: complete network information
    prop: dictionary with properties of the filter
    EVERY dictionary requires the key 'active' which can be either true or false,
    in the case of false, no filter will be applied
    A filter also needs to have a key 'type', to identify the filter. Currently,
    'Phase' and 'Custom' are supported:
    Phase --> filter als pores which are NOT inside the specified phase
                requires following fields:
                PhaseID   - phase which contains the network : string    OR
                FilterCriteria - Criteria to apply for the image filter,
                                 should return true if the value is valid, false if should be removed
                                 If PhaseID is located in the dict, it will overwrite any other filter
                                 : anonymous function of signature (int) : bool
                ImagePath - path to the image on which the filter is based (assuming integers) : string
                optional:
                Scale     - scale which was applied to the provided coordinates in contrast to the image
                            coordinate values will be divided by this value! : float
                AxisSwitch - switch image axes, e.g. in the case that the network is arranged along the x-axis
                             whereas the original image had the z-axis as main direction : [int, int]
    Custom--> a custom filter can be provided by adding a key 'Filter', which
                is a function object with signature Filter(network) : list(porelabels)
    """
    if not prop['active']:
        return []

    print('Starting filtering process')
    if prop['type'] == 'Phase':
        if 'PhaseID' in prop:
            phase_id = prop['PhaseID']
            prop['FilterCriteria'] = lambda v: v == phase_id
        filter = prop['FilterCriteria']

        image_path = prop['ImagePath']
        image = TiffUtils.ReadTiffStack(image_path)
        # the tiffs have x and y-axis switched, correcting that here
        image = np.swapaxes(image, 0, 1)
        bounds = image.shape - np.array((1, 1, 1))
        coords = network.coords.copy()
        scale = prop['Scale'] if 'Scale' in prop else 1
        coords /= scale
        coords = coords.astype(int)
        if 'AxisSwitch' in prop:
            axes = prop['AxisSwitch']
            # be aware! numpy provides views of the data and we have to explictly copy
            # one array to make it work!
            coords[:, axes[0]], coords[:, axes[1]] = coords[:, axes[1]], coords[:, axes[0]].copy()
        p_labels = []
        for n in range(len(coords)):
            i, j, k = coords[n]
            if i > bounds[0] or j > bounds[1] or k > bounds[2]:
                continue
            if not (filter(image[i, j, k])):
                p_labels.append(n)
    elif prop['type'] == 'Custom':
        p_labels = prop['Filter'](network)
    else:
        raise Exception('filter type ' + str(prop['type']) + ' is not known!')

    if 'WriteVTK' in prop:
        if prop['WriteVTK']:
            coords = network.coords.copy()[p_labels, :]
            dpore = network['pore.diameter'].copy()[p_labels]
            if 'VTKOutputRescaleToImage' in prop:
                coords = coords/scale if prop['VTKOutputRescaleToImage'] else coords
            vtkfname = prop['VTKOutputPath']+'_filtered.vtk'
            print('Writing ' + str(coords.shape[0]) + ' pores to ' + vtkfname)
            WritePoresToVTK(filename=vtkfname, dpore=dpore, coords=coords, quality=20)

    return p_labels


def ImportPNExtractToOpenPNM(path: str, prefix: str, scale: float = 1,
                             axis: int = 0, silent: bool = True, dict_filter={'active': False}):
    if not path:
        raise ValueError('the path to the files is empty, cannot read anything!')

    if silent:
        printIf = lambda pstr: None     # noqa: E731
    else:
        printIf = lambda pstr: print(pstr)  # noqa: E731

    options = {}
    options['scale'] = scale
    if dict_filter.boundaries:
        options['boundaries'] = dict_filter.boundaries
    pn = ic.network_from_statoil(path=path, prefix=prefix, options=options)

    # coords, dpore, vpore, shape_pore = ReadPoresPNExtract(path, ext, silent=silent)
    # num_pores = len(coords[:, 0])
    # conns, dthroat, shape_throat = ReadThroatsPNExtract(path, ext, silent)

    # # pnextract assigns a value of -1 to boundary throats
    # # however, openpnm assigns BCs to pores, so we need to label those instead
    # mask_1 = conns[:, 0] == -1
    # mask_2 = conns[:, 1] == -1

    # pores_bc = np.append(conns[mask_1, 1], conns[mask_2, 0])
    # pores_bc = np.unique(pores_bc)

    # mask = mask_1 | mask_2

    # conns = conns[~mask, :]
    # dthroat = dthroat[~mask]

    # # conduct scaling if necessary
    # dthroat *= scale
    # dpore *= scale
    # coords *= scale
    # vpore *= scale**3

    # # create the network
    # pn = op.network.Network(coords=coords, conns=conns)
    # pn['pore.diameter'] = dpore
    # pn['throat.diameter'] = dthroat
    # pn['pore.volume'] = vpore
    # pn['pore.shape_factor'] = shape_pore
    # pn['throat.shape_factor'] = shape_throat

    # # assign boundary labels
    # max_dim = np.max(coords[:, axis])
    # min_dim = np.min(coords[:, axis])
    # delta_dim = (max_dim - min_dim) * 0.5

    # mask_outlet = np.zeros((num_pores), dtype=bool)
    # mask_outlet[pores_bc] = True
    # mask_inlet = mask_outlet
    # label_out = np.reshape(coords[:, axis] > (max_dim - delta_dim), (num_pores))
    # label_in = np.reshape(coords[:, axis] < (min_dim + delta_dim), (num_pores))
    # mask_outlet = mask_outlet & label_out
    # mask_inlet = mask_inlet & label_in

    # # filter with previously determined boundary pores
    # pn.set_label(label='inlet', pores=mask_inlet)
    # pn.set_label(label='outlet', pores=mask_outlet)

    printIf('Raw network:')
    printIf(pn)

    # applying custom filter
    if 'Scale' not in dict_filter:
        dict_filter['Scale'] = scale
    p_labels = FilterPores(network=pn, prop=dict_filter)
    if len(p_labels) > 0:
        printIf('Filtered ' + str(len(p_labels)) + ' pores according to user specifications')
        op.topotools.trim(network=pn, pores=p_labels)
    num_pores = len(pn.coords)

    # removing isolated clusters within internal domain
    mask = np.zeros(num_pores) == 0
    p_labels = op.topotools.find_isolated_clusters(pn, mask=mask, inlets=pn.pores('inlet'))
    printIf('\nFound ' + str(len(p_labels)) + ' pores in isolated clusters')
    if (len(p_labels) > 0):
        printIf('Removing isolated pores')
        op.topotools.trim(network=pn, pores=p_labels)

    # here we need to manually remove isolated pores in the boundary domain
    am = pn.create_adjacency_matrix()
    adj_sum = am.sum(axis=1, dtype=int)
    mask = adj_sum == 0
    mask = np.squeeze(np.asarray(mask))
    bc_pore_mask = np.zeros(len(pn['pore.diameter']), dtype=bool)
    bc_pore_mask[pn.pores('inlet')] = True
    bc_pore_mask[pn.pores('outlet')] = True
    mask = mask & (bc_pore_mask)
    p_isolated_bc = np.nonzero(mask)[0]
    if (len(p_isolated_bc) > 0):
        printIf('Removing isolated pores in boundary')
        op.topotools.trim(network=pn, pores=p_isolated_bc)

    # test if system is now valid
    am = pn.create_adjacency_matrix()
    adj_sum = am.sum(axis=1, dtype=int)
    mask = adj_sum == 0
    mask = np.squeeze(np.asarray(mask))
    p_isolated_pores = np.nonzero(mask)[0]
    if (len(p_isolated_pores) > 0):
        print('Found isolated pores after removal, something is wrong!')

    dpore = pn['pore.diameter']
    dthroat = pn['throat.diameter']
    conns = pn['throat.conns']

    mask_1 = dthroat >= dpore[conns[:, 0]]
    mask_2 = dthroat >= dpore[conns[:, 1]]
    mask = mask_1 | mask_2
    if (np.count_nonzero(mask) > 0):
        printIf('\nAdjusting throats with weird diameters')
        dthroat[mask] = 0.9 * np.min(dpore[conns[mask, :]])
        num_adjusted = np.count_nonzero(mask)
        printIf('Adjusted ' + str(num_adjusted) + ' pores')

    pn['throat.diameter'] = dthroat

    printIf('\nAssigning geometric properties\n')
    mods = op.models.collections.geometry.spheres_and_cylinders
    mods.pop('throat.diameter', None)
    mods.pop('pore.diameter', None)
    mods.pop('pore.seed', None)
    mods.pop('pore.volume', None)
    mods.pop('pore.shape_factor', None)
    mods.pop('throat.shape_factor', None)
    mods['hydraulic_conductance'] = op.models.physics.hydraulic_conductance.valvatne_blunt
    pn.add_model_collection(mods)
    pn.regenerate_models()
    printIf('Updated Network with geometric properties:')
    printIf(pn)

    return pn


def WriteToVTK(particles: vtk.vtkAppendPolyData, out_file: str):
    particles.Update()
    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(particles.GetOutput())
    writer.SetFileName(out_file)
    writer.Update()


def WritePoresToVTK(coords, dpore, filename: str, quality: int):
    if coords.shape[0] != dpore.shape[0]:
        raise Exception('coordinates and pore incompatible')
    if (len(coords) == 0):
        print('No pores provided for writing, skipping writing of ' + filename)
        return

    all_spheres = vtk.vtkAppendPolyData()
    for i in range(len(dpore)):
        sphere = vtk.vtkSphereSource()
        sphere.SetThetaResolution(quality)
        sphere.SetPhiResolution(quality)
        sphere.SetRadius(dpore[i]*0.5)
        sphere.SetCenter(coords[i, 0], coords[i, 1], coords[i, 2])
        sphere.Update()
        all_spheres.AddInputData(sphere.GetOutput())

    WriteToVTK(all_spheres, filename)


def WriteNetworkToVTK(network, filename: str, quality: int):
    coords = network['pore.coords']
    dpore = network['pore.diameter']
    WritePoresToVTK(filename=filename, coords=coords, dpore=dpore, quality=quality)


def ConvertPNExtractToVTK(file_base: str, ext: str, quality: int):
    coords, dpore = ReadPoresPNExtract(file_base, ext)
    WritePoresToVTK(filename=file_base + '_pores.vtk', coords=coords, dpore=dpore, quality=quality)

    # all_cylinders = vtk.vtkAppendPolyData()
    # conns, dthroat = ReadThroatsPNExtract(file_base, ext)
    # mask = (conns[:,0] > -1) | (conns[:,1] > -1)
    # conns = conns[mask,:]
    # dthroat = dthroat[mask]
    # num_throat = len(dthroat)
    # coord_left = coords[conns[:,0]]
    # coord_right = coords[conns[:,1]]
    # coord_cyl = (coord_left + coord_right) * 0.5
    # dist = coord_left - coord_right
    # dist_sqr = np.sum(dist**2, 1)
    # height_cyl = phi = theta = np.zeros(num_throat, dtype=float)
    # for i in range(num_throat):
    #     height_cyl[i] = math.sqrt(dist_sqr[i])
    #     phi[i] = math.atan2(dist[i,0], dist[i,1])
    #     theta = math.acos(dist[i,2]/height_cyl[i])
    # phi = math.atan2(dist[:,0], dist[:,1])
    # theta = math.acos(dist[:,2]/height_cyl)
    # quality = quality * 2
    # for i in range(num_throat):
    #     cylinder = vtk.vtkCylinderSource()
    #     mapper = vtk.vtkPolyDataMapper()
    #     actor = vtk.vtkActor()
    #     cylinder.SetResolution(quality)
    #     cylinder.SetRadius(dthroat[i]*0.5)
    #     cylinder.SetCenter(coord_cyl[i, 0], coord_cyl[i, 1], coord_cyl[i, 2])
    #     cylinder.SetHeight(height_cyl[i])
    #     mapper.SetInputData(cylinder.GetOutput())
    #     actor.SetMapper(mapper)
    #     actor.RotateY(theta[i])
    #     actor.RotateZ(phi[i])
    #     cylinder.Update()
    #     all_cylinders.AddInputData(cylinder)
