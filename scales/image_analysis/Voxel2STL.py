import numpy as np
from skimage import measure
from stl import mesh
from tqdm import tqdm
import glob
from joblib import Parallel, delayed
import os


def Extract(image:str, fname: str, opt):
    """
    Extracts an STL mesh from the provided image path
    """
    raise Exception('needs to be redesigned with PointCloud2STL!')
    verbose = opt.get('verbose', True)
    level = opt.get('level', None)
    step_size = opt.get('step_size', 1)
    fname, ext = os.splitext(fname)
    fname = f'{fname}.stl'
    if verbose:
        print('Determining iso surfaces - this may take a while')
    vertices, faces, _, _ = measure.marching_cubes(image,
                                                   level=level,
                                                   step_size=step_size)
    num_faces = len(faces)
    if verbose:
        print(f'Computed {num_faces} facets')

    # Convert to mesh and save
    if verbose:
        print('Converting to Mesh')
    data_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in tqdm(enumerate(faces), total=num_faces, disable=(not verbose)):
        for j in range(3):
            data_mesh.vectors[i][j] = vertices[f[j], :]

    print('Saving to file: ' + fname)
    data_mesh.save(fname)


def ExtractInstanceFromList(n, imlist, num_images, out_base: str, opt):
    if opt.get('verbose', True):
        print(f'Processing image {n}/{num_images}')
    if out_base is None:
        fname, _ = os.splitext(imlist[n])
    elif not hasattr(out_base, '__len__'):
        fname = f'{out_base}_{n}'
    elif hasattr(out_base, '__len__'):
        fname = out_base[n]

    Extract(image=imlist[n], fname=fname, opt=opt)


def ExtractSeries(args):
    ext = args.get('format', 'tiff')
    if args.folder:
        folders = args['folder']
        args.pop('folder', None)
        if not hasattr(folders, '__len__'):
            folders = [folders]
        for folder in folders:
            list_im = glob.glob(f'{folder}*{ext}')
            if len(list_im) == 0:
                print(f'Cannot find any images in {folder} with the extension {ext}')
            args['source'] = list_im
            ExtractSeries(args=args)

    # here we can now extract the images
    if 'source' not in args:
        print('No source files (key: source) were found in the arguments!')
        return

    source = args['source']
    out_base = args.get('fout', None)
    run_in_parallel = args.get('parallel', False)
    num_jobs = args.get('njobs', 1) if run_in_parallel else 1
    args['verbose'] = args.get('verbose', (not run_in_parallel))
    num_images = len(source)
    Parallel(n_jobs=num_jobs)(
        delayed(ExtractInstanceFromList)(n,
                                         list_images=source,
                                         num_images=num_images,
                                         out_base=out_base,
                                         opt=args) for n in range(num_images))
