# PrepareImage.py
# requires pylibtiff
from libtiff import TIFF
from PIL import Image, ImageSequence
import numpy as np
import os


def CheckFileExtension(path: str, formats):
    """
    tests, if the provided path ends in one of the provided formats and additionally
    provides information, if the path has any ending at all
    input:
    path (str):     path for testing
    formats:        list of formats for testing
    return:
    correct_extension (bool):  if the extension was found in formats
    has_extension (bool):      if the path has an extension
    """
    _, ext = os.path.splitext(path)
    has_extension = ext != ''
    correct_extension = ext in formats
    return correct_extension, has_extension


def IsTiff(path: str):
    """
    Tests, if the provided path is actually a tiff
    input:
    path(str):  path for testing
    for returns, see CheckFileExtension
    """
    return CheckFileExtension(path, ['.tif', '.tiff', '.TIF', '.TIFF'])


def ReadTiffStack(image: str):
    """
    Reads in a tiff image and returns a numpy ndarray
    image: string with image name, relative or absolute
    """
    correct_extension, has_extension = IsTiff(image)
    if not has_extension:
        image = image + '.tif'
    elif not correct_extension:
        raise Exception('Cannot read image file, ' + image + ' is not an accepted filename!')

    tif = TIFF.open(image, mode='r') # open tiff file in read mode
    # read an image in the current TIFF directory as a numpy array
    imstack = list(tif.iter_images())
    if len(imstack) == 0:
        imstack = imstack[0]
    else:
        # converting to 3d array, note that numpy will set the list as
        # dimension 0, whereas it is more consistent to use the stack as
        # dimension 2
        imstack = np.array(imstack, dtype=type(imstack[0][0,0]))
        imstack = np.moveaxis(imstack, 0, 2)
    return imstack

    imstack_libtiff = imstack
    # alternative approach
    im = Image.open(image)
    imstack = list(enumerate(ImageSequence.Iterator(im)))
    num_images = len(imstack)
    imshape = np.array(imstack[0][1]).shape
    image = np.zeros((imshape[0], imshape[1], num_images))
    for n in range(num_images):
        image[:,:,n] = np.array(imstack[n][1])
    
    return image


def WriteTiffStack(fname: str, data):
    """
    Writes data to a tiff image
    fname: string with image name, relative or absolute
    data: array with data
    """
    raise Exception('this function does not work yet')
    correct_extension, has_extension = IsTiff(fname)
    if not has_extension:
        fname = fname + '.tif'
    elif not correct_extension:
        print('WARNING: ' + fname + ' has the wrong extension for a Tiff image!')

    tif = TIFF.open(fname, mode='w') # open tiff file in write mode
    tif.write_image(data)
