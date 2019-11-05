#
# Py-Alpha-AMD Registration Framework
# Author: Johan Ofverstedt, Jo Gay
# Reference: Fast and Robust Symmetric Image Registration Based on Distances Combining Intensity and Spatial Information
#
# Copyright 2019 Johan Ofverstedt
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
#

#
# Functions to open and pre-process images. Only tested with greyscale.
#
import numpy as np
from PIL import Image
#from matplotlib import pyplot as plt ## Only for exmple usage below

import filters

def preProcessImage(im, norm=True, blur=None, equalize=False, quantize=None):
    """Take a uint8 ndarray representing an image and preprocess it according to arguments.
    Note: returns a floating point (0-1) image.
    """

    #Convert to float to avoid any overflow or rounding issues
    im = np.array(im, dtype='float64')
    if blur and blur > 0:
        im = filters.gaussian_filter(im, blur)

    if norm:
        im = filters.normalize(im, 0.0, None)
    else:
        im = im/255. #convert to floats between 0 and 1 without normalizing

    if equalize:        
        im = filters.image_histogram_equalization(im)

    if quantize:
        im = np.rint(im * (quantize-1))/(quantize-1)
    
    return im

def openAndPreProcessImage(path, copyOrig=False, preproc={}):
    """Open an image file, optionally make a copy, and apply preprocessing to it.
    """
    try:
        im = Image.open(path).convert('L') #Open as a uint8 image
    except FileNotFoundError:
        print(f'Error: {path} not found')
        return
    except OSError:
        print(f'Error: Cannot open {path}, please check image formats supported by PIL.Image')
        return
    im = np.asarray(im)#[125:375,125:375] #Take a smaller region for speed
    
    # Also return an unprocessed copy of original image, if required
    im_orig = im.copy() if copyOrig else None
    
    return preProcessImage(im, **preproc), im_orig

#
#if __name__ == '__main__':
#    ## Example usage
#    myPath = "../test_images/reference_example.png"
#    preproc = {'norm':True, 'blur':1.0, 'equalize':False, 'quantize':32}
#    
#    ## Open and preprocess from a filename
#    processed_im, orig_im = openAndPreProcessImage(myPath, \
#                                             copyOrig=True, preproc=preproc)
#    plt.subplot(1,2,1)
#    plt.imshow(orig_im, cmap='gray', vmin=0, vmax=255)
#    plt.subplot(1,2,2)
#    plt.imshow(processed_im, cmap='gray', vmin=0, vmax=1)
#    plt.show()
#
#    ## Preprocess from an existing image array
#    preproc = {'norm':True, 'blur':2.0, 'equalize':True}
#    processed_im = preProcessImage(orig_im, **preproc)
#    plt.subplot(1,2,1)
#    plt.imshow(orig_im, cmap='gray', vmin=0, vmax=255)
#    plt.subplot(1,2,2)
#    plt.imshow(processed_im, cmap='gray', vmin=0, vmax=1)
#    plt.show()
    
    