
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
# Image filters
#

# Import Numpy/Scipy
import numpy as np
import scipy.ndimage as ndimage

def image_histogram_equalization(image, number_bins=256):
    """ Apply histogram equalization to an image and return the equalized image.
    Adapted from
    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html
    
    Arguments:
        image       : 2D ndarray of floating point values between 0 and 1
        number_bins : How many bins should be used to create the equalized image, max 256.
                      NOTE: Tested only with 256 bins.

    Returns:
        2D ndarray of fp values between 0 and 1
    
    """
    hist, bins = np.histogram(image.flatten()*(number_bins-1),number_bins,[0,number_bins])

    cdf = hist.cumsum()
    
    cdf_m = np.ma.masked_equal(cdf,0) #mask out anything below the minimum intensity
    cdf_m = number_bins*(cdf_m - cdf_m.min())/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0)
    image_rint = np.rint(image*(number_bins-1)).astype('uint8')
    return (cdf[image_rint.flatten()]/number_bins).reshape(image.shape)


def add_noise(image, noise_type, gauss_amount=0.4, sp_amount=0.004):
    """
    Arguments:
    image : ndarray
        Input image data as floating point ndarray
    mode : str
        One of the following strings, selecting the type of noise to add:
    
        'gauss'     Gaussian-distributed additive noise.
        'poisson'   Poisson-distributed noise generated from the data.
        's&p'       Replaces random pixels with 0 or 1.
        'speckle'   Multiplicative noise using out = image + n*image,where
                    n,is uniform noise with specified mean & variance.
    gauss_amount : float (default 0.004). 
        For Gaussian noise, the weighting for the standard deviation of the noise
    sp_amount : float (default 0.004). 
        For S&P noise, the proportion of pixels that will be affected
    Returns:
        ndarray of image with noise added
        
    Adapted from an answer to 
    https://stackoverflow.com/questions/14435632/impulse-gaussian-and-salt-and-pepper-noise-with-opencv
    by Shubham Pachori
    """
    if noise_type.lower() == "gauss":
        gauss = np.random.normal(0,image.std()*gauss_amount,image.shape)
        return image + gauss
    elif noise_type.lower() == "s&p":
        s_vs_p = 0.5
        out = image.copy()
        # Salt mode
        num_salt = np.ceil(sp_amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[tuple(coords)] = 1
        # Pepper mode
        num_pepper = np.ceil(sp_amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[tuple(coords)] = 0
        return out
    elif noise_type.lower() == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        return np.random.poisson(image * vals) / float(vals)
    elif noise_type.lower() == "speckle":
        gauss = np.random.randn(*image.shape)
        return image + image * gauss

    raise NotImplementedError(f"Noise type {noise_type} not recognized")

def gaussian_filter(image, sigma):
    if sigma <= 0.0:
        return image
    else:
        return ndimage.gaussian_filter(image, sigma)

def downsample(image, n):
    n = np.int(n) # make sure we have an integer
    if image.ndim == 1:
        return image[0::n]
    elif image.ndim == 2:
        return image[0::n, 0::n]
    elif image.ndim == 3:
        return image[0::n, 0::n, 0::n]
    elif image.ndim == 4:
        return image[0::n, 0::n, 0::n, 0::n]
    else:
        raise 'Not implemented yet for dimensions other than 1-4.'

def _normalize_and_clip(image, mn, mx):
    return np.clip((image-mn)/(mx-mn), 0.0, 1.0)

def _normalize_with_mask(image, percentile, mask):
    n = image.size
    vec = image.reshape((n,))
    vec_mask = mask.reshape((n,))
    sorted_vec = np.sort(vec[vec_mask])
    m = sorted_vec.size
    mn_ind = np.clip(np.int(n*percentile), 0, m-1)
    mx_ind = np.clip(m-mn_ind-1, 0, m-1)
    return _normalize_and_clip(image, sorted_vec[mn_ind], sorted_vec[mx_ind])    

def _normalize_with_no_mask(image, percentile):
    n = image.size
    vec = image.reshape((n,))
    sorted_vec = np.sort(vec)
    mn_ind = np.clip(np.int(n*percentile), 0, n-1)
    mx_ind = np.clip(n-mn_ind-1, 0, n-1)
    return _normalize_and_clip(image, sorted_vec[mn_ind], sorted_vec[mx_ind])

def _normalize_with_zero_percentile_no_mask(image):
    return _normalize_and_clip(image, np.amin(image), np.amax(image))

def _normalize_with_zero_percentile_with_mask(image, mask):
    return _normalize_and_clip(image, np.amin(image[mask]), np.amax(image[mask]))
    
def normalize(image, percentile=0.0, mask=None):
    if mask is None:
        if percentile == 0.0:
            return _normalize_with_zero_percentile_no_mask(image)
        else:
            return _normalize_with_no_mask(image, percentile)
    else:
        if percentile == 0.0:
            return _normalize_with_zero_percentile_with_mask(image, mask)
        else:
            return _normalize_with_mask(image, percentile, mask)

if __name__ == '__main__':
    np.random.seed(1000)

    im = np.arange(24).reshape((4, 6)) / 48.0

    im_comp = im + 0.2
    im_comp = np.random.permutation(im_comp)
    
    # Add outliers
    im_comp[0, 0] = -0.5
    im_comp[3, 4] = 2.0

    print(im_comp)

    res0 = normalize(im_comp, 0.0, None)
    print(res0)

    res1 = normalize(im_comp, 0.1, None)
    print(res1)

    mask = np.zeros(im.shape, 'bool')
    mask[1:-1, 1:-1] = True
    res2 = normalize(im_comp, 0.0, mask)
    print(res2)

    res3 = normalize(im_comp, 0.05, mask)
    print(res3)    

    im1 = np.arange(16*8).reshape((16, 8))
    print(im1)
    im1_ds2 = downsample(im1, 2)
    print(im1_ds2)
    im1_ds4 = downsample(im1, 4)
    print(im1_ds4)    