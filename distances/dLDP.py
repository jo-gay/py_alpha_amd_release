#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# Py-Alpha-AMD Registration Framework
# Authors: Johan Ofverstedt, Jo Gay
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
# Discriminative Local Derivative Patterns, distance class. An implementation of
# Jiang, D., Shi, Y., Chen, X., Wang, M., and Song, Z. (2017).  
# Fast and robust multimodal imageregistration using a local derivative pattern.
# Medical physics, 44(2):497–509.
# https://aapm.onlinelibrary.wiley.com/doi/pdf/10.1002/mp.12049
# 

import transforms
import numpy as np
from matplotlib import pyplot as plt


def x_deriv(image, mode='difference', sigma=3.0):
    """Calculate the derivative in the x direction for each pixel in
    the given image.
    
    The derivative is calculated as d(x,y) = I(x, y) - I(x+1, y).

    Arguments:
        image: should be provided as a 2D ndarray. Indexing is in y, x order
        mode: one of 'difference' or 'gaussian'. Diff returns the left finite difference
              while Gaussian uses a gaussian filter for a smoother result
        sigma: parameter for gaussian derivative

    Returns:
        2D ndarray of derivatives, same shape as image. Final column is zeros.
    """
    if mode == 'difference':
        return np.pad(image[...,:-1] - image[...,1:], ((0,0),(0,1)), mode='constant')
    
    raise NotImplementedError('Gaussian derivatives not yet implemented')

def y_deriv(image, mode='difference', sigma=3.0):
    """Calculate the derivative in the y direction for each pixel in
    the given image.
    
    The derivative is calculated as d(x,y) = I(x, y) - I(x, y+1), i.e. the left 
    finite difference.

    Arguments:
        image: should be provided as a 2D ndarray. Indexing is in y, x order
        mode: one of 'difference' or 'gaussian'. Diff returns the left finite difference
              while Gaussian uses a gaussian filter for a smoother result
        sigma: parameter for gaussian derivative

    Returns:
        2D ndarray of derivatives, same shape as image. Final row is zeros.
    """
    if mode == 'difference':
        return np.pad(image[:-1,...] - image[1:,...], ((0,1),(0,0)), mode='constant')
    raise NotImplementedError('Gaussian derivatives not yet implemented')

def deriv_sign(image, direction):
    """Calculate the sign of the derivative in the given direction for each pixel in
    the given image.
    
    The derivative is calculated as I'(Z_0) = I(Z_0) - I(Z_x), i.e. the  
    finite difference where x is one of the neighbouring pixels, depending on the 
    direction specified. Method from Zhang et al "Local Derivative Pattern 
    Versus Local Binary Pattern".

    Arguments:
        image: should be provided as a 2D ndarray. Indexing is in y, x order
        direction: 0 => 0 degrees
                   1 => 45 degrees
                   2 => 90 degrees
                   3 => 135 degrees

    Returns:
        2D ndarray of derivative directions, same shape as image:
            True where derivative is positive
            False where derivative is negative
            False where derivative is undefined due to edges.
    """
    if direction == 0:
        diff = (image[:, :-1] - image[:, 1:]) >= 0
        diff = np.pad(diff, ((0,0),(0,1)), mode='constant')
    elif direction == 1:
        diff = (image[1:, :-1] - image[:-1, 1:]) >= 0
        diff = np.pad(diff, ((1,0),(0,1)), mode='constant')
    elif direction == 2:
        diff = (image[1:, :] - image[:-1, :]) >= 0
        diff = np.pad(diff, ((1,0),(0,0)), mode='constant')
    else:
        diff = (image[1:, 1:] - image[:-1, :-1]) >= 0
        diff = np.pad(diff, ((1,0),(1,0)), mode='constant')
        
    return diff

def deriv_4way(image):
    """Calculate the derivative in four directions (0, 45, 90, 135 degrees) for 
    each pixel in the given image.
    
    Returns:
        4-channel derivative image, with shape (*image.shape, 4)
    """
    return np.stack([deriv_sign(image, i) for i in range(4)], axis=-1)

def masked_not_xor(region1, region2, mask):
    """ Given two image regions of the same size (2D ndarrays of booleans), and 
    a mask relating to one of them, return a ndarray of the same size containing:
    for elements that are not masked 
        NOT XOR (True if the elements are the same)
    for elements that are masked 
        -1
    """
    ret = np.where(mask, \
                   np.logical_not(np.logical_xor(region1, region2)), \
                   -1)
    return ret

def not_xor(region1, region2):
    """ Given two image regions of the same size (2D ndarrays of booleans) return 
        an ndarray of the same size containing: NOT XOR (True if the elements are the same)
    """
    return np.logical_not(np.logical_xor(region1, region2))

def extend_mask(mask, pixels=1):
    """Given a binary mask, extend the masked out area by a number of pixels such that 
    an unmasked pixel becomes masked if any of its diagonal neighbours (at specified
    distance) are masked.
    
    Shifts the mask by pixels in each of 4 directions (45 degrees, 135 degrees,
    225 degrees, 315 degrees). Any pixel covered by the shifted mask will be masked.
    This will only work for a mask with relatively continuous areas masked out.
    """
    newmask = np.pad(mask, ((pixels,pixels),(pixels,pixels)), mode='constant')
    p2 = 2*pixels
    newmask[p2:,p2:] = np.logical_and(newmask[p2:,p2:], mask)
    newmask[:-p2,p2:] = np.logical_and(newmask[:-p2,p2:], mask)
    newmask[p2:,:-p2] = np.logical_and(newmask[p2:,:-p2], mask)
    newmask[:-p2,:-p2] = np.logical_and(newmask[:-p2,:-p2], mask)
    newmask[:p2,:]=0
    newmask[-p2:,:]=0
    newmask[:,:p2]=0
    newmask[:,-p2:]=0
    
    return newmask[pixels:-pixels, pixels:-pixels]

def binaryVectorToInt(v):
    """Convert a vector of 1s and 0s to an integer.

    If the vector contains any -1s return 0.
    """
    if -1 in v:
        return 0
    if v.shape == (8,):
        return int(''.join(map(str, v)), base=2)
    print('Possible error in vector:', v)

"""Create a vectorized version of the above to work on an ndarray
"""
binaryVectorsToInt = np.vectorize(binaryVectorToInt, signature='(n)->()')


def padded_neighbour_xor(image, direction, padvalue=0):
    """For a given binary image / feature map of size (x, y, channels), for each pixel return
    the xor of its own value with that of its 'direction'th neighbour, where neighbours 
    are numbered from 1 to 8 clockwise from the top left corner. For pixels which do not
    have a neighbour in that direction, return padding value given.
    
    This is for LDP but not used for dLDP
    """
    if direction == 1:
        ret = np.pad(np.logical_xor(image[1:,1:,...], image[:-1,:-1,...]).astype('int'), ((1,0),(1,0),(0,0)), \
                     mode='constant', constant_values=padvalue)
    elif direction == 2:
        ret = np.pad(np.logical_xor(image[1:,:,...], image[:-1,:,...]).astype('int'), ((1,0),(0,0),(0,0)), \
                     mode='constant', constant_values=padvalue)
    elif direction == 3:
        ret = np.pad(np.logical_xor(image[1:,:-1,...], image[:-1,1:,...]).astype('int'), ((1,0),(0,1),(0,0)), \
                     mode='constant', constant_values=padvalue)
    elif direction == 4:
        ret = np.pad(np.logical_xor(image[:,:-1,...], image[:,1:,...]).astype('int'), ((0,0),(0,1),(0,0)), \
                     mode='constant', constant_values=padvalue)
    elif direction == 5:
        ret = np.pad(np.logical_xor(image[:-1,:-1,...], image[1:,1:,...]).astype('int'), ((0,1),(0,1),(0,0)), \
                     mode='constant', constant_values=padvalue)
    elif direction == 6:
        ret = np.pad(np.logical_xor(image[:-1,:,...], image[1:,:,...]).astype('int'), ((0,1),(0,0),(0,0)), \
                     mode='constant', constant_values=padvalue)
    elif direction == 7:
        ret = np.pad(np.logical_xor(image[:-1,1:,...], image[1:,:-1,...]).astype('int'), ((0,1),(1,0),(0,0)), \
                     mode='constant', constant_values=padvalue)
    elif direction == 8:
        ret = np.pad(np.logical_xor(image[:,1:,...], image[:,:-1,...]).astype('int'), ((0,0),(1,0),(0,0)), \
                     mode='constant', constant_values=padvalue)
        
    return ret

def neighbour_deriv_signs(d1, d2):
    """Given two derivative images, compare the sign of each pixel in the first image
    with the signs of the derivatives of its neighbours in the second image.
    
    Return 8-bit descriptor for each pixel where 1 indicates different signs.
    """
#     descriptor = np.zeros((*d1.shape, 8), dtype='int') - 1
#     descriptor[1:, 1:, 0] = np.logical_xor(d1[1:,1:], d2[:-1,:-1])#, mask[:-1,:-1])
#     descriptor[1:, :, 1] = np.logical_xor(d1[1:,:], d2[:-1,:])#, mask[:-1,:])
#     descriptor[1:, :-1, 2] = np.logical_xor(d1[1:,:-1], d2[:-1,1:])#, mask[:-1,1:])
#     descriptor[:, :-1, 3] = np.logical_xor(d1[:,:-1], d2[:,1:])#, mask[:,1:])
#     descriptor[:-1, :-1, 4] = np.logical_xor(d1[:-1,:-1], d2[1:,1:])#, mask[1:,1:])
#     descriptor[:-1, :, 5] = np.logical_xor(d1[:-1,:], d2[1:,:])#, mask[1:,:])
#     descriptor[:-1, 1:, 6] = np.logical_xor(d1[:-1,1:], d2[1:,:-1])#, mask[1:,:-1])
#     descriptor[:, 1:, 7] = np.logical_xor(d1[:,1:], d2[:,:-1])#, mask[:,:-1])
    
#     descriptor = np.zeros((8, *d1.shape), dtype='int') - 1
#     descriptor[0, 1:, 1:] = np.logical_xor(d1[1:,1:], d2[:-1,:-1])#, mask[:-1,:-1])
#     descriptor[1, 1:, :] = np.logical_xor(d1[1:,:], d2[:-1,:])#, mask[:-1,:])
#     descriptor[2, 1:, :-1] = np.logical_xor(d1[1:,:-1], d2[:-1,1:])#, mask[:-1,1:])
#     descriptor[3, :, :-1] = np.logical_xor(d1[:,:-1], d2[:,1:])#, mask[:,1:])
#     descriptor[4, :-1, :-1] = np.logical_xor(d1[:-1,:-1], d2[1:,1:])#, mask[1:,1:])
#     descriptor[5, :-1, :] = np.logical_xor(d1[:-1,:], d2[1:,:])#, mask[1:,:])
#     descriptor[6, :-1, 1:] = np.logical_xor(d1[:-1,1:], d2[1:,:-1])#, mask[1:,:-1])
#     descriptor[7, :, 1:] = np.logical_xor(d1[:,1:], d2[:,:-1])#, mask[:,:-1])
    
#     descriptor = np.zeros((8, *d1.shape), dtype='int')
#     descriptor[0] = np.pad(np.logical_xor(d1[1:,1:], d2[:-1,:-1]), ((1,0),(1,0)), \
#                            mode='constant', constant_values=-1)
#     descriptor[1] = np.pad(np.logical_xor(d1[1:,:], d2[:-1,:]), ((1,0),(0,0)), \
#                            mode='constant', constant_values=-1)
#     descriptor[2] = np.pad(np.logical_xor(d1[1:,:-1], d2[:-1,1:]), ((1,0),(0,1)), \
#                            mode='constant', constant_values=-1)
#     descriptor[3] = np.pad(np.logical_xor(d1[:,:-1], d2[:,1:]), ((0,0),(0,1)), \
#                            mode='constant', constant_values=-1)
#     descriptor[4] = np.pad(np.logical_xor(d1[:-1,:-1], d2[1:,1:]), ((0,1),(0,1)), \
#                            mode='constant', constant_values=-1)
#     descriptor[5] = np.pad(np.logical_xor(d1[:-1,:], d2[1:,:]), ((0,1),(0,0)), \
#                            mode='constant', constant_values=-1)
#     descriptor[6] = np.pad(np.logical_xor(d1[:-1,1:], d2[1:,:-1]), ((0,1),(1,0)), \
#                            mode='constant', constant_values=-1)
#     descriptor[7] = np.pad(np.logical_xor(d1[:,1:], d2[:,:-1]), ((0,0),(1,0)), \
#                            mode='constant', constant_values=-1) 


#     return descriptor


    dshape = (8, d1.shape[0]+2, d1.shape[1]+2)
    descriptor = np.zeros(dshape, dtype='bool')
    
    # pad the reference image with zeros all around
    ref = np.pad(d1, ((1,1), (1,1)), mode='constant', constant_values=0)

    descriptor[0] = np.logical_xor(ref, \
                                   np.pad(d2, ((2,0),(2,0)), \
                                          mode='constant', constant_values=0))
    descriptor[1] = np.logical_xor(ref, \
                                   np.pad(d2, ((2,0),(1,1)), \
                                          mode='constant', constant_values=0))
    descriptor[2] = np.logical_xor(ref, \
                                   np.pad(d2, ((2,0),(0,2)), \
                                          mode='constant', constant_values=0))
    descriptor[3] = np.logical_xor(ref, \
                                   np.pad(d2, ((1,1),(0,2)), \
                                          mode='constant', constant_values=0))
    descriptor[4] = np.logical_xor(ref, \
                                   np.pad(d2, ((0,2),(0,2)), \
                                          mode='constant', constant_values=0))
    descriptor[5] = np.logical_xor(ref, \
                                   np.pad(d2, ((0,2),(1,1)), \
                                          mode='constant', constant_values=0))
    descriptor[6] = np.logical_xor(ref, \
                                   np.pad(d2, ((0,2),(2,0)), \
                                          mode='constant', constant_values=0))
    descriptor[7] = np.logical_xor(ref, \
                                   np.pad(d2, ((1,1),(2,0)), \
                                          mode='constant', constant_values=0))
#     descriptor[:,1,:]=-1
#     descriptor[:,-2,:]=-1
#     descriptor[:,:,1]=-1
#     descriptor[:,:,-2]=-1
    return descriptor[:, 1:-1, 1:-1]


class dLDPDistance:
    def __init__(self, version='dLDP_8', mode='difference', interpolation='nearest'):
        """Set up the SSD measure.
        
        """
        self.version = version
        self.ref_image = None
        self.flo_image = None
        self.ref_mask = None
        self.ref_dLDP = None
        
        self.diff_mode = mode
        self.interpolation = interpolation

        self.sampling_fraction = 1.0
        
        self.best_val = 0
        self.best_trans = None

    def dLDP_as_image(self, dLDP):
        """Transform the representation into an image for visualization. 
        
        Map the descriptor for each pixel to a value between 0 and 2^(descriptor length)-1, 
        then rescale to values between 0 and 1. Where the descriptor contains any -1 values,
        this is mapped to zero.
        """
        intData = binaryVectorsToInt(dLDP.astype(int))
        intData = intData/(pow(2, len(dLDP[0,0,:]))-1)
        return intData
    
    def create_LDP(self, image, mask=None):
        """Create a Local Derivative Pattern descriptor for a given
        image and optional mask, using 4 directions.
        
        Args:
            image: ndarray
            mask:  ndarray of true/false specifying which pixels to include/ignore. if None,
            all pixels will be included
            mode:  default 'finite' (this is the basic version which calculates the gradient as the difference
                                     between the intensity at a pixel and that at its neighbour).
                   alterative 'gaussian' to be added TODO
        Returns:
            LDP descriptor: 32 element binary array for each pixel in the given image. 
                            Contains -1 where LDP could not be calculated due to edge
                            of image or mask
            mask:           adjusted mask to exclude pixels for which the full 32 bit
                            descriptor could not be calculated
        """
        if mask is None:
            mask = np.ones(image.shape, dtype='bool')

        #Calculate derivatives in four directions
        Iprime = deriv_4way(image)
        #Convert derivatives to binary where True means >= 0
        Iprime = Iprime >= 0
        
        
        # For each of the 8 neighbours, starting from above left and going clockwise,
        # determine whether the derivative at the central pixel has the same sign 
        # as that of its neighbour, in each of the 4 directions,
        # and save these results into the descriptor

#        descriptor = np.zeros((8, *Iprime.shape), dtype='int') - 1
#        descriptor[0, 1:, 1:, :] = np.logical_xor(Iprime[1:,1:,:], Iprime[:-1,:-1,:])#, mask[:-1,:-1])
#        descriptor[1, 1:, :, :] = np.logical_xor(Iprime[1:,:,:], Iprime[:-1,:,:])#, mask[:-1,:])
#        descriptor[2, 1:, :-1, :] = np.logical_xor(Iprime[1:,:-1,:], Iprime[:-1,1:,:])#, mask[:-1,1:])
#        descriptor[3, :, :-1, :] = np.logical_xor(Iprime[:,:-1,:], Iprime[:,1:,:])#, mask[:,1:])
#        descriptor[4, :-1, :-1, :] = np.logical_xor(Iprime[:-1,:-1,:], Iprime[1:,1:,:])#, mask[1:,1:])
#        descriptor[5, :-1, :, :] = np.logical_xor(Iprime[:-1,:,:], Iprime[1:,:,:])#, mask[1:,:])
#        descriptor[6, :-1, 1:, :] = np.logical_xor(Iprime[:-1,1:,:], Iprime[1:,:-1,:])#, mask[1:,:-1])
#        descriptor[7, :, 1:, :] = np.logical_xor(Iprime[:,1:,:], Iprime[:,:-1,:])#, mask[:,:-1])

        # New method takes 54% of the amount of time of the above:
        descriptor=[]
        for i in range(8):
            descriptor.append(padded_neighbour_xor(Iprime, direction=i+1, padvalue=-1))
        descriptor = np.array(descriptor)
        
        # Want the 4 directions to be stacked in the descriptor so more the new 'neighbour' axis
        # to the end and then reshape to flatten the last two dimensions.
        descriptor = np.rollaxis(descriptor, 0, 4)
        descriptor = descriptor.reshape(*descriptor.shape[:2], -1)
        
        # mask out pixels on the edge of the image where we can't have a full descriptor
        mask[0,:]=0
        mask[-1,:]=0
        mask[:,0]=0
        mask[:,-1]=0
        
        # mask out pixels on the edge of the (remaining) unmasked region, where neighbouring pixels
        # are outside the mask
        mask = extend_mask(mask, pixels=1)
        
        return descriptor, mask
        
    def create_dLDP(self, image, mask=None):
        """Create a discriminative Local Derivative Pattern descriptor for a given
        image and optional mask.
        
        Args:
            image: ndarray
            mask:  ndarray of true/false specifying which pixels to include/ignore. if None,
            all pixels will be included
            mode:  default 'finite' (this is the basic version which calculates the gradient as the difference
                                     between the intensity at a pixel and that at its neighbour).
                   alterative 'gaussian' to be added TODO
        Returns:
            dLDP descriptor: 8 or 48 element binary array for each pixel in the given image. 
                             Contains -1 where dLDP could not be calculated due to edge
                             of image or mask
            mask:            adjusted mask to exclude pixels for which the full 16 bit
                             descriptor could not be calculated
        """
        
        if self.version.lower() == 'ldp':
            return self.create_LDP(image, mask)
        
        if mask is None:
            mask = np.ones(image.shape, dtype='bool')
    
        #mask out the first row and first/last columns as we won't have derivatives in these posns
        mask[0,:] = False
        mask[:,0] = False
        mask[:,-1] = False
    #     mask[-1,:] = False
    
    #     #Increase the masked out area by one pixel in each direction, so that we do
    #     #not compare a pixel with a neighbour outside the mask.
        mask = extend_mask(mask, pixels=1)
    
    
        #Calculate derivatives in two/four directions
        #Not interested in the actual values, just the signs. convert to true/false,
        #where true corresponds to a positive derivative
        derivs=[]
        dirs = range(4)
        comparisons = 6
        if self.version.lower() == 'dldp_8':
            dirs = [0, 2]
            comparisons = 1
        for d in dirs:
            derivs.append(deriv_sign(image, d))
    
        # For each pair of directions, 
        # check whether the derivative at the central pixel has the same sign 
        # as the other derivative of each of its neighbours,
        # and save these results into the descriptor
        descriptor = np.zeros((comparisons, 8, *image.shape), dtype='bool')
        c=0#counter
        for idx, i in enumerate(derivs[:-1]):
            for j in derivs[(idx+1):]:
                descriptor[c] = neighbour_deriv_signs(i, j)
                c += 1
    
        #Now descriptor shape is (1 or 6, 8, m, n). reshape to (m, n, 8 or 48)
        descriptor = descriptor.reshape((-1, *descriptor.shape[2:]))
        descriptor = np.rollaxis(descriptor, 0, 3)
    
        return descriptor, mask
        

    def set_ref_image(self, image, mask=None):
        """Set the reference (fixed) image to be used by the distance measure.
        
        Args:
            image: ndarray of image data in float64 format.
            image will not be modified but a copy should be passed in if original may change
            during execution.
        """
        self.ref_image = image
        
        #If a new mask is supplied then set it
        if mask is not None:
            self.ref_mask = mask
        #If none is supplied and none already exists, or the one supplied has invalid shape, set to ones.
        if self.ref_mask is None or self.ref_mask.shape != self.ref_image.shape:
            self.ref_mask = np.ones(self.ref_image.shape, 'bool')
        
        self.ref_dLDP, self.ref_mask = self.create_dLDP(self.ref_image, self.ref_mask)
        
        if False:
            im0 = self.dLDP_as_image(self.ref_dLDP[...,:8])
            im90 = self.dLDP_as_image(self.ref_dLDP[...,8:])
            
            plt.figure(figsize=(16,16))
            plt.subplot(131)
            plt.imshow(image, cmap='gray', vmin=0, vmax=1)
            plt.subplot(132)
            plt.imshow(im0, cmap='gray', vmin=0, vmax=1)
            plt.subplot(133)
            plt.imshow(im90, cmap='gray', vmin=0, vmax=1)
            plt.show()


    def set_flo_image(self, image):
        """Set the floating (moving) image to be used by the distance measure.
        
        Args:
            image: ndarray of image data in float64 (will be converted to uint8) or uint8 format.
            image will not be modified but a copy should be passed in if original may change
            during execution.
        """
        self.flo_image = image

    def set_sampling_fraction(self, sampling_fraction=1.0):
        """Set the proportion of image pixels to be used for calculating MI.
        
        This can be used to reduce execution time, particularly with adjusted MI.
        Or it could, if it was implemented.
        
        Args:
            sampling_fraction: number between 0 and 1, default 1.0
        """
        self.sampling_fraction = sampling_fraction
        if sampling_fraction != 1.0:
            raise NotImplementedError('Sampling fraction not implemented')
        
    def _get_center_point(self, image):
        """Return the centre coordinates in x,y order.

        Args:
            image: ndarray
        """
        rows, cols = image.shape
        return cols/2, rows/2

    def initialize(self):
        """Do nothing. All initialization is handled during instantiation."""
        pass
    
    def dLDPdist(self, ref_dLDP, flo_dLDP):
        """Calculate the dLDP distance measure for the given pixels. 
        
        This is simply the average hamming distance between the descriptor arrays.
        
        Args:
            ref_dLDP, flo_dLDP: ndarrays containing dLDP descriptors for each 
            pixel. Mask should be applied first.
        Returns:
            average hamming distance between pairs of pixels.
        """
        assert(ref_dLDP.shape == flo_dLDP.shape), "Error: cannot compare dLDP representations for" + \
                                                  " different sized images"

        return np.sum(np.logical_xor(ref_dLDP, flo_dLDP))/(ref_dLDP.shape[0])

    def value_and_derivatives(self, transform):
        """Apply the given transform and calculate the dLDP distance between the 
        resulting pair of images.

        Gradient is not implemented.
        
        Args:
            transform: an object implementing the transform API (subclass of BaseTransform)
        
        Returns:
            dLDP measure for given transform, None
        """
        params=transform.get_params()
        c_trans = transforms.make_image_centered_transform(transform, \
                                                     self.ref_image, self.flo_image)

        # Only compare pixels that were part of the floating image, not background.
        # Use a mask of the same shape, transformed in the same way, to determine 
        # which pixels are within range
        mask = np.ones(self.flo_image.shape, dtype='bool')

        # Create the output image
        warped_image = np.zeros(self.ref_image.shape)
        warped_mask = np.zeros(self.ref_image.shape)
    
        # Transform the floating image into the reference image space by applying transformation 'c_trans'
        c_trans.warp(In = self.flo_image, Out = warped_image, mode=self.interpolation, bg_value = 0)
        c_trans.warp(In = mask, Out = warped_mask, mode='nearest', bg_value = 0)
        
        # If the overlap is less than 40% then exclude it
        if len(warped_image[np.logical_and(warped_mask > 0, self.ref_mask > 0)]) < 0.4*np.prod(self.flo_image.shape): #too small an overlap, skip it.
            return np.inf, None
        
        warped_image_dLDP, warped_mask = self.create_dLDP(warped_image, warped_mask)

        value = self.dLDPdist(self.ref_dLDP[np.logical_and(warped_mask > 0, self.ref_mask > 0)], \
                            warped_image_dLDP[np.logical_and(warped_mask > 0, self.ref_mask > 0)])
        
        paramstr=' '.join(['%.5f'%p for p in params])
#        print(f'[{paramstr}] -> %.5f'%value)
#        if value > self.best_val:
#            self.best_val = value
#            self.best_trans = transform.copy()
#            print('New best value %2.4f at ('%value, ', '.join(['%8.3f']*len(params))%tuple(params), ')')
        grad = None

        return value, grad