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
# Discriminative Local Derivative Patterns, distance class. 
#

import transforms
import numpy as np


def x_deriv(image):
    """Calculate the derivative in the x direction for each pixel in
    the given image.
    
    The derivative is calculated as d(x,y) = I(x, y) - I(x+1, y).

    Arguments:
        image should be provided as a 2D ndarray. Indexing is in y, x order

    Returns:
        2D ndarray of derivatives, same shape as image. Final column is zeros.
    """
    return np.pad(image[...,:-1] - image[...,1:], ((0,0),(0,1)), mode='constant')

def y_deriv(image):
    """Calculate the derivative in the y direction for each pixel in
    the given image.
    
    The derivative is calculated as d(x,y) = I(x, y) - I(x, y+1), i.e. the left 
    finite difference.

    Arguments:
        image should be provided as a 2D ndarray. Indexing is in y, x order

    Returns:
        2D ndarray of derivatives, same shape as image. Final row is zeros.
    """
    return np.pad(image[:-1,...] - image[1:,...], ((0,1),(0,0)), mode='constant')
    

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

class dLDPDistance:
    def __init__(self):
        """Set up the SSD measure.
        
        """
        self.ref_image = None
        self.flo_image = None
        self.ref_mask = None
        self.ref_dLDP = None

        self.sampling_fraction = 1.0
        
        self.best_val = 0
        self.best_trans = None

    def create_dLDP(self, image, mask=None, mode='finite'):
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
            dLDP descriptor: 16 element binary array for each pixel in the given image. 
                             Contains -1 where dLDP could not be calculated due to edge
                             of image or mask
            mask:            adjusted mask to exclude pixels for which the full 16 bit
                             descriptor could not be calculated
        """
        
        if mask is None:
            mask = np.ones(image.shape, dtype='bool')

        #Calculate derivatives in both directions
        xd = x_deriv(image, mode)
        yd = y_deriv(image, mode)
        
        #Not interested in the actual values, just the signs. convert to true/false,
        #where true corresponds to a positive derivative
        xd = xd >= np.abs(xd)
        yd = yd >= np.abs(yd)

        #mask out the final row and column as we don't have derivatives for these
        mask[:,-1] = False
        mask[-1,:] = False
        
        descriptor = np.zeros((*image.shape, 16), dtype='int') - 1
        
        # For each of the 8 directions, starting from above left and going clockwise,
        # determine whether the x derivative at the central pixel has the same sign 
        # as the y derivative of its neighbour,
        # and save these results into the descriptor
        descriptor[1:, 1:, 0] = masked_not_xor(xd[1:,1:], yd[:-1,:-1], mask[:-1,:-1])
        descriptor[1:, :, 1] = masked_not_xor(xd[1:,:], yd[:-1,:], mask[:-1,:])
        descriptor[1:, :-1, 2] = masked_not_xor(xd[1:,:-1], yd[:-1,1:], mask[:-1,1:])
        descriptor[:, :-1, 3] = masked_not_xor(xd[:,:-1], yd[:,1:], mask[:,1:])
        descriptor[:-1, :-1, 4] = masked_not_xor(xd[:-1,:-1], yd[1:,1:], mask[1:,1:])
        descriptor[:-1, :, 5] = masked_not_xor(xd[:-1,:], yd[1:,:], mask[1:,:])
        descriptor[:-1, 1:, 6] = masked_not_xor(xd[:-1,1:], yd[1:,:-1], mask[1:,:-1])
        descriptor[:, 1:, 7] = masked_not_xor(xd[:,1:], yd[:,:-1], mask[:,:-1])
        
        # Now do the same but using the y derivative of the central pixel vs x 
        # derivatives of neighbours
        descriptor[1:, 1:, 8] = masked_not_xor(yd[1:,1:], xd[:-1,:-1], mask[:-1,:-1])
        descriptor[1:, :, 9] = masked_not_xor(yd[1:,:], xd[:-1,:], mask[:-1,:])
        descriptor[1:, :-1, 10] = masked_not_xor(yd[1:,:-1], xd[:-1,1:], mask[:-1,1:])
        descriptor[:, :-1, 11] = masked_not_xor(yd[:,:-1], xd[:,1:], mask[:,1:])
        descriptor[:-1, :-1, 12] = masked_not_xor(yd[:-1,:-1], xd[1:,1:], mask[1:,1:])
        descriptor[:-1, :, 13] = masked_not_xor(yd[:-1,:], xd[1:,:], mask[1:,:])
        descriptor[:-1, 1:, 14] = masked_not_xor(yd[:-1,1:], xd[1:,:-1], mask[1:,:-1])
        descriptor[:, 1:, 15] = masked_not_xor(yd[:,1:], xd[:,:-1], mask[:,:-1])
        
        # Adjust the mask to exclude pixels where the full 8-vector could not be 
        # calculated (for a rectangle this is the outer rows and columns)
        has_result = np.min(descriptor, axis=2) > -1
        mask = np.logical_and(mask, has_result)
        
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
        assert(ref_dLDP.shape == flo_dLDP.shape)
        return np.logical_xor(ref_dLDP, flo_dLDP)/len(ref_dLDP.flatten())

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
        c_trans.warp(In = self.flo_image, Out = warped_image, mode='spline', bg_value = 0)
        c_trans.warp(In = mask, Out = warped_mask, mode='nearest', bg_value = 0)
        
        # If the overlap is less than 40% then exclude it
        if len(warped_image[np.logical_and(warped_mask > 0, self.ref_mask > 0)]) < 0.4*np.prod(self.flo_image.shape): #too small an overlap, skip it.
            return np.inf, None
        
        warped_image_dLDP, warped_mask = self.create_dLDP(warped_image, warped_mask)

        value = self.dLDPdist(self.ref_dLDP[np.logical_and(warped_mask > 0, self.ref_mask > 0)], \
                            warped_image_dLDP[np.logical_and(warped_mask > 0, self.ref_mask > 0)])
        
#        if value > self.best_val:
#            self.best_val = value
#            self.best_trans = transform.copy()
#            print('New best value %2.4f at ('%value, ', '.join(['%8.3f']*len(params))%tuple(params), ')')
        grad = None

        return value, grad