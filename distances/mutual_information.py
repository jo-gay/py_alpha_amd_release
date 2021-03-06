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
# Mutual information measure class
#

import transforms
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score, mutual_info_score

from functools import partial
norm_mi_arithmetic = partial(normalized_mutual_info_score, average_method='arithmetic')


class MIDistance:
    def __init__(self, fun, levels=256):
        """Set up the mutual information distance measure.
        
        Args:
            fun: a mutual information function e.g. sklearn.metrics.{adjusted_mutual_info_score, 
            normalized_mutual_info_score, mutual_info_score}. Could in fact be any function that takes
            lists of (integer) intensity values for two images as flattened arrays and returns a 
            similarity measure. If a string, will be interpreted as one of scikit-learn MI fns.
            levels: number of intensity levels in image, default 256. If a floating point image
            is provided, will be multiplied by this number minus 1 and rounded to nearest integer
            for mutual information comparison.
        """
        self.ref_image = None
        self.flo_image = None
        self.ref_mask = None
        self.set_mutual_info_fun(fun)

        self.sampling_fraction = 1.0
        self.nLevels = levels - 1
        
        self.best_val = 0
        self.best_trans = None

    def set_mutual_info_fun(self, mutual_info_fn=None):
        """Set or change the choice of mutual information function. 
        
        It must take a pair of 1D arrays and return a scalar. 
        Args:
            fun: If it is a string, interpret as one of three sklearn.metrics functions
            otherwise, assume it is a function that can be called as above.
        """
        if mutual_info_fn == 'mi' or mutual_info_fn is None:
            self.mi_fun = mutual_info_score
        elif mutual_info_fn == 'normalized' or mutual_info_fn == 'norm':
            self.mi_fun = norm_mi_arithmetic
        elif mutual_info_fn == 'adjusted' or mutual_info_fn == 'adj':
            self.mi_fun = adjusted_mutual_info_score
        else:
            self.mi_fun = mutual_info_fn


    def set_ref_image(self, image, mask=None):
        """Set the reference (fixed) image to be used by the distance measure.
        
        Args:
            image: ndarray of image data in float64 (will be converted to uint8) or uint8 format.
            image will not be modified but a copy should be passed in if original may change
            during execution.
        """
        #For mutual information we need integer types so adjust the input images if necessary
        if image.dtype == np.dtype('float64'):
            self.ref_image = np.rint(image*self.nLevels).astype('uint8')
        else:
            self.ref_image = image
        
        #If a new mask is supplied then set it
        if mask is not None:
            self.ref_mask = mask
        #If none is supplied and none already exists, or the one supplied has invalid shape, set to ones.
        if self.ref_mask is None or self.ref_mask.shape != self.ref_image.shape:
            self.ref_mask = np.ones(self.ref_image.shape, 'bool')

    def set_flo_image(self, image):
        """Set the floating (moving) image to be used by the distance measure.
        
        Args:
            image: ndarray of image data in float64 (will be converted to uint8) or uint8 format.
            image will not be modified but a copy should be passed in if original may change
            during execution.
        """
        #For mutual information we need integer types so adjust the input images if necessary
        if image.dtype == np.dtype('float64'):
            self.flo_image = np.rint(image*self.nLevels).astype('uint8')
        else:
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

    def value_and_derivatives(self, transform):
        """Apply the given transform and calculate the resulting mutual info score. Return the negative of this.

        Gradient is not implemented.
        
        Args:
            transform: an object implementing the transform API (subclass of BaseTransform)
        
        Returns:
            negative of MI score for given transform, None
        """
        params=transform.get_params()
        c_trans = transforms.make_image_centered_transform(transform, \
                                                     self.ref_image, self.flo_image)

        # Only compare pixels that were part of the floating image, not background.
        # Use a mask of the same shape, transformed in the same way, to determine 
        # which pixels are within range
        mask = np.ones(self.flo_image.shape)

        # Create the output image
        warped_image = np.zeros(self.ref_image.shape)
        warped_mask = np.zeros(self.ref_image.shape)
    
        # Transform the floating image into the reference image space by applying transformation 'c_trans'
        c_trans.warp(In = self.flo_image, Out = warped_image, mode='nearest', bg_value = 0)
        c_trans.warp(In = mask, Out = warped_mask, mode='nearest', bg_value = 0)
        if len(warped_image[np.logical_and(warped_mask > 0, self.ref_mask > 0)]) < 0.4*np.prod(self.flo_image.shape): #too small an overlap, skip it.
            return 0, None

        # Cast back to integer values for mutual information comparison
        warped_image = np.where(warped_image < 0, 0, warped_image)
        warped_image = np.where(warped_image > self.nLevels, self.nLevels, warped_image)
        warped_image = np.rint(warped_image).astype('uint8')

        value = self.mi_fun(self.ref_image[np.logical_and(warped_mask > 0, self.ref_mask > 0)], \
                            warped_image[np.logical_and(warped_mask>0, self.ref_mask > 0)])
#        if value > 1:
#            print("MI function returned a value > 1")
#        if value > self.best_val:
#            self.best_val = value
#            self.best_trans = transform.copy()
#            print('New best value %2.4f at ('%value, ', '.join(['%8.3f']*len(params))%tuple(params), ')')
        grad = None

        return -value, grad