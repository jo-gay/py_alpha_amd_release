#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 12:58:10 2019

@author: jo
"""
import transforms
import numpy as np

class MIDistance:
    def __init__(self, fun):
        self.ref_image = None
        self.flo_image = None
        self.mi_fun = fun

        self.sampling_fraction = 1.0
        
        self.best_val = 0
        self.best_trans = None

    #For mutual information we need integer types so adjust the input images if necessary
    def set_ref_image(self, image):
        if image.dtype == np.dtype('float64'):
            self.ref_image = np.rint(image*255).astype('uint8')
        else:
            self.ref_image = image

    #For mutual information we need integer types so adjust the input images if necessary
    def set_flo_image(self, image):
        if image.dtype == np.dtype('float64'):
            self.flo_image = np.rint(image*255).astype('uint8')
        else:
            self.flo_image = image

    def set_sampling_fraction(self, sampling_fraction):
        self.sampling_fraction = sampling_fraction
        if sampling_fraction != 1.0:
            raise NotImplementedError('Sampling fraction not implemented')
        
    ''' 
    Calculate the centre coordinates in x,y order
    Image should be ndarray (indexed in y,x order)
    '''
    def get_center_point(self, image):
        rows, cols = image.shape
        return cols/2, rows/2

    def initialize(self):
        pass

    '''
    Apply the given transform and calculate the resulting mutual info score.
    Gradient is not implemented
    '''
    def value_and_derivatives(self, transform):
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
        c_trans.warp(In = self.flo_image, Out = warped_image, mode='spline', bg_value = 0)
        c_trans.warp(In = mask, Out = warped_mask, mode='nearest', bg_value = 0)

        # Cast back to integer values for mutual information comparison
        warped_image = np.rint(warped_image).astype('uint8')

        value = self.mi_fun(self.ref_image[warped_mask>0], warped_image[warped_mask>0])
        
        if value > self.best_val:
            self.best_val = value
            self.best_trans = transform.copy()
            print('New best value %2.4f at ('%value, ', '.join(['%8.3f']*len(params))%tuple(params), ')')
        grad = None

        return -value, grad