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

# Example script showing how to use registration framework

### General imports used in this example script
import numpy as np
import sys
import time
import os
from PIL import Image

### Imports for registration framework
import transforms
import filters
from register import Register


### Set registration parameters for alphaAMD model
# Levels
alpha_levels = 7
# Pixel-size
spacing = np.array([1.0, 1.0])
# Run the symmetric version of the registration framework
symmetric_measure = True
# Use the squared Euclidean distance
squared_measure = False

### General registration parameters
# The number of iterations (unless gradient threshold reached)
param_iterations = 2000
# The fraction of the points to sample randomly (0.0-1.0)
param_sampling_fraction = 0.25
# Number of iterations between each printed output (with current distance/gradient/parameters)
param_report_freq = 500

### Directory to save output files - any existing files will be overwritten
outdir = './test_images/output/'

def main():
    np.random.seed(1000)
    
    if len(sys.argv) < 3:
        print(f'{sys.argv[0]}: Too few parameters. Give the path to two gray-scale image files.')
        print(f'Example: python {sys.argv[0]} reference_image floating_image')
        return False

    ref_im_path = sys.argv[1]
    flo_im_path = sys.argv[2]

    ### Open the images to be registered and convert to greyscale with intensity between 0-1
    preproc = {'norm':True}
    ref_im, ref_im_orig = filters.openAndPreProcessImage(ref_im_path, copyOrig=True, preproc=preproc)
    flo_im, flo_im_orig = filters.openAndPreProcessImage(flo_im_path, copyOrig=True, preproc=preproc)

    ### All pixels in the images will be included, and weighted equally
    weights1 = np.ones(ref_im.shape)
    mask1 = np.ones(ref_im.shape, 'bool')
    weights2 = np.ones(flo_im.shape)
    mask2 = np.ones(flo_im.shape, 'bool')

    ### Initialize registration framework for 2d images
    reg = Register(2)
    reg.set_image_data(ref_im, flo_im, mask1, mask2, weights1, weights2)
    
    ### Choose a registration model
    reg.set_model('alphaAMD', alpha_levels=alpha_levels, \
                  symmetric_measure=symmetric_measure, \
                  squared_measure=squared_measure)
#    reg.set_model('mi')
    
    ### Setup the Gaussian pyramid resolution levels (if required)
    reg.add_pyramid_level(4, 5.0)
    reg.add_pyramid_level(2, 3.0)
    reg.add_pyramid_level(1, 0.0)

    ### Scale all transform parameters to approximately the same order of magnitude, based on sizes of images
    diag = 0.5 * (transforms.image_diagonal(ref_im, spacing) + transforms.image_diagonal(flo_im, spacing))

    ### Create the initial transform and add it to the registration framework 
    ### (switch between affine/rigid transforms by commenting/uncommenting)
#    # e.g. Affine
#    initial_transform = transforms.AffineTransform(2)
#    param_scaling = np.array([1.0/diag, 1.0/diag, 1.0/diag, 1.0/diag, 1.0, 1.0])
    # e.g. Rigid 2D
    initial_transform = transforms.Rigid2DTransform()
    param_scaling = np.array([1.0/diag, 1.0, 1.0])
#    # e.g. Composite scale + rigid
#    param_scaling = np.array([1.0/diag, 1.0/diag, 1.0, 1.0])
#    initial_transform = transforms.CompositeTransform(2, [transforms.ScalingTransform(2, uniform=True), \
#                                                transforms.Rigid2DTransform()])

    reg.add_initial_transform(initial_transform, param_scaling=param_scaling)


    ### Choose an optimizer and set optimizer-specific parameters
    ### For GD and adam, learning-rate / Step lengths given by [[start1, end1], [start2, end2] ...] (for each pyramid level)
    ### For gridsearch, bounds are specified for each parameter in the transform chosen above (e.g. Affine: 6 params)
#    reg.set_optimizer('adam', \
#                      gradient_magnitude_threshold=0.01, \
#                      iterations=param_iterations
#                      )
#    reg.set_optimizer('gd', \
#                      step_length=np.array([1., 0.5, 0.25]), \
#                      end_step_length=np.array([0.4, 0.2, 0.01]), \
#                      gradient_magnitude_threshold=0.01, \
#                      iterations=param_iterations
#                      )
    reg.set_optimizer('gridsearch', \
                      bounds=[[-0.5,0.5],[-5,5],[-5,5]], \
                      steps=21 \
                      )
#    reg.set_optimizer('scipy', \
#                      iterations=param_iterations, \
#                      epsilon=0.00001 \
#                      )



    ### Set up other registration framework parameters
    reg.set_report_freq(param_report_freq)
    reg.set_sampling_fraction(param_sampling_fraction)

    ### Create output directory
    directory = os.path.dirname(outdir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    ### Start the pre-processing
    reg.initialize(outdir)
    
    ### Control the formatting of numpy
    np.set_printoptions(suppress=True, linewidth=200)

    ### Start the registration
    reg.run()

    (transform, value) = reg.get_output(0)

    ### Warp final image
    c = transforms.make_image_centered_transform(transform, ref_im, flo_im, spacing, spacing)

    ### Print out transformation parameters and status
    print('Starting from %s, optimizer terminated with message: %s'%(str(initial_transform.get_params()), \
                                                                    reg.get_output_messages()[0]))
    print('Final transformation parameters: %s.' % str(transform.get_params()))

    ### Create the output image
    ref_im_warped = np.zeros(ref_im.shape)
    mask = np.ones(flo_im_orig.shape, dtype='bool')
    warped_mask = np.zeros(ref_im.shape, dtype='bool')

    ### Transform the floating image into the reference image space by applying transformation 'c'
    c.warp(In = flo_im_orig, Out = ref_im_warped, in_spacing=spacing, out_spacing=spacing, mode='spline', bg_value = 0.0)
    c.warp(In = mask, Out = warped_mask, in_spacing=spacing, out_spacing=spacing, mode='spline', bg_value = 0.0)

    ### Save the registered image
    Image.fromarray(ref_im_warped).convert('RGB').save(outdir+'registered.png')
    
    ### Compute the absolute difference image between the reference and registered images
    D1 = np.abs(ref_im_orig-ref_im_warped)
    err = np.mean(D1[warped_mask])
    print("Err: %f" % err)

    ### Save the difference image to file
    Image.fromarray(D1).convert('RGB').save(outdir+'diff.png')

    return True

if __name__ == '__main__':
    start_time = time.time()
    res = main()
    end_time = time.time()
    if res == True:
        print("Elapsed time: " + str((end_time-start_time)))
