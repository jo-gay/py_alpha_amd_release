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

# Example of how to use registration framework


# Import Numpy/Scipy
import numpy as np

# Import transforms
import transforms

# Import generators and filters
import filters

# Import registration framework
from register import Register

# Import misc
import sys
import time
import os
from PIL import Image


# Registration Parameters
alpha_levels = 7
# Pixel-size
spacing = np.array([1.0, 1.0])
# Run the symmetric version of the registration framework
symmetric_measure = True
# Use the squared Euclidean distance
squared_measure = False

# The number of iterations
param_iterations = 3000
# The fraction of the points to sample randomly (0.0-1.0)
param_sampling_fraction = 0.1
# Number of iterations between each printed output (with current distance/gradient/parameters)
param_report_freq = 500

# Directory to save output files - any existing files will be overwritten
outdir = './test_images/output/'

def main():
    np.random.seed(1000)
    
    if len(sys.argv) < 3:
        print(f'{sys.argv[0]}: Too few parameters. Give the path to two gray-scale image files.')
        print(f'Example: python {sys.argv[0]} reference_image floating_image')
        return False

    ref_im_path = sys.argv[1]
    flo_im_path = sys.argv[2]

    ref_im = Image.open(ref_im_path).convert('L')
    flo_im = Image.open(flo_im_path).convert('L')
    ref_im = np.asarray(ref_im)/255.
    flo_im = np.asarray(flo_im)/255.

    # Make copies of original images
    ref_im_orig = ref_im.copy()
    flo_im_orig = flo_im.copy()

    # Preprocess images
    ref_im = filters.normalize(ref_im, 0.0, None)
    flo_im = filters.normalize(flo_im, 0.0, None)

    weights1 = np.ones(ref_im.shape)
    mask1 = np.ones(ref_im.shape, 'bool')
    weights2 = np.ones(flo_im.shape)
    mask2 = np.ones(flo_im.shape, 'bool')

    # Initialize registration framework for 2d images
    reg = Register(2)
    reg.set_image_data(ref_im, flo_im, mask1, mask2, weights1, weights2)
    
    # Choose a registration model
    reg.set_model('alphaAMD', alpha_levels=alpha_levels, \
                  symmetric_measure=symmetric_measure, \
                  squared_measure=squared_measure)
    
    # Setup the Gaussian pyramid resolution levels
    reg.add_pyramid_level(4, 5.0)
    reg.add_pyramid_level(2, 3.0)
    reg.add_pyramid_level(1, 0.0)

    # Choose an optimizer and set optimizer-specific parameters
    # For GD and adam, learning-rate / Step lengths given by [[start1, end1], [start2, end2] ...] (for each pyramid level)
    reg.set_optimizer('gd', \
                      step_lengths=np.array([[1., 1.], [1., 0.5], [0.5, 0.1]]), \
                      gradient_magnitude_threshold=0.0001, \
                      iterations=param_iterations
                      )
#    reg.set_optimizer('scipy', \
#                      iterations=param_iterations
#                      )

    # Scale all transform parameters to approximately the same order of magnitude, based on sizes of images
    diag = 0.5 * (transforms.image_diagonal(ref_im, spacing) + transforms.image_diagonal(flo_im, spacing))

    # Create the initial transform and add it to the registration framework 
    # (switch between affine/rigid transforms by commenting/uncommenting)
#    # Affine
#    param_scaling = np.array([1.0/diag, 1.0/diag, 1.0/diag, 1.0/diag, 1.0, 1.0])
#    reg.add_initial_transform(transforms.AffineTransform(2), param_scaling=param_scaling)
    # Rigid 2D
    param_scaling = np.array([1.0/diag, 1.0, 1.0])
    reg.add_initial_transform(transforms.Rigid2DTransform(), param_scaling=param_scaling)

    # Set up other registration framework parameters
    reg.set_report_freq(param_report_freq)
    reg.set_sampling_fraction(param_sampling_fraction)

    # Create output directory
    directory = os.path.dirname(outdir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Start the pre-processing
    reg.initialize(outdir)
    
    # Control the formatting of numpy
    np.set_printoptions(suppress=True, linewidth=200)

    # Start the registration
    reg.run()

    (transform, value) = reg.get_output(0)

    ### Warp final image
    c = transforms.make_image_centered_transform(transform, ref_im, flo_im, spacing, spacing)

    # Print out transformation parameters
    print('Transformation parameters: %s.' % str(transform.get_params()))

    # Create the output image
    ref_im_warped = np.zeros(ref_im.shape)
    mask = np.ones(flo_im_orig.shape, dtype='bool')
    warped_mask = np.zeros(ref_im.shape, dtype='bool')

    # Transform the floating image into the reference image space by applying transformation 'c'
    c.warp(In = flo_im_orig, Out = ref_im_warped, in_spacing=spacing, out_spacing=spacing, mode='spline', bg_value = 0.0)
    c.warp(In = mask, Out = warped_mask, in_spacing=spacing, out_spacing=spacing, mode='spline', bg_value = 0.0)

    # Save the registered image
    Image.fromarray(ref_im_warped).convert('RGB').save(outdir+'registered.png')
    
    # Compute the absolute difference image between the reference and registered images
    D1 = np.abs(ref_im_orig-ref_im_warped)
    err = np.mean(D1[warped_mask])
    print("Err: %f" % err)

    Image.fromarray(D1).convert('RGB').save(outdir+'diff.png')

    return True

if __name__ == '__main__':
    start_time = time.time()
    res = main()
    end_time = time.time()
    if res == True:
        print("Elapsed time: " + str((end_time-start_time)))
