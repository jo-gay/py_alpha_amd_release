
#
# Py-Alpha-AMD Registration Framework
# Author: Johan Ofverstedt
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
# Example script for affine registration
#

# Import Numpy/Scipy
import numpy as np
#import scipy as sp
from PIL import Image


# Import transforms
#from transforms import CompositeTransform
#from transforms import AffineTransform
#from transforms import Rigid2DTransform
#from transforms import Rotate2DTransform
#from transforms import TranslationTransform
#from transforms import ScalingTransform
import transforms

# Import optimizers
#from optimizers import GradientDescentOptimizer

# Import generators and filters
#import generators
import filters

# Import registration frameworks
import models

# Import misc
#import math
import sys
import time
import os

# Choice of registration method out of ('alphaAMD', 'MI')
param_method = 'mi'
# Registration Parameters
alpha_levels = 7
# Pixel-size
spacing = np.array([1.0, 1.0])
# Run the symmetric version of the registration framework
symmetric_measure = True
# Use the squared Euclidean distance
squared_measure = False

# The number of iterations
param_iterations = 5000
# The fraction of the points to sample randomly (0.0-1.0)
param_sampling_fraction = 1.0
# Number of iterations between each printed output (with current distance/gradient/parameters)
param_report_freq = 0

# Choice of optimizer from those available ('sgd', 'adam', 'scipy')
param_optimizer = 'gridsearch'

# Where should output files be saved
param_outdir = './test_images/output/'

# Default path for ref and flo images
example_ref_im = './test_images/reference_example.png'
example_flo_im = './test_images/floating_example.png'

#ref_im_path = './test_images/roi_148184_6.tif'
#flo_im_path = './test_images/mpm_148184_6_gs.tif'

def main():
    #np.random.seed(1000)

    if len(sys.argv) > 1:
        ref_im_path = sys.argv[1]
    else:
        ref_im_path = example_ref_im
    if len(sys.argv) > 2:
        flo_im_path = sys.argv[2]
    else:
        flo_im_path= example_flo_im

    print('Registering floating image %s with reference image %s'%(flo_im_path, ref_im_path))
    print('Similarity measure %s, optimizer %s'%(param_method, param_optimizer))

    ref_im = Image.open(ref_im_path).convert('L')
    flo_im = Image.open(flo_im_path).convert('L')
    ref_im = np.asarray(ref_im)
    flo_im = np.asarray(flo_im)

    # Save copies of original images
    ref_im_orig = ref_im.copy()
    flo_im_orig = flo_im.copy()

    # Initialize registration model for 2d images and do specific preprocessing and setup for that model
    if param_method.lower() == 'alphaamd':
        reg = models.RegisterAlphaAMD(2)
        reg.set_alpha_levels(alpha_levels)
        ref_im = filters.normalize(ref_im, 0.0, None)
        flo_im = filters.normalize(flo_im, 0.0, None)
    elif param_method.lower() == 'mi':
        ref_im = filters.normalize(ref_im, 0.0, None)
        flo_im = filters.normalize(flo_im, 0.0, None)
        reg = models.RegisterMI(2)
    else:
        raise NotImplementedError('Method must be one of alphaAMD, MI')
    reg.set_report_freq(param_report_freq)

    # Generic initialization steps required for every registration model
    weights1 = np.ones(ref_im.shape)
    mask1 = np.ones(ref_im.shape, 'bool')
    weights2 = np.ones(flo_im.shape)
    mask2 = np.ones(flo_im.shape, 'bool')

    reg.set_reference_image(ref_im)
    reg.set_reference_mask(mask1)
    reg.set_reference_weights(weights1)

    reg.set_floating_image(flo_im)
    reg.set_floating_mask(mask2)
    reg.set_floating_weights(weights2)

    # Setup the Gaussian pyramid resolution levels
    reg.add_pyramid_level(4, 5.0)
    reg.add_pyramid_level(2, 3.0)
    reg.add_pyramid_level(1, 0.0)

    # Learning-rate / Step lengths [[start1, end1], [start2, end2] ...] (for each pyramid level)
    step_lengths = np.array([[1., 1.], [1., 0.5], [0.5, 0.1]])
    
    # Estimate an appropriate parameter scaling based on the sizes of the images.
    diag = transforms.image_diagonal(ref_im, spacing) + transforms.image_diagonal(flo_im, spacing)
    diag = 2.0/diag

    # Create the transform and add it to the registration framework (switch between affine/rigid transforms by commenting/uncommenting)
    # Affine
    reg.add_initial_transform(transforms.AffineTransform(2), param_scaling=np.array([diag, diag, diag, diag, 1.0, 1.0]))
    # Rigid 2D
    #reg.add_initial_transform(transforms.Rigid2DTransform(), param_scaling=np.array([diag, 1.0, 1.0]))

    # Set the parameters
    reg.set_iterations(param_iterations)
    reg.set_gradient_magnitude_threshold(1e-6)
    reg.set_sampling_fraction(param_sampling_fraction)
    reg.set_step_lengths(step_lengths)
    reg.set_optimizer(param_optimizer)

    # Create output directory
    directory = os.path.dirname(param_outdir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Start the pre-processing
    reg.initialize(param_outdir)
    
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

    # Transform the floating image into the reference image space by applying transformation 'c'
    c.warp(In = flo_im_orig, Out = ref_im_warped, in_spacing=spacing, out_spacing=spacing, mode='spline', bg_value = 0.0)

    # Save the registered image
    Image.fromarray(ref_im_warped).convert('RGB').save(param_outdir+'registered.png')

    # Compute the absolute difference image between the reference and registered images
    D1 = np.abs(ref_im_orig-ref_im_warped)
    err = np.mean(D1)
    print("Err: %f" % err)

    Image.fromarray(D1).convert('RGB').save(param_outdir+'diff.png')

    return True

if __name__ == '__main__':
    start_time = time.time()
    res = main()
    end_time = time.time()
    if res == True:
        print("Elapsed time: " + str((end_time-start_time)))
