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
# Registration framework for Mutual Information
#

# Import Numpy/Scipy
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score, mutual_info_score

# Import distances
from distances import MIDistance

# Import Register base class
from models import Register

class RegisterMI(Register):

    def __init__(self, dims):
        super().__init__(dims)
        self.mi_fun = mutual_info_score
    
    # set or change the choice of mutual information function. It must take a pair of 1D vectors and return a scalar
    def set_mutual_info_fun(self, fun):
        self.mi_fun = fun
        
    def make_dist_measure(self, ref_resampled, ref_mask_resampled, ref_weights, \
                          flo_resampled, flo_weights, flo_mask_resampled, pyramid_factor):

        mi_dist = MIDistance(self.mi_fun)
        mi_dist.set_ref_image(ref_resampled)
        mi_dist.set_flo_image(flo_resampled)

        mi_dist.initialize()
        return mi_dist

