#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
# Registration framework for AlphaAMD
#

# Import Numpy/Scipy
import numpy as np

# Import Register base class
from models import Register

# Import distances required for alphaAMD
from distances import QuantizedImage
from distances import alpha_amd
from distances import symmetric_amd_distance


class RegisterAlphaAMD(Register):
    
    def set_alpha_levels(self, alpha_levels):
        self.alpha_levels = alpha_levels

    def make_dist_measure(self, ref_resampled, ref_mask_resampled, ref_weights, \
                          flo_resampled, flo_weights, flo_mask_resampled, pyramid_factor):

        ref_diag = np.sqrt(np.square(np.array(ref_resampled.shape)*self.ref_spacing).sum())
        flo_diag = np.sqrt(np.square(np.array(flo_resampled.shape)*self.flo_spacing).sum())

        q_ref = QuantizedImage(ref_resampled, self.alpha_levels, ref_weights, self.ref_spacing*pyramid_factor, remove_zero_weight_pnts = True)
        q_flo = QuantizedImage(flo_resampled, self.alpha_levels, flo_weights, self.flo_spacing*pyramid_factor, remove_zero_weight_pnts = True)

        tf_ref = alpha_amd.AlphaAMD(q_ref, self.alpha_levels, ref_diag, self.ref_spacing*pyramid_factor, ref_mask_resampled, ref_mask_resampled, interpolator_mode='linear', dt_fun = None, mask_out_edges = True)
        tf_flo = alpha_amd.AlphaAMD(q_flo, self.alpha_levels, flo_diag, self.flo_spacing*pyramid_factor, flo_mask_resampled, flo_mask_resampled, interpolator_mode='linear', dt_fun = None, mask_out_edges = True)

        symmetric_measure = True
        squared_measure = False

        sym_dist = symmetric_amd_distance.SymmetricAMDDistance(symmetric_measure=symmetric_measure, squared_measure=squared_measure)

        sym_dist.set_ref_image_source(q_ref)
        sym_dist.set_ref_image_target(tf_ref)

        sym_dist.set_flo_image_source(q_flo)
        sym_dist.set_flo_image_target(tf_flo)

        sym_dist.set_sampling_fraction(self.sampling_fraction)

        sym_dist.initialize()
        return sym_dist

