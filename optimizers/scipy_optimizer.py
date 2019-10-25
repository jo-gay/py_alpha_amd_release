#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

# Hook into scipy minimize function to optimize image registration for an arbitrary
# measure of dissimilarity. Useful for measures where no analytical gradient is 
# available e.g. MI.


import numpy as np
from scipy.optimize import minimize

def _default_report_callback(opt):
    iteration = opt.get_iteration()
    value = opt.get_value()
    param = opt.get_transform().get_params()
    print("#%d. --- Value: " % (iteration) + str(value) + ", Param: " + str(param))

class SciPyOptimizer:
#    methods = ['Nelder-Mead','Powell', 'CG', 'BFGS', 'Newton-CG',\
#               'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr', \
#               'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov']
    

    def __init__(self, measure, transform):
        """Initialize optimizer with a distance measure and initial transform.
        
        Args:
        measure - must have a function get_value() taking a transform object and returning
                  a scalar distance metric, such that smaller distance = better registration,
                  should be initialized as needed with image data etc
        transform - usually a subclass of TransformBase, must have get_params() method
        """
        self.method = 'BFGS'
        self.transform = transform
        self.minimizer_options = {}
        self.param_scaling = None
        self.measure = measure

        self.last_value = np.nan
        self.iteration = 0
        self.termination_reason = ""
        self.report_freq = 0
        self.report_func = _default_report_callback
        self.value_history = []

    def _set_scalings(self, scalings):
        """Set a scaling proportion for each parameter being optimized over.
        
        Transform parameters will be multiplied by scalings (list of same length as params)
        in order to harmonize step lengths for optimizer
        """
        if self.param_scaling is None:
            self.param_scaling = scalings.copy()
        else:
            self.param_scaling[:] = scalings[:]
    
    def get_iteration(self):
        return self.iteration

    def get_success_flag(self):
        return self.success
    
    def get_termination_reason(self):
        return self.termination_reason
    
    def set_report_freq(self, freq):
        if freq < 0:
            freq = 0
        self.report_freq = freq
    
    def set_report_callback(self, func, additive = True):
        if additive == True:
            if func is not None:
                old_func = self.report_func
                if old_func is not None:
                    def glue(opt):
                        old_func(opt)
                        func(opt)
                    self.report_func = glue
                else:
                    self.report_func = func
        else:
            self.report_func = func
    
    def get_value(self):
        return self.last_value

    def get_value_history(self):
        return self.value_history

    def get_transform(self):
        return self.transform

    def _fun_to_minimize(self, params):
        """Return the value of the distance measure at the given parameters, and print
        status report if required.

        Args:
            params - List, the unscaled transform parameters
        Returns:
            scalar value of distance metric at the given transform
        """

        self.transform.set_params(params * self.param_scaling)

        v, _ = self.measure.value_and_derivatives(self.transform)

        self.last_value = v
        self.iteration += 1

        if self.report_freq > 0 and \
            (self.iteration % self.report_freq == 0) and self.report_func:
            self.report_func(self)
            
#        # DEBUG
#        print('Evaluating registration at ' + \
#              ','.join(['%.5f']*len(self.transform.get_params()))%tuple(self.transform.get_params()))
#        
#        print('rescaled from ' + \
#              ','.join(['%.5f']*len(params))%tuple(params))
#        print('Gives distance %.4f'%v)
#        # END DEBUG

        return v
        
    def set_options(self, options): #TODO change to list the parameters with defaults?
        """Set options for the optimizer.
        Dictionary keys:
            'method': String, name of optimization method to use. If not specified, 
                    use current method.
            'param_scaling': List of floats, one per parameter in transform. If not 
                    specified, use current scaling
            'minimizer_opts': Dictionary of other options to pass on to scipy 
                    library method. If not specified, use current options
            'reset_minimizer': Boolean. If True, any existing minimizer options will
                    be removed, even if not specified in minimizer_opts. Otherwise
                    existing settings are updated.
        """
        self.method = options.get('method', self.method)
        self._set_scalings(options.get('param_scaling', self.param_scaling))
        self._set_minimizer_opts(options.get('minimizer_opts', self.minimizer_options), \
                                 options.get('reset_minimizer', False))
        
    def _set_minimizer_opts(self, minim_opts, clear=True):
        """Set or update options dict which will be passed to the scipy minimizer function.
        
        For any options not included in arguments, use existing values, unless clear=True then remove them.
        """
        if clear:
            self.minimizer_options = {}
        self.minimizer_options.update(minim_opts)
    
    def optimize(self):
        """Use scipy.optimize.minimize to perform the optimization with parameters specified."""
        start_params = self.transform.get_params() / self.param_scaling
        
        x_opt = minimize(self._fun_to_minimize, start_params, \
                         method=self.method, \
                         options=self.minimizer_options)

        self.transform.set_params(x_opt.x * self.param_scaling)
        self.termination_reason = x_opt.message
        self.success = x_opt.success
        self.value_history = x_opt.fun
        
        print('%s optimizer terminated with status %s and message %s'%(self.method, \
                                                                       self.success, \
                                                                       self.termination_reason))

        return x_opt.nit
