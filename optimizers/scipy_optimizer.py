#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 17:03:13 2019

@author: jo

Hook into scipy minimize function to optimize image registration for an arbitrary
measure of (dis)similarity. Useful for measures where no analytical gradient is 
available e.g. MI.

"""



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
    

    '''
    Initialize optimizer with 
    - measure (must have a function get_value() taking a transform object and returning
               a distance measure, such that smaller distance = better registration,
               should be initialized with image data etc);
    - transform (usually subclass of TransformBase, must have get_params());
    - optimization method out of the ones available in scipy.optimize.minimize.
    '''    
    def __init__(self, measure, transform, method='BFGS'):
        self.method = method
        self.transform = transform
        self.minimizer_options = {}
        self.param_scaling = None
        self.measure = measure

        self.step_length = 1.0
        self.end_step_length = None
        self.gradient_magnitude_threshold = 1e-6
        self.last_value = np.nan
#        self.last_grad = np.zeros((transform.get_param_count(),))
        self.iteration = 0
        self.termination_reason = ""
        self.report_freq = 0
        self.report_func = _default_report_callback
        self.value_history = []

    '''
    Depending on the choice of optimizer this may not have any effect
    '''
    def set_step_length(self, step_length, end_step_length = None):
        self.step_length = step_length
        self.end_step_length = end_step_length
    
    '''
    Depending on the choice of optimizer this may not have any effect
    '''
    def set_gradient_magnitude_threshold(self, gmt):
        self.gradient_magnitude_threshold = gmt
#    
#    def set_scaling(self, index, scale):
#        if self.param_scaling is None:
#            self.param_scaling = np.ones((self.transform.get_param_count(),))
#        self.param_scaling[index] = scale
#    

    '''
    Transform parameters will be multiplied by scalings (list of same length as params)
    in order to harmonize step lengths for optimizer
    '''
    def set_scalings(self, scalings):
        if self.param_scaling is None:
            self.param_scaling = scalings
        else:
            self.param_scaling[:] = scalings[:]
    
    def get_iteration(self):
        return self.iteration
    
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

    '''
    Set the measure to be minimized. This must have a get_value function which takes a transform, 
    and returns a distance measure, such that smaller distance = better registration
    '''
    def set_measure(self, measure):
        self.measure = measure

    '''
    Wrapper for function to minimize. This will be called by the optimizer, giving scaled params
    Rescale and do reporting if required
    '''
    def fun_to_minimize(self, params):

        self.transform.set_params(params * self.param_scaling)

        v, _ = self.measure.value_and_derivatives(self.transform)

        self.last_value = v
        self.iteration += 1

        if self.report_freq > 0 and (self.iteration % self.report_freq == 0) and self.report_func is not None:
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
        
    '''
    Set or update options dict which will be passed to the minimizer
    '''
    def set_minimizer_options(self, options, clear=True):
        if clear:
            self.minimizer_options = {}
        self.minimizer_options.update(options)
    
    '''
    Perform the optimization with an optional max number of iterations specified
    These are minimizer iterations (each iteration normally uses several function
    evaluations). Different from the iterations used for the reporting frequency 
    which are based on the number of function evaluations. TODO: fix this by
    adding a reporting callback on the minimize function
    '''
    def optimize(self, iterations=None):

        if iterations:
            self.minimizer_options['maxiter'] = iterations

        start_params = self.transform.get_params() / self.param_scaling
        
        x_opt = minimize(self.fun_to_minimize, start_params, \
                         method=self.method, \
                         options=self.minimizer_options)

        self.transform.set_params(x_opt.x * self.param_scaling)
        self.termination_reason = x_opt.message
        self.success = x_opt.success
        self.value_history = x_opt.fun
        
        print('%s optimizer terminated with status %s and message %s'%(self.method, self.success, self.termination_reason))
#        print(x_opt)

        return x_opt.nit
