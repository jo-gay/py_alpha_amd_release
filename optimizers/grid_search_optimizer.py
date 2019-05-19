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
# Grid search optimizer class
#

import numpy as np


def _default_report_callback(opt):
    """Define the default callback for reporting on the progress of the optimizer."""
    iteration = opt.get_iteration()
    value = opt.get_value()
    param = opt.get_transform().get_params()
    totalIts = np.prod(list(map(len, opt.get_steps())))
    strParam = ','.join(["%5g"]*len(param))%tuple(param)
    print("#%d (%4.1f%%). --- Value: %10f" % (iteration, 100*iteration/totalIts, value) + ", Param: (%s)"%(strParam))


def _appendItems(currlist, suffixlist):
    """Generator function to yield the next set of parameters.
    
    Given a list of lists, and a list of suffixes, yield the list plus the suffix 
    for each entry in suffixlist and in list.
    e.g. currlist=[[1,2,3],[1,2,4]] and suffixlist=[0,1] will yield
    [1,2,3,0],[1,2,3,1],[1,2,4,0], and [1,2,4,1] in that order
    """
    for l in currlist:
        for s in suffixlist:
            yield l + [s]


class GridSearchOptimizer:
    """Calculate (dis)similarity between two images at each point on a grid of possible transforms.
    """    
    def __init__(self, measure, transform, bounds, steps):
        """Set up the grid search optimizer.
        
        Args:
            measure: a way to compare two images. Optimizer calls measure.get_value() at each grid point 
            and returns the grid point with the lowest value. This should be initialized with images etc.
            transform: instance of a subclass of TransformBase, must have get_params() & set_params() methods.
            bounds: a list of same length as transform.get_params(), each entry is a min/max tuple for 
            that parameter
            steps: either a single integer, in which case the grid has that many steps equally spaced between 
            bounds[0] and bounds[1] inclusive in each dimension, or a list of integers of the same length 
            as transform.get_params(), in which case the grid has the specified number of steps in each 
            dimension (also linearly spaced between the bounds), or a list of lists giving the exact points 
            required in each dimension. In the latter case the contents of bounds will be ignored
        """
        self.measure = measure
        self.transform = transform
        self.last_value = np.nan
        self.best_value = np.inf
        self.iteration = 0
        self.report_freq = 0
        self.report_func = _default_report_callback
        self.value_history = []
        self.steps = []
        self.set_steps(bounds, steps)

    ''' compatibility functions '''
    def set_step_length(self, *args, **kwargs):
        """Not used in this optimizer."""
        pass
    def set_scalings(self, *args, **kwargs):
        """Not used in this optimizer."""
        pass
    def set_gradient_magnitude_threshold(self, *args, **kwargs):
        """Not used in this optimizer."""
        pass

    ''' reporting functions'''    
    def set_report_freq(self, freq):
        """Set or update the frequency with which the report callback function will be called."""
        if freq < 0:
            freq = 0
        self.report_freq = freq
    
    def set_report_callback(self, func, additive = True):
        """Set or update the report callback function.
        
        Args:
            func: the reporting function to use
            additive: if True, existing reporting function will also be called (default True)
        """
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
            
    def get_iteration(self):
        """Return the number of iterations completed so far."""
        return self.iteration

    def get_value(self):
        """Return the most recent measure value that has been found."""
        return self.last_value

    def get_value_history(self):
        """Return the list of all measure values that have been found."""
        return self.value_history
    
    def get_transform(self):
        """Return the current transform. At the end of the grid search this will be set to the best transform found."""
        return self.transform
    
    '''real functions for grid search optimizer '''
    def get_steps(self):
        """Return the list of points that will be evaluated for each parameter."""
        return self.steps
    
    def set_steps(self, bounds, steps):
        """Set the paramater values to be searched in each dimension.
        
        Args:
            bounds: a list containing a min/max tuple for each parameter
            steps: either a single integer, in which case the grid has that many steps equally spaced between 
            bounds[0] and bounds[1] inclusive in each dimension,
            or a list of integers of the same length as transform.get_params(), in which case the grid has
            the specified number of steps in each dimension (also linearly spaced between the bounds),
            or a list of lists giving the exact points required in each dimension. In the latter case
            the contents of bounds will be ignored
        """
        if type(steps) == int:
            self.steps = [np.linspace(b1,b2,steps) for b1,b2 in bounds]
        elif type(steps) == list and type(steps[0]) == int:
            self.steps = [np.linspace(b1, b2, s) for (b1, b2), s in zip(bounds, steps)]
        else:
            self.steps = steps.copy()

    def next_grid_pt(self):
        """Generator function to return the next grid point to be evaluated.
        
        Note: this generates all of the combinations of parameters and returns them one by one. Uses
        a lot of memory. New better version below.
        """
        combos=[[]]
        #For each parameter, add all of the possible options for that parameter,
        #pre-calculated in self.steps[p], to the end of each of the parameter
        #combinations that have been found so far, that include the previous parameters only.
        for p in self.steps:
            combos = _appendItems(combos, p)

        for result in combos:
            yield result
            
    def get_next_point(self):
        """Generator function to return the next set of parameters to evaluate
        """
        #Get the index of the current step in each dimension
        nparams = len(self.transform.get_params())
        indices = [0]*nparams
        #Get the number of steps in each dimension
        lengths = [len(self.steps[i]) for i in range(nparams)]

        end = False
        while not end:
            yield [self.steps[i][indices[i]] for i in range(nparams)]

            #Increment the index of the last paramenter and then check whether it goes over the end
            indices[-1] += 1
            for p in reversed(range(nparams)):
                if indices[p] == lengths[p]:
                    indices[p] = 0
                    if p > 0:
                        indices[p-1] += 1
                    else:
                        end = True

    def optimize(self, iterations=None):
        """Perform the grid search.
        
        Args:
            iterations: argument is kept for compatibility with other optimizers
            but is ignored here
            
        Returns:
            value found at best transform on grid, according to the provided distance measure
        """
        for params in self.get_next_point():
            self.transform.set_params(params)

            v, _ = self.measure.value_and_derivatives(self.transform)

            if v < self.best_value:
                self.best_value = v
                self.best_params = params
#                print('New best value %2.4f at ('%v, ', '.join(['%8.3f']*len(params))%tuple(params), ')')

            self.value_history.append(v)
            self.last_value = v
            self.iteration += 1

            if self.report_freq > 0 and (self.iteration % self.report_freq == 0) and self.report_func is not None:
                self.report_func(self)

        
        self.transform.set_params(self.best_params)
        return self.best_value

