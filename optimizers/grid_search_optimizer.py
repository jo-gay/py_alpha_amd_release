#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 16:26:05 2019

@author: jo
"""
import numpy as np


def _default_report_callback(opt):
    iteration = opt.get_iteration()
    value = opt.get_value()
    param = opt.get_transform().get_params()
    print("#%d. --- Value: " % (iteration) + str(value) + ", Param: " + str(param))


'''
Generator function to yield the next set of parameters

Given a list of lists, and a list of suffixes, yield the list plus the suffix 
for each entry in suffixlist and in list.
e.g. currlist=[[1,2,3],[1,2,4]] and suffixlist=[0,1] will yield
[1,2,3,0],[1,2,3,1],[1,2,4,0],[1,2,4,1] in some order
'''
def appendItems(currlist, suffixlist):
    for l in currlist:
        for s in suffixlist:
            yield l + [s]


class GridSearchOptimizer:
    '''
    Initialize optimizer with 
    - measure (must have a function get_value() taking a transform object and returning
               a distance measure, such that smaller distance = better registration,
               should be initialized with image data etc);
    - transform (usually subclass of TransformBase, must have get_params());
    - bounds (list of same length as transform.get_params(), each entry is a min/max tuple
    - steps (either a single integer, in which case grid has this many steps in each dimension,
             or a list of integers of the same length as transform.get_params(),
             or a list of lists giving the exact points required in each dimension. In the latter case
             the contents of bounds will be ignored)
    '''    
    def __init__(self, measure, transform, bounds, steps):
        self.measure = measure
        self.transform = transform
        self.last_value = np.nan
        self.best_value = -np.inf
        self.iteration = 0
        self.report_freq = 0
        self.report_func = _default_report_callback
        self.value_history = []
        self.steps = []
        self.set_steps(bounds, steps)

    ''' compatibility functions '''
    def set_step_length(self, *args, **kwargs):
        pass
    def set_scalings(self, *args, **kwargs):
        pass
    def set_gradient_magnitude_threshold(self, *args, **kwargs):
        pass

    ''' reporting functions'''    
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
    
    '''real functions for grid search optimizer '''
    def get_steps(self):
        return self.steps
    
    def set_steps(self, bounds, steps):
        if type(steps) == int:
            self.steps = [np.linspace(b1,b2,steps) for b1,b2 in bounds]
        elif type(steps) == list and type(steps[0]) == int:
            self.steps = [np.linspace(b1, b2, s) for (b1, b2), s in zip(bounds, steps)]
        else:
            self.steps = steps.copy()

    '''
    Generator function. Returns the next grid point to be evaluated
    '''
    def next_grid_pt(self):
        combos=[[]]
        #For each parameter, add all of the possible options for that parameter,
        #pre-calculated in self.steps[p], to the end of each of the parameter
        #combinations that have been found so far, that include the previous parameters only.
        for p in self.steps:
            combos = appendItems(combos, p)

        for result in combos:
            yield result


    '''
    Perform the grid search (iterations is kept for compatibility with other optimizers
    but is ignored here)
    '''
    def optimize(self, iterations=None):

        for params in self.next_grid_pt():
            self.transform.set_params(params)

            v, _ = self.measure.value_and_derivatives(self.transform)

#            if v > self.best_value:
#                self.best_value = v
#                self.best_params = params
#
            self.value_history.append(v)
            self.last_value = v

            if self.report_freq > 0 and (self.iteration % self.report_freq == 0) and self.report_func is not None:
                self.report_func(self)
        
        self.transform.set_params(self.measure.best_trans.get_params())
        return self.measure.best_val

#    
#if __name__ == '__main__':
#    g = GridSearchOptimizer(0, None, [(0,1),(10,20),(-3,0)], 5)
#    for pt in g.next_grid_pt():
#        print('Calculating MI at', pt)
#    