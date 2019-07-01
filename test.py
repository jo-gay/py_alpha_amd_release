#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 11:14:53 2019

@author: jo


Test dLDP and LDP functionality
"""

import unittest
import numpy as np

from distances import dLDPDistance

class TestLDP(unittest.TestCase):
    def setUp(self):
        self.size = (5,6)

        self.im1 = np.ones(self.size, dtype='float')
        self.ldp1 = np.zeros((*self.size,32), dtype='int')
        self.dldp1 = np.zeros((*self.size,48), dtype='int')

        self.mask1 = np.ones(self.size, dtype='bool')
        
        #Outer border of two rows and columns will be masked out. The derivatives
        #may not be available for the outermost row/column, so the LDP for cells
        #neighbouring the outer row/column cannot be trusted
        self.mask1[0:2,:] = False
        self.mask1[-2:,:] = False
        self.mask1[:,0:2] = False
        self.mask1[:,-2:] = False
        
        self.im1[2,3] = 0
        self.ldp1[2,2,...]=[0,0,0,1,0,0,0,0]*4
        self.ldp1[2,3,...]=[1,1,1,1,1,1,1,1]*4
        
        #Old 2-way versions
#        self.dldp1[2,2,...]=[0,0,0,1,0,0,0,0]*2
#        self.dldp1[2,3,...]=[1,1,1,1,1,1,1,1]*2
        
        #new 4-way versions
        self.dldp1[2,2,...]=[0,0,0,1,0,0,0,0]*6
        self.dldp1[2,3,...]=[1,1,1,1,1,1,1,1]*6
        
    def test_LDP(self):
        LDP = dLDPDistance(mode='diff', interpolation='nearest')
        ldp1, mask1 = LDP.create_LDP(self.im1)
        
        self.assertTrue(np.alltrue(mask1 == self.mask1))
        self.assertTrue(np.alltrue(ldp1[mask1,:] == self.ldp1[self.mask1,:]))

    def test_dLDP(self):
        dLDP = dLDPDistance(mode='diff', interpolation='nearest')
        dldp1, mask1 = dLDP.create_dLDP(self.im1)
        
        self.assertTrue(np.alltrue(mask1 == self.mask1))
        self.assertTrue(np.alltrue(dldp1[mask1,:] == self.dldp1[self.mask1,:]))
        
if __name__ == '__main__':
    unittest.main()
