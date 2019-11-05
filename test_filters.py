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

#
# Testing image filters
#

import unittest
import numpy as np
from matplotlib import pyplot as plt

import filters


class TestImFilt(unittest.TestCase):
    def setUp(self):
        #Test image per https://en.wikipedia.org/wiki/Histogram_equalization (retrieved 2019-11-05)
        self.test_im = np.array([[52,55,61,59,79,61,76,61],\
                        [62,59,55,104,94,85,59,71],\
                        [63,65,66,113,144,104,63,72],\
                        [64,70,70,126,154,109,71,69],\
                        [67,73,68,106,122,88,68,68],\
                        [68,79,60,70,77,66,58,75],\
                        [69,85,64,58,55,61,65,83],\
                        [70,87,69,68,65,73,78,90]
                       ])/255.
        self.hist_equalized = np.array([[0,12,53,32,190,53,174,53],\
                               [57,32,12,227,219,202,32,154],\
                               [65,85,93,239,251,227,65,158],\
                               [73,146,146,247,255,235,154,130],\
                               [97,166,117,231,243,210,117,117],\
                               [117,190,36,146,178,93,20,170],\
                               [130,202,73,20,12,53,85,194],\
                               [146,206,130,117,85,166,182,215]
                              ])
    
        self.test_im2 = np.array([list(range(255))]*255)/255.


    def test_hist_eq(self):
        """Test histogram equalization function. Todo: test different numbers of bins."""
        equalized = filters.image_histogram_equalization(self.test_im)
        self.assertTrue(np.equal(np.rint(equalized*255.), self.hist_equalized).all())

    def test_sp_noise(self):
        """Test salt and pepper noise by displaying some images and checking the number
        of altered pixels is similar to that requested."""
        noise_amts = [0.001, 0.005, 0.01, 0.1]
        for i, amt in enumerate(noise_amts):
            plt.subplot(2,2,i+1)
            sp_noisy = filters.add_noise(self.test_im2, 's&p', sp_amount=amt)
            diffs = np.equal(self.test_im2, sp_noisy).flatten()
            diffs = (len(diffs) - sum(diffs))/float(len(diffs))
            plt.imshow(sp_noisy, cmap='gray', vmin=0, vmax=1)
            self.assertAlmostEqual(amt, diffs, delta=amt/10)
        plt.suptitle("Salt and Pepper noise")
        plt.show()

    def test_gauss_noise(self):
        """Test gaussian noise by displaying some images for visual checking.
        
        Todo: Add proper tests of noise amounts and distribution.
        """
        noise_amts = [0.01, 0.1, 0.2, 0.5]
        for i, amt in enumerate(noise_amts):
            plt.subplot(2,2,i+1)
            sp_noisy = filters.add_noise(self.test_im2, 'gauss', gauss_amount=amt)
#            diffs = np.equal(self.test_im2, sp_noisy).flatten()
#            diffs = (len(diffs) - sum(diffs))/float(len(diffs))
            plt.imshow(sp_noisy, cmap='gray', vmin=0, vmax=1)
#            self.assertAlmostEqual(amt, diffs, delta=amt/10)
        plt.suptitle("Gaussian noise")
        plt.show()
            
if __name__ == '__main__':
    unittest.main()
