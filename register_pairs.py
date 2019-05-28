#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 16:30:03 2019

@author: jo

A script to align image pairs with lots of parameters hard-coded.
"""

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
#py_alpha_amd imports
import sys, os, csv, time

import transforms, filters, models, optimizers
from PIL import Image
#from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score, mutual_info_score

import re #for matching filenames to a required pattern

local_aligned_folder = '../data/aligned_190508/'
local_mpm_folder = '../data/processed/'
server_aligned_folder = '/data/jo/MPM Skin Deep Learning Project/aligned_190508/'
server_mpm_folder = '/data/jo/MPM Skin Deep Learning Project/processed/'
local_sr_folder = '../data/struct_reprs/pca/'
server_sr_folder = '/data/jo/MPM Skin Deep Learning Project/struct_reprs/pca/'

def getNextPair(verbose=False, server=False):
    """Generator function to get next pair of matching images out of a complicated directory structure.
    """
    al_pattern = re.compile("^final\.tif aligned to ([0-9]+)+\.tif$", re.IGNORECASE)
    if server:
        aligned_folder = server_aligned_folder
        mpm_folder = server_mpm_folder
    else:
        aligned_folder = local_aligned_folder
        mpm_folder = local_mpm_folder
        
    aligned_subdirs = ['dysplastic/', 'malignant/', 'healthy/']


    for sd in aligned_subdirs:
        (_, slidedirs, _) = next(os.walk(aligned_folder+sd)) #get list of subdirs within this one

        for slide in slidedirs:
            (_, _, files) = next(os.walk(aligned_folder+sd+slide)) #get list of files within this folder

            for f in files:
                #Check each file in the subdir. If the file is an aligned file it will match the pattern above
                m = al_pattern.match(f)
                if m:
                    #Find the corresponding MPM image (only 148184 and 148185 currently exist)
                    roi_idx = m.group(1)
                    mpm_path = mpm_folder + slide + '_' + m.group(1) + '_gs.tif'
                    if os.path.isfile(mpm_path):
                        pass
                    else:
                        if verbose:
                            print("Unable to find MPM image", mpm_path)
                        continue

                    #If we found a mosaic then read the aligned image file
                    al_path = aligned_folder + sd + slide + '/' + f
                    if os.path.isfile(al_path): #should always be true since we got the f from the directory
                        pass
                    else:
                        if verbose:
                            print("Unable to find image file", al_path)
                        continue
                        
                    yield slide, roi_idx, mpm_path, al_path

def getNextSRPair(folder, verbose=False):
    """Generator function to get next pair of images from structural representations folder.
    
    Look through all the files in the given folder. For each one where the filename matches 
    the pattern, look for one that has the same name except with bf. If found, return the 
    paths and also the sample and region ids
    """
    pattern = re.compile("^([0-9]+)_([0-9]+)_mpm_.*$")
    (_, _, files) = next(os.walk(folder))
    for f in files:
        m = pattern.match(f)
        if not m: #Not an MPM file or at least, not one matching our pattern
            continue
        bf_path = folder + f.replace('mpm', 'bf')
        if os.path.isfile(bf_path):
            yield m.group(1), m.group(2), folder+f, bf_path
        else:
            if verbose:
                print('MPM file %s: matching bf %s not found'%(f, bf_path))
                    
def getRandomTransform(maxRot=10, maxScale=1.15, maxTrans=8):
    """
    Choose a random 2D transform (scale, rotation, and translation in two directions, where scale is a single value 
    applied in both x and y dimensions). Max rotation given in degrees (in either direction). Max translation in
    pixels and max scale in proportion.
    Return a transform object implementing it
    """
    highlimits = [maxRot, maxScale, maxTrans, maxTrans]
    lowlimits = [-l for l in highlimits]
    lowlimits[1] = 1/maxScale
    rot, scale, xtrans, ytrans = np.random.uniform(low=lowlimits, high=highlimits)
    
    scaleT = transforms.ScalingTransform(2, uniform=True)
    scaleT.set_params([scale,])
    rigidT = transforms.Rigid2DTransform()
    rigidT.set_params([np.pi*rot/180, xtrans, ytrans])
    
    return transforms.CompositeTransform(2, [scaleT, rigidT])


def get_transf_error(t1, t2, size):
    """
    Given two transforms, return the total euclidean distance between the 
    transformed positions of the four corner points of an image of size=(y, x) pixels
    """
    
    #The points to be transformed
    pts = [[0, 0],[0, size[0]],[size[1],0],[size[1],size[0]]]
    
    #Transform the 4 corners
    out1 = t1.transform(pts)
    out2 = t2.transform(pts)
#    print('Corner points', pts)
#    print('New posns under transform 1:', out1)
#    print('New posns under transform 2:', out2)
    
    #Get the distance
    dists = [np.abs(o1-o2) for o1, o2 in zip(out1, out2)]
#    print('Distances between pairs:', dists)
    
    return sum([np.sqrt(a*a + b*b) for a,b in dists])


def preProcessImageAndCopy(path):
    im = Image.open(path).convert('L')
    im = np.asarray(im)#[125:375,125:375] #Take a smaller region for speed
    
    # Keep unmodified copies of original images
    im_orig = im.copy()
    
    # Normalize
    im = filters.normalize(im, 0.0, None)
    
    # Normalizing gives us values between 0 and 1. Now quantize into 4 bits 
    # to reduce the amount noise in the mutual information calculation and
    # reduce the time taken to calculate it, too.
    # Instead, do this later on in the distance measure.  
#    im = np.rint(im*63).astype('uint8')
    
    return im, im_orig
    
def centreAndApplyTransform(image, transform, outsize, mode='nearest'):
        transformed_im = np.zeros(outsize, image.dtype)
        t = transforms.make_image_centered_transform(transform, \
                                                 transformed_im, image)
        t.warp(In = image, Out = transformed_im, mode=mode, bg_value = 0)
        return transformed_im
    
def register_pairs(server=False):
    results=[]
    limit = 100
    folder = local_sr_folder
    if server:
        folder = server_sr_folder

    for slide, roi_idx, mpm_path, al_path in getNextSRPair(folder):
        limit -= 1
        if limit < 0:
            break
        # Open and prepare the images
        ref_im, ref_im_orig = preProcessImageAndCopy(al_path)
        flo_im, flo_im_orig = preProcessImageAndCopy(mpm_path)
#        flo_im = flo_im[20:-20, 20:-20] #crop floating image so that it stays within ref
    
        # Choose a transform at random
#        rndTrans = getRandomTransform(maxRot=5,maxScale=1.025,maxTrans=2)
#        print("Random transform applied: %r"%rndTrans.get_params())
    
        # Apply the transform to the reference image, increasing the canvas size to avoid cutting off parts
#        ref_im = centreAndApplyTransform(ref_im, rndTrans, np.rint(np.array(ref_im.shape)*1.5).astype('int'))
        
        # Show the images we are working with
        print("Aligning images for sample %s, region %s"%(slide, roi_idx) \
#              + ". A transform of %r has been applied to the reference image"%str(rndTrans.get_params()))
               + " from folder %s"%folder)
        plt.figure(figsize=(12,6))
        plt.subplot(121)
        plt.imshow(ref_im, cmap='gray', vmin=0, vmax=1)
        plt.title("Reference image")
        plt.subplot(122)
        plt.imshow(flo_im, cmap='gray', vmin=0, vmax=1)
        plt.title("Floating image")
        plt.show()
    
        # Choose an optimizer to find the position maximising the mutual information
        reg = models.RegisterAlphaAMD(2)
        reg.set_optimizer('adam')
        
        # Since we have warped the original reference image, create a mask so that only the relevant
        # pixels are considered. Use the same warping function as above
#        ref_mask = np.ones(ref_im_orig.shape, 'bool')
#        ref_mask = centreAndApplyTransform(ref_mask, rndTrans, np.rint(np.array(ref_im_orig.shape)*1.5).astype('int'))
        
        reg.set_image_data(ref_im, \
                           flo_im, \
#                           ref_mask=ref_mask, \
                           ref_mask=np.ones(ref_im.shape, 'bool'), \
                           flo_mask=np.ones(flo_im.shape, 'bool'), \
                           ref_weights=None, \
                           flo_weights=None
                           )
        
        t = transforms.CompositeTransform(2, [transforms.ScalingTransform(2, uniform=True), \
                                              transforms.Rigid2DTransform()])
        #MI
#        reg.set_mutual_info_fun('normalized')
#        reg.add_initial_transform(t)
#        # There is no evidence so far, that pyramid levels lead the search towards the MI maximum.
#        reg.add_pyramid_levels(factors=[1, ], sigmas=[0.0,])
#        # Try with a blurred full-resolution image first (or only)
#        reg.add_pyramid_levels(factors=[1,1], sigmas=[5.0,5.0])

        #AlphaAMD
        # Estimate an appropriate parameter scaling based on the sizes of the images (not used in grid search).
        reg.set_sampling_fraction(0.1)
        reg.set_iterations(1000)
        diag = transforms.image_diagonal(ref_im) + transforms.image_diagonal(flo_im)
        diag = 2.0/diag
        reg.add_initial_transform(t, param_scaling=np.array([diag, diag, 1.0, 1.0]))
        # Learning-rate / Step lengths [[start1, end1], [start2, end2] ...] (for each pyramid level)
        step_lengths = np.array([[1., 1.], [1., 0.5], [0.5, 0.1]])
        reg.set_step_lengths(step_lengths)
        reg.set_gradient_magnitude_threshold(1e-6)
        reg.set_alpha_levels(7)
        reg.add_pyramid_levels(factors=[4, 2, 1], sigmas=[5.0, 3.0, 0.0])


        reg.set_report_freq(500)
    
        # Create output directory
        directory = os.path.dirname("./tmp/")
        if not os.path.exists(directory):
            os.makedirs(directory)
    
        # Start the pre-processing
        reg.initialize("./tmp/")
        
        # Start the registration
        reg.run()
    
        (transform, value) = reg.get_output(0)
    
        ### Warp final image
        c = transforms.make_image_centered_transform(transform, ref_im, flo_im)
    
    #    # Print out transformation parameters
    #    print('Transformation parameters: %s.' % str(transform.get_params()))
    
        # Create the output image
        im_warped = np.zeros(ref_im.shape)
    
        # Transform the floating image into the reference image space by applying transformation 'c'
        c.warp(In = flo_im_orig, Out = im_warped, mode='spline', bg_value = 0.0)
    
        # Show the images we ended up with
        print("Aligned images for sample %s, region %s"%(slide, roi_idx))
        plt.figure(figsize=(12,6))
        plt.subplot(121)
        plt.imshow(ref_im, cmap='gray')
        plt.title("Reference image")
        plt.subplot(122)
        plt.imshow(im_warped, cmap='gray')
        plt.title("Floating image")
        plt.show()

#        centred_gt_trans = transforms.make_image_centered_transform(rndTrans, \
        centred_gt_trans = transforms.make_image_centered_transform(transforms.IdentityTransform(2), \
                                                 ref_im, flo_im)
        err = get_transf_error(c, centred_gt_trans, flo_im.shape)
        print("Estimated transform:\t [", ','.join(['%.4f']*len(c.get_params()))%tuple(c.get_params())+"]")
#        print("True transform:\t\t [", ','.join(['%.4f']*len(rndTrans.get_params()))%tuple(rndTrans.get_params())+"]")
        print("Error: %5f"%err)
        
        resultLine = (slide, roi_idx, *c.get_params(), err, -value)
#        resultLine = (slide, roi_idx, *rndTrans.get_params(), *c.get_params(), err)
        results.append(resultLine)
    
    
    #print(results)
    if server:
        outfile = server_aligned_folder+'PCANetSR_alphaAMD_adam.csv'
    else:
        outfile = local_aligned_folder+'PCANetSR_alphaAMD_adam.csv'
    with open(outfile, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(["Slide", "Region", "Est_scale", "Est_rot", "Est_x", "Est_y", "Error", "Value"])
#        writer.writerow(["Slide", "Region", "GT_scale", "GT_rot", "GT_x", "GT_y", "Est_scale", "Est_rot", "Est_x", "Est_y", "Error"])
        writer.writerows(results)


def create_MI_surfaces(server=False):
    results=[]
    limit = 1
    for slide, roi_idx, mpm_path, al_path in getNextPair():
        limit -= 1
        if limit < 0:
            break
        # Open and prepare the images
        ref_im, ref_im_orig = preProcessImageAndCopy(al_path)
        flo_im, flo_im_orig = preProcessImageAndCopy(mpm_path)

        reg = models.RegisterMI(2)
        reg.set_optimizer('gridsearch')
        reg.set_mutual_info_fun('normalized')
        
        # Since we have warped the original reference image, create a mask so that only the relevant
        # pixels are considered. Use the same warping function as above
        
        reg.set_image_data(ref_im, \
                           flo_im, \
                           ref_mask=np.ones(ref_im.shape, 'bool'), \
                           flo_mask=np.ones(flo_im.shape, 'bool'), \
                           ref_weights=None, \
                           flo_weights=None
                           )

        reg.add_pyramid_levels(factors=[1,], sigmas=[5.0,])

        t = transforms.CompositeTransform(2, [transforms.ScalingTransform(2, uniform=True), \
                                              transforms.Rigid2DTransform()])
        reg.add_initial_transform(t)
        reg.set_report_freq(500)
    
        # Create output directory
        directory = os.path.dirname("./tmp/")
        if not os.path.exists(directory):
            os.makedirs(directory)
    
        # Start the pre-processing
        reg.initialize("./tmp/")
        
        # Start the registration
        reg.run()
        
        values = - np.array(reg.get_value_history(0, 0))

        if True: #2d surface
            pts = 81
            values.resize((pts,pts))
            
            plt.figure(figsize=(8,8))
#            plt.contourf(np.linspace(0.8, 1.2, pts), np.linspace(-0.25*np.pi, 0.25*np.pi, pts), values)
            plt.contourf(np.linspace(-20, 20, pts), np.linspace(-20, 20, pts), values)
            plt.colorbar()
            plt.title("Mutual Information surface at offset scale and rotation\n"\
                      +"with different translations (smoothed)")
#            plt.xlabel('Scale')
#            plt.ylabel('Rotation')
            plt.show()



        
if __name__ == '__main__':
    start_time = time.time()
    server_paths = False
    if len(sys.argv) > 1:
        server_paths = True
    register_pairs(server_paths)
#    create_MI_surfaces(server_paths)
    end_time = time.time()
    print("Elapsed time: " + str((end_time-start_time)))