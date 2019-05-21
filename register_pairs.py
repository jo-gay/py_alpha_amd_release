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
import sys, os, csv

import transforms, filters, models, optimizers
from PIL import Image
#from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score, mutual_info_score

import re #for matching filenames to a required pattern


def getNextPair(verbose=False):
    """Generator function to get next pair of matching images out of a complicated directory structure.
    """
    al_pattern = re.compile("^final\.tif aligned to ([0-9]+)+\.tif$", re.IGNORECASE)
    aligned_folder = '/data/jo/MPM Skin Deep Learning Project/aligned_190508'
    aligned_subdirs = ['dysplastic/', 'malignant/', 'healthy/']

    mpm_folder = '/data/jo/MPM Skin Deep Learning Project/processed/'

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


"""MAIN CONTROL LOOP"""
results=[]
for slide, roi_idx, mpm_path, al_path in getNextPair():
    
    ref_im = Image.open(al_path).convert('L')
    flo_im = Image.open(mpm_path).convert('L')
    ref_im = np.asarray(ref_im)[200:400,200:400] #Take a smaller region for now
    flo_im = np.asarray(flo_im)[200:400,200:400]
    
    # Keep unmodified copies of original images
    ref_im_orig = ref_im.copy()
    flo_im_orig = flo_im.copy()
    
    # Normalize
    ref_im = filters.normalize(ref_im, 0.0, None)
    flo_im = filters.normalize(flo_im, 0.0, None)

    # Choose a transform at random
    rndTrans = getRandomTransform(5,1.05,5)

    # Apply the transform to the reference image
    transformed_ref_im = np.zeros(np.rint(np.array(ref_im.shape)*1.5).astype('int'))#increase the canvas size to avoid cutting off too much
    rndTrans = transforms.make_image_centered_transform(rndTrans, \
                                             transformed_ref_im, ref_im)
    rndTrans.warp(In = ref_im, Out = transformed_ref_im, mode='nearest', bg_value = 0)
    ref_im = transformed_ref_im
    
#    # Show the images we are working with
#    print("Aligning images for sample %s, region %s"%(slide, roi_idx) + \
#          ". A transform of %r has been applied to the reference image"%str(rndTrans.get_params()))
#    plt.figure(figsize=(12,6))
#    plt.subplot(121)
#    plt.imshow(transformed_ref_im, cmap='gray', vmin=0, vmax=1)
#    plt.title("Reference image")
#    plt.subplot(122)
#    plt.imshow(flo_im, cmap='gray', vmin=0, vmax=1)
#    plt.title("Floating image")
#    plt.show()

    # Use a grid search to find the position maximising the mutual information
    reg = models.RegisterMI(2)
    reg.set_mutual_info_fun('normalized')
    
    # Generic initialization steps required for every registration model
    mask1 = np.ones(ref_im_orig.shape, 'bool')
    ref_mask = np.zeros(ref_im.shape, 'bool')
    rndTrans.warp(In = mask1, Out = ref_mask, mode='nearest', bg_value = 0)
    mask2 = np.ones(flo_im.shape, 'bool')

    reg.set_reference_image(ref_im)
    reg.set_reference_mask(ref_mask)

    reg.set_floating_image(flo_im)
    reg.set_floating_mask(mask2)

    weights1 = np.ones(ref_im.shape)
    weights2 = np.ones(flo_im.shape)
    reg.set_reference_weights(weights1)
    reg.set_floating_weights(weights2)

    # Setup the Gaussian pyramid resolution levels
    reg.add_pyramid_level(4, 5.0)
    reg.add_pyramid_level(2, 3.0)
    reg.add_pyramid_level(1, 0.0)

    # Learning-rate / Step lengths [[start1, end1], [start2, end2] ...] (for each pyramid level)
    step_lengths = np.array([[1., 1.], [1., 0.5], [0.5, 0.1]])
    reg.set_step_lengths(step_lengths)
    
    # Estimate an appropriate parameter scaling based on the sizes of the images.
    diag = transforms.image_diagonal(ref_im) + transforms.image_diagonal(flo_im)
    diag = 2.0/diag
    
    t = transforms.CompositeTransform(2, [transforms.ScalingTransform(2, uniform=True), \
                                          transforms.Rigid2DTransform()])
    reg.add_initial_transform(t, param_scaling=np.array([diag, diag, 1.0, 1.0]))

    reg.set_optimizer('gridsearch')
    reg.set_report_freq(5000)

    # Create output directory
    directory = os.path.dirname("./tmp")
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Start the pre-processing
    reg.initialize("./tmp")
    
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

#    # Show the images we ended up with
#    print("Aligned images for sample %s, region %s"%(slide, roi_idx))
#    plt.figure(figsize=(12,6))
#    plt.subplot(121)
#    plt.imshow(ref_im, cmap='gray')
#    plt.title("Reference image")
#    plt.subplot(122)
#    plt.imshow(im_warped, cmap='gray')
#    plt.title("Floating image")
#    plt.show()
#
#    print("Estimated transform:", c.get_params())
#    print("True transform:", rndTrans.get_params())
    
    err = get_transf_error(c, rndTrans, flo_im.shape)
    
    resultLine = (slide, roi_idx, *rndTrans.get_params(), *c.get_params(), err)
    results.append(resultLine)


#print(results)

outfile = '/data/jo/MPM Skin Deep Learning Project/aligned_190508/recovered_synthetic_trans_mi.csv'
with open(outfile, 'w') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(["Slide", "Region", "GT_scale", "GT_rot", "GT_x", "GT_y", "Est_scale", "Est_rot", "Est_x", "Est_y", "Error"])
    writer.writerows(results)
        