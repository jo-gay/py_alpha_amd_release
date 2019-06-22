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

from register import Register
import transforms, filters
from PIL import Image

import re #for matching filenames to a required pattern

local_aligned_folder = '../data/aligned_190508/'
local_mpm_folder = '../data/processed/'
local_separate_mpm_folder = '../data/unprocessed/'
local_sr_folder = '../data/struct_reprs/pca/'
server_aligned_folder = '/data/jo/MPM Skin Deep Learning Project/aligned_190508/'
server_mpm_folder = '/data/jo/MPM Skin Deep Learning Project/processed/'
server_separate_mpm_folder = '/data/jo/MPM Skin Deep Learning Project/'
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
    return

def getNextSRPair(folder, verbose=False):
    """Generator function to get next pair of images from structural representations folder.
    
    Look through all the files in the given folder. For each one where the filename matches 
    the pattern, look for one that has the same name except with bf. If found, return the 
    paths and also the sample and region ids
    """
#    pattern = re.compile("^([0-9]+)_([0-9]+)_mpm_.*$")
    pattern = re.compile("^psr_shg_([0-9a-zA-Z]+)_([0-9]+).tif$")
    (_, _, files) = next(os.walk(folder))
    for f in files:
        m = pattern.match(f)
        if not m: #Not an MPM file or at least, not one matching our pattern
            continue
        
#        bf_path = folder + f.replace('mpm', 'bf')
#        if os.path.isfile(bf_path):
#            yield m.group(1), m.group(2), folder+f, bf_path
#        else:
#            if verbose:
#                print('MPM file %s: matching bf %s not found'%(f, bf_path))
        tpef_path = folder + f.replace('shg', 'tpef')
        if os.path.isfile(tpef_path):
            yield m.group(1), m.group(2), folder+f, tpef_path
        else:
            if verbose:
                print('SHG file %s: matching TPEF %s not found'%(f, tpef_path))
    return
                
def getNextMPMPair(verbose=False, server=False, norm=False, blur=0.0):
    """Generator function to get next pair of matching images out of a complicated directory structure.
    
    The SHG and two TPEF images are stored in the same folder. The two TPEF images need to be averaged.
    
    NOTE: unlike the above functions which return the path to the images, this one reads and returns
    the images themselves.
    """
    shg_pattern = re.compile("^([0-9]+)-shg.tif$", re.IGNORECASE)
    if server:
        folder = server_separate_mpm_folder
    else:
        folder = local_separate_mpm_folder
        
    subdirs = ['Dysplastic/', 'Malignant/', 'Healthy/']

    for sd in subdirs:
        (_, slidedirs, _) = next(os.walk(folder+sd)) #get list of subdirs within this one

        for slide in slidedirs:
            full_path = folder + sd + slide + '/med/'
            (_, _, files) = next(os.walk(full_path)) #get list of files within this folder

            for f in files:
                #Check each file in the subdir. If the file is an aligned file it will match the pattern above
                m = shg_pattern.match(f)
                if m:
                    #Find the corresponding MPM image (only 148184 and 148185 currently exist)
                    roi_idx = m.group(1)
                    tpef1_path = full_path + roi_idx + '-tpef1.tif'
                    tpef2_path = full_path + roi_idx + '-tpef2.tif'
                    if os.path.isfile(tpef1_path) and os.path.isfile(tpef2_path):
                        pass
                    else:
                        if verbose:
                            print("Unable to find TPEF images", tpef1_path, tpef2_path)
                        continue

                    #If we found a set of three then read them and average the TPEFs
                    shg, _ = preProcessImageAndCopy(full_path+f, norm=norm, blur=blur)
                    tpef1, _ = preProcessImageAndCopy(tpef1_path, norm=norm, blur=blur)
                    tpef2, _ = preProcessImageAndCopy(tpef2_path, norm=norm, blur=blur)
                        
                    yield slide, roi_idx, shg, (tpef1+tpef2)/2
    
    # Finished going through the files so end the generator
    return
                    
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



def chooseGrid(centreTrans):
    """Choose a reasonable range for each parameter, centered around the transform given.
    
    Args:
        centreTrans is a transform object of the type required (only affine and composite(scale, rigid2d)
        are handled so far). The grid will be centred around the parameters in this transform.
        
    Returns:
        bounds gives the min and max for each parameter
        steps is the number of points that should be linearly spaced between these bounds. If an integer,
        the same number of steps is used in each dimension. If a list, it gives the number of steps for
        each dimension.
        
    Note: The total number of function evaluations is steps to the power of 6 for affine or steps^4 for 
    a composite scale+rigid transform. (or np.prod(steps) if steps varies). Running time increases dramatically
    with additional steps.
    """
    centre = centreTrans.get_params()
    #Affine parameter bounds:
    if len(centre) == 6:
        bounds = np.array([[-0.1, 0.1], [-0.25, 0.25], [-0.25, 0.25], [-0.1, 0.1], [-5, 5], [-5, 5]])
        steps = 11
    #Composite (scale + rigid2d) parameter bounds:
    elif len(centre) == 4:
        bounds = np.array([[-0.05, 0.05], [-5*np.pi/180, 5*np.pi/180], [-10, 10], [-10, 10]])
        steps = 11
    
    if False: #Enable for MI surfaces
        bounds = np.array([[0, 0], [0, 0], [-20, 20], [-20, 20]])
        steps = [1,1,81,81]

    bounds = [bounds[i] + centre[i] for i in range(len(centre))]

    return bounds, steps


#def MI_at(reg, transform, ref, ref_mask, flo):
#    """ Calculate the Mutual Information at a particular transform.
#    """
#    midist = distances.MIDistance(reg.mi_fun, levels=32)
#    midist.set_ref_image(ref, ref_mask)
#    midist.set_flo_image(flo)
#    v, _ = midist.value_and_derivatives(transform)
#    return v
#
#def SSD_at(reg, transform, ref, ref_mask, flo):
#    """ Calculate the Mean of Squares Difference at a particular transform.
#    """
#    ssdist = distances.SSDistance()
#    ssdist.set_ref_image(ref, ref_mask)
#    ssdist.set_flo_image(flo)
#    v, _ = ssdist.value_and_derivatives(transform)
#    return v
#
#def alphaAMD_at(reg, transform, ref, ref_mask, flo):
#    """ Calculate the alphaAMD metric at a particular transform.
#    """
#    v, _ = reg.measure.value_and_derivatives(transform)
#    return v

def preProcessImageAndCopy(path, norm=True, blur=None):
    im = Image.open(path).convert('L')
    im = np.asarray(im)#[125:375,125:375] #Take a smaller region for speed
    
    # Keep unmodified copies of original images
    im_orig = im.copy()
    
    if blur and blur > 0:
        im = filters.gaussian_filter(im, blur)
    # Normalize
    if norm:
        im = filters.normalize(im, 0.0, None)
    else:
        im = im/255. #convert to floats without normalizing
    
    # Normalizing gives us values between 0 and 1. Now quantize into 4 bits 
    # to reduce the amount noise in the mutual information calculation and
    # reduce the time taken to calculate it, too.
    # Instead, do this later on in the distance measure.  
#    im = np.rint(im*63).astype('uint8')
    
    return im, im_orig

def get_MI_gridmax(mi_grid_resultsfile):
    """Read a file containing grid search results and return dictionary of max values.
    
    Args:
        slide, region: Together these identify the images being aligned
    """
    with open(mi_grid_resultsfile, 'r') as f:
        rdr = csv.reader(f, delimiter=',')
        mi_gridmax = {(row[0], row[1]): row[7:12] for row in rdr}

    return mi_gridmax
            
    
def centreAndApplyTransform(image, transform, outsize, mode='nearest'):
        transformed_im = np.zeros(outsize, image.dtype)
        t = transforms.make_image_centered_transform(transform, \
                                                 transformed_im, image)
        t.warp(In=image, Out=transformed_im, mode=mode, bg_value=0)
        return transformed_im

def add_multiple_startpts(reg, count=5, p_scaling=None):
    for _ in range(count):
        reg.add_initial_transform(getRandomTransform(maxRot=5, maxScale=1.05, maxTrans=10),\
                                  param_scaling=p_scaling)
    
    
def register_pairs(server=False):

    results=[]
    skip = 0 #25 #manual way to skip pairs that have already been processed
    limit = 200

    np.random.seed(999) #For testing, make sure we get the same transforms each time
    rndTransforms = [getRandomTransform(maxRot=5,maxScale=1.05,maxTrans=10) for _ in range(limit)]
    #Reverse the list as we will pop transforms from the far end. Want these to be the same, even 
    #if we change the limit later.
    rndTransforms.reverse()

    folder = local_sr_folder
    if server:
        folder = server_sr_folder
    if server:
        outfile = server_separate_mpm_folder+'PartIII_test1.csv'
    else:
        outfile = local_separate_mpm_folder+'PartIII_test1.csv'

        
    id_trans = transforms.CompositeTransform(2, [transforms.ScalingTransform(2, uniform=True), \
                                          transforms.Rigid2DTransform()])
#    id_trans.set_params([1.,0.2,10.,10.]) #Nelder mead doesn't work starting from zeros
    
#    grid_params = get_MI_gridmax(local_separate_mpm_folder+'PartI_test4.csv')
    
#    for slide, roi_idx, ref_path, flo_path in getNextSRPair(folder):
#    for slide, roi_idx, mpm_path, al_path in getNextPair():
    for slide, roi_idx, ref_im, flo_im in getNextMPMPair(verbose=True, server=False, norm=False, blur=0.0):
        # Take the next random transform
        rndTrans = rndTransforms.pop()

        if skip > 0:
            print("Skipping %s_%s"%(slide, roi_idx))
            skip -= 1
            continue
        limit -= 1
        if limit < 0:
            break
        # Open and prepare the images: if using SRs then don't normalize (or do?)
#        ref_im, ref_im_orig = preProcessImageAndCopy(ref_path, norm=False, blur=0.0)
#        flo_im, flo_im_orig = preProcessImageAndCopy(flo_path, norm=False, blur=0.0)
#        ref_im, ref_im_orig = preProcessImageAndCopy(al_path)
#        flo_im, flo_im_orig = preProcessImageAndCopy(mpm_path)
        #If aligning SHG + TPEF, keep a copy of the TPEF (floating) for later
        flo_im_orig = flo_im.copy()
        ref_im_orig = ref_im.copy()
        
        print("Random transform applied: %r"%rndTrans.get_params())
    
        # Apply the transform to the reference image, increasing the canvas size to avoid cutting off parts
        ref_im = centreAndApplyTransform(ref_im, rndTrans, np.rint(np.array(ref_im.shape)*1.5).astype('int'))
        
        # Show the images we are working with
        print("Aligning images for sample %s, region %s"%(slide, roi_idx) \
#              + ". A transform of %r has been applied to the reference image"%str(rndTrans.get_params())
#               + " from folder %s"%folder)
        )
        if False:
            plt.figure(figsize=(12,6))
            plt.subplot(121)
            plt.imshow(ref_im, cmap='gray', vmin=0, vmax=1)
            plt.title("Reference image")
            plt.subplot(122)
            plt.imshow(flo_im, cmap='gray', vmin=0, vmax=1)
            plt.title("Floating image")
            plt.show()
    
        # Choose a model, set basic parameters for that model
        reg = Register(2)
#        reg.set_model('alphaAMD', alpha_levels=7, symmetric_measure=True, squared_measure=False)
#        reg.set_model('MI', mutual_info_fun='norm')
#        reg.set_model('ssd')
        reg.set_model('dLDP')

        # Choose an optimzer, set basic parameters for it
#        reg.set_optimizer('adam', gradient_magnitude_threshold=1e-6)
#        reg.set_optimizer('sgd', gradient_magnitude_threshold=1e-6)
#        reg.set_optimizer('scipy', gradient_magnitude_threshold=1e-9, epsilon=0.02)
        bounds, steps = chooseGrid(id_trans)
        reg.set_optimizer('gridsearch', bounds=bounds, steps=steps)

        # Since we have warped the original reference image, create a mask so that only the relevant
        # pixels are considered. Use the same warping function as above
        ref_mask = np.ones(ref_im_orig.shape, 'bool')
        ref_mask = centreAndApplyTransform(ref_mask, rndTrans, np.rint(np.array(ref_im_orig.shape)*1.5).astype('int'))
        
        reg.set_image_data(ref_im, \
                           flo_im, \
                           ref_mask=ref_mask, \
#                           ref_mask=np.ones(ref_im.shape, 'bool'), \
                           flo_mask=np.ones(flo_im.shape, 'bool'), \
                           ref_weights=None, \
                           flo_weights=None
                           )


        
        #Grid search
        reg.add_initial_transform(id_trans)
        
        #All except alphaAMD
        # There is no evidence so far, that pyramid levels lead the search towards the MI maximum.
        reg.add_pyramid_levels(factors=[1,], sigmas=[0.0,])
        # Try with a blurred full-resolution image first (or only)
#        reg.add_pyramid_levels(factors=[1,1], sigmas=[5.0,0.0])


#        #AlphaAMD
#        reg.set_sampling_fraction(0.1)
#        reg.set_iterations(3000)
#
#        #Scipy and AlphaAMD
#        # Estimate an appropriate parameter scaling based on the sizes of the images (not used in grid search).
#        diag = transforms.image_diagonal(ref_im) + transforms.image_diagonal(flo_im)
#        diag = 2.0/diag
##        p_scaling = np.array([diag*100, diag*100, 5.0, 5.0])
#        p_scaling = np.array([diag, diag, 1.0, 1.0])
##        p_scaling = np.array([1.0, 100.0, 1.0/diag, 1.0/diag])
#        reg.add_initial_transform(id_trans, param_scaling=p_scaling)

#        # in addition to the ID transform, add a bunch of random starting points
#        add_multiple_startpts(reg, count=20, p_scaling=p_scaling)
        
#         #AlphaAMD
#        # Learning-rate / Step lengths [[start1, end1], [start2, end2] ...] (for each pyramid level)
#        step_lengths = np.array([[1., 1.], [1., 0.5], [0.5, 0.1]])
#        reg.set_step_lengths(step_lengths)
#        reg.add_pyramid_levels(factors=[4, 2, 1], sigmas=[5.0, 3.0, 0.0])

#        #MI starting from gridmax already found
#        if not (slide, roi_idx) in grid_params:
#            print(f'Grid results not found for slide {slide}, region {roi_idx}')
#            continue
#        starting_params = grid_params[(slide, roi_idx)]
#        s_trans = transforms.ScalingTransform(2, uniform=True)
#        s_trans.set_params(starting_params[0])
#        r_trans = transforms.Rigid2DTransform()
#        r_trans.set_params(starting_params[1:4])
#        starting_trans = transforms.CompositeTransform(2, [s_trans, r_trans])
#        reg.add_initial_transform(starting_trans, param_scaling=p_scaling)

        reg.set_report_freq(500)
    
        # Create output directory
        directory = os.path.dirname("./tmp/")
        if not os.path.exists(directory):
            os.makedirs(directory)
    
        # Start the pre-processing
        reg.initialize("./tmp/")
        
        # Start the registration
        reg.run(verbose=True)
    
        out_transforms, values = reg.get_outputs()
        transform = out_transforms[np.argmin(values)]
        value = np.min(values)
    
        ### Warp final image
        c = transforms.make_image_centered_transform(transform, ref_im, flo_im)
    
#       # Print out transformation parameters
#        print('Transformation parameters: %s.' % str(transform.get_params()))
    
        # Create the output image
        im_warped = np.zeros(ref_im.shape)
    
        # Transform the floating image into the reference image space by applying transformation 'c'
        c.warp(In = flo_im_orig, Out = im_warped, mode='nearest', bg_value = 0.0)
    
        # Show the images we ended up with
        if False:
            print("Aligned images for sample %s, region %s"%(slide, roi_idx))
            plt.figure(figsize=(12,6))
            plt.subplot(121)
            plt.imshow(ref_im, cmap='gray', vmin=0, vmax=1)
            plt.title("Reference image")
            plt.subplot(122)
            plt.imshow(im_warped, cmap='gray', vmin=0, vmax=1)
            plt.title("Floating image")
            plt.show()

        centred_gt_trans = transforms.make_image_centered_transform(rndTrans, \
                                                 ref_im, flo_im)
        
        gtVal = reg.get_value_at(rndTrans)
        err = get_transf_error(c, centred_gt_trans, flo_im.shape)
        print("Estimated transform:\t [", ','.join(['%.4f']*len(c.get_params()))%tuple(c.get_params())+"] with value %.4f"%(value))
        print("True transform:\t\t [", ','.join(['%.4f']*len(rndTrans.get_params()))%tuple(rndTrans.get_params())+"] with value %.4f"%(gtVal))
        print("Average corner error: %5f"%(err/4))
        print("Value difference: %.5f"%(gtVal-value))
#        print("Improvement over gridmax: %.5f"%(-value - float(starting_params[-1])))
        
#        resultLine = (slide, roi_idx, *c.get_params(), err, -value)
        
        successFlag = reg.get_flags()
        if len(successFlag) == 0:
            successFlag = 'N/A'
        else:
            successFlag = successFlag[-1][-1] #just take the flag for the last transform and last pyramid level
        resultLine = (slide, roi_idx, *rndTrans.get_params(), \
                      gtVal, \
                      *c.get_params(), \
                      value, \
                      err, successFlag, \
                      time.strftime('%Y-%m-%d %H:%M:%S'))
        results.append(resultLine)
        with open(outfile, 'a') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(resultLine)
    


def create_MI_surfaces(server=False):
    limit = 1
    for slide, roi_idx, mpm_path, al_path in getNextPair():
        limit -= 1
        if limit < 0:
            break
        # Open and prepare the images
        ref_im, ref_im_orig = preProcessImageAndCopy(al_path)
        flo_im, flo_im_orig = preProcessImageAndCopy(mpm_path)

        reg = Register(2)
        reg.set_model('MI', mutual_info_fun='norm')

        t = transforms.CompositeTransform(2, [transforms.ScalingTransform(2, uniform=True), \
                                              transforms.Rigid2DTransform()])

        bounds, steps = chooseGrid(t)
        reg.set_optimizer('gridsearch', bounds=bounds, steps=steps)
        
        # Since we have warped the original reference image, create a mask so that only the relevant
        # pixels are considered. Use the same warping function as above
        
        reg.set_image_data(ref_im, \
                           flo_im, \
                           ref_mask=np.ones(ref_im.shape, 'bool'), \
                           flo_mask=np.ones(flo_im.shape, 'bool'), \
                           ref_weights=None, \
                           flo_weights=None
                           )

        reg.add_pyramid_levels(factors=[1,], sigmas=[3.0,])

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
    print('Start time:', time.strftime('%Y-%m-%d %H:%M:%S'))
    server_paths = False
    if len(sys.argv) > 1:
        server_paths = True
    register_pairs(server_paths)
#    create_MI_surfaces(server_paths)
    end_time = time.time()
    print("Elapsed time: %.1f seconds"%(end_time-start_time))