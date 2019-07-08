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

def getNextSRPair(folder, server=False, norm=True, blur=3.0, verbose=False, order=None):
    """Generator function to get next pair of images from structural representations folder.
    
    Look through all the files in the given folder. For each one where the filename matches 
    the pattern, look for one that has the same name except with bf. If found, return the 
    paths and also the sample and region ids
    
    If order is set to True, determine what order the MPM files are found in, and return
    SRs in that order, to facilitate comparison of results.
    """
    
    if order:
        for slide, region, _, _ in getNextMPMPair(verbose=verbose, server=server, load=False):
            shg_path, tpef_path = getSRPair(folder, slide, region, verbose)
            shg_im, _ = OpenAndPreProcessImage(shg_path, norm=norm, blur=blur, copyOrig=False)
            tpef_im, _ = OpenAndPreProcessImage(tpef_path, norm=norm, blur=blur, copyOrig=False)
            yield slide, region, shg_im, tpef_im
    else:
#        pattern = re.compile("^([0-9]+)_([0-9]+)_mpm_.*$") #for combined MPM and BF files
        pattern = re.compile("^psr_shg_([0-9a-zA-Z]+)_([0-9]+).tif$") #for v1 SHG and TPEF files with blur=3 and cs diff for shg vs tpef
#        pattern = re.compile("^([0-9a-zA-Z]+)_([0-9]+)_shg_psr.png$") #for v2 SHG and TPEF files with blur=3 and c1=c2=0.1 for both
#        pattern = re.compile("^([0-9a-zA-Z]+)_([0-9]+)_shg_psr_blur5.png$") #for individual SHG and TPEF files with blur=5 and c1=c2=0.1 for both
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
                shg_im, _ = OpenAndPreProcessImage(folder+f, norm=norm, blur=blur, copyOrig=False)
                tpef_im, _ = OpenAndPreProcessImage(tpef_path, norm=norm, blur=blur, copyOrig=False)
                yield m.group(1), m.group(2), shg_im, tpef_im
            else:
                if verbose:
                    print('SHG file %s: matching TPEF %s not found'%(f, tpef_path))
    return

def getSRPair(folder, slide, region, verbose=False):
    """Get specified pair of images from specified structural representations folder.
    """
    shg_path = folder+f"psr_shg_{slide}_{region}.tif" #old blur=3.0, c=0.1/0.2 files (v1)
#    shg_path = folder+f"{slide}_{region}_shg_psr.png" #new blur=3.0, c=0.1 files (v2)
#    shg_path = folder+f"{slide}_{region}_shg_psr_blur5.png" #new blur=5.0, c=0.1 files (v3)
    tpef_path = shg_path.replace('shg', 'tpef')
    if os.path.isfile(shg_path) and os.path.isfile(tpef_path):
        return shg_path, tpef_path
    
    message = ""
    if not os.path.isfile(shg_path):
        message += f"File {shg_path} not found. "
    if not os.path.isfile(tpef_path):
        message += f"File {tpef_path} not found. "
    return None, message
                
def getNextMPMPair(verbose=False, server=False, norm=False, blur=0.0, load=True, rotate=False, quantize=None):
    """Generator function to get next pair of matching images out of a complicated directory structure.
    
    The SHG and two TPEF images are stored in the same folder. The two TPEF images need to be averaged.
    
    NOTE: unlike the above functions which return the path to the images, this one reads and returns
    the images themselves, unless load=False in which case it just returns the next slide and region index.
    """
    shg_pattern = re.compile("^([0-9]+)-shg.tif$", re.IGNORECASE)
    if server:
        folder = server_separate_mpm_folder
        subdirs = ['Dysplastic Tissue/', 'Malignant Tissue/', 'Healthy Tissue/']
    else:
        folder = local_separate_mpm_folder
        subdirs = ['Dysplastic/', 'Malignant/', 'Healthy/']

    for sd in subdirs:
        (_, slidedirs, _) = next(os.walk(folder+sd)) #get list of subdirs within this one
        for slide in slidedirs:
            full_path = folder + sd + slide + '/med/'
            if verbose:
                print(f'Processing files in directory {full_path}')

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
                        if not load:
                            yield slide, roi_idx, None, None
                            continue
                    else:
                        if verbose:
                            print("Unable to find TPEF images", tpef1_path, tpef2_path)
                        continue

                    #If we found a set of three then read them and average the TPEFs
                    shg, _ = OpenAndPreProcessImage(full_path+f, norm=norm, blur=blur, quantize=quantize)
                    tpef1, _ = OpenAndPreProcessImage(tpef1_path, norm=norm, blur=blur, quantize=quantize)
                    tpef2, _ = OpenAndPreProcessImage(tpef2_path, norm=norm, blur=blur, quantize=quantize)
                    
                    if rotate: #rotate the images by 90 degress so they have the same orientation as the brightfield images
                        shg = np.asarray(Image.fromarray(shg).rotate(-90))
                        tpef1 = np.asarray(Image.fromarray(tpef1).rotate(-90))
                        tpef2 = np.asarray(Image.fromarray(tpef2).rotate(-90))
                        
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



def gridBounds(centreTrans, scale_range, rot_range, transl_range):
    """Calculate range for each parameter, centered around the transform given.
    
    Args:
        centreTrans is a transform object of the type required (only affine and composite(scale, rigid2d)
        are handled so far). The grid will be centred around the parameters in this transform.
        scale_range - the max distance around the centre for scale parameter(s), as a proportion of the image
        size (e.g. if centre is scale 1 then 0.1 gives scale 0.9 - 1.1)
        rot_range - the max rotation in degrees
        transl_range - the max translation in pixels
        
        For affine transforms the scale and rotation parameters are applied to the diag and non-diag elements
        of the 2x2 matrix respectively.
        
    Returns:
        bounds gives the min and max for each parameter
    """
    centre = centreTrans.get_params()
    #Affine parameter bounds:
    if len(centre) == 6:
        bounds = np.array([[-scale_range, scale_range], \
                           [-rot_range*np.pi/180, rot_range*np.pi/180], \
                           [-rot_range*np.pi/180, rot_range*np.pi/180], \
                           [-scale_range, scale_range], \
                           [-transl_range, transl_range], \
                           [-transl_range, transl_range]])
    #Composite (scale + rigid2d) parameter bounds:
    elif len(centre) == 4:
        bounds = np.array([[-scale_range, scale_range], \
                           [-rot_range*np.pi/180, rot_range*np.pi/180], \
                           [-transl_range, transl_range], \
                           [-transl_range, transl_range]])
    
    if False: #Enable for MI surfaces
        bounds = np.array([[-scale_range, scale_range], [0, 0], [-20, 20], [-20, 20]])

    bounds = [bounds[i] + centre[i] for i in range(len(centre))]

    return bounds

def OpenAndPreProcessImage(path, norm=True, blur=None, copyOrig=False, quantize=None):
    im = Image.open(path).convert('L')
    im = np.asarray(im)#[125:375,125:375] #Take a smaller region for speed
    
    # Also return an unprocessed copy of original image, if required
    im_orig = None
    if copyOrig:
        im_orig = im.copy()
    
    if blur and blur > 0:
        im = filters.gaussian_filter(im, blur)
    # Normalize
    if norm:
        im = filters.normalize(im, 0.0, None)
    else:
        im = im/255. #convert to floats without normalizing
        
    #Quantize into a number of intensity bins
    if quantize:
        im = np.rint(im * (quantize-1))/(quantize-1)
    
    return im, im_orig

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
        reg.add_initial_transform(getRandomTransform(maxRot=5, maxScale=1.05, maxTrans=10), \
                                  param_scaling=p_scaling)
    
    
def register_pairs(server=False):
    #TODO. Docstrings.
    results=[]
    id_trans = transforms.CompositeTransform(2, [transforms.ScalingTransform(2, uniform=True), \
                                          transforms.Rigid2DTransform()])
    
    ##### Running parameters to update each time #####
    modelname = 'mi' #['alphaAMD', 'MI', 'MSE', 'dLDP']
#    modelparams = {} #mse
#    modelparams = {'alpha_levels':7, 'symmetric_measure':True, 'squared_measure':False}
    modelparams = {'mutual_info_fun':'norm'}

    optname = 'gridsearch' #['gd', 'adam', 'gridsearch', 'bfgs']
#    optparams = {'gradient_magnitude_threshold':1e-9, 'epsilon':0.1} #bfgs
    optparams = {'bounds':gridBounds(id_trans, 0.05, 5, 10), 'steps':11} #gridsearch
#    optparams = {'gradient_magnitude_threshold':1e-6} #adam, gd
    
    norm = True
    blur = 3.0
    skip = 3 #25 #manual way to skip pairs that have already been processed
    results_file = 'PartII_test1.3.csv'
    limit = 25-skip
    ##### End running parameters #####

    np.random.seed(999) #For testing, make sure we get the same transforms each time
    rndTransforms = [getRandomTransform(maxRot=5,maxScale=1.05,maxTrans=10) for _ in range(limit+skip+1)]
    #Reverse the list as we will pop transforms from the far end. Want these to be the same, even 
    #if we change the limit later.
    rndTransforms.reverse()

    folder = local_sr_folder
    if server:
        folder = server_sr_folder
    if server:
        outfile = server_separate_mpm_folder+results_file
    else:
        outfile = local_separate_mpm_folder+results_file

        
#    id_trans.set_params([1.,0.2,10.,10.]) #Nelder mead doesn't work starting from zeros

    #OPTION: Starting from gridmax already found
#    grid_params = get_MI_gridmax(local_separate_mpm_folder+'PartI_test4.csv')
    
    for slide, roi_idx, ref_im, flo_im in getNextSRPair(folder, order=True, verbose=True, server=server, norm=norm, blur=blur):
#    for slide, roi_idx, mpm_path, al_path in getNextPair():
#    for slide, roi_idx, ref_im, flo_im in getNextMPMPair(verbose=True, server=server, norm=norm, blur=blur):
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
#        ref_im, ref_im_orig = OpenAndPreProcessImage(al_path, copyOrig=True)
#        flo_im, flo_im_orig = OpenAndPreProcessImage(mpm_path, copyOrig=True)
        #If aligning SHG + TPEF, keep a copy of the SHG (reference) image 
        #as it was before random transform is applied
#        flo_im_orig = flo_im.copy()
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
        reg.set_model(modelname, **modelparams)

        # Choose an optimzer, set basic parameters for it
        reg.set_optimizer(optname, **optparams)

        # Since we have warped the original reference image, create a mask so that only the relevant
        # pixels are considered. Use the same warping function as above
        ref_mask = np.ones(ref_im_orig.shape, 'bool')
        ref_mask = centreAndApplyTransform(ref_mask, rndTrans, np.rint(np.array(ref_im_orig.shape)*1.5).astype('int'))
        
        reg.set_image_data(ref_im, \
                           flo_im, \
                           ref_mask=ref_mask, \
                           flo_mask=np.ones(flo_im.shape, 'bool'), \
                           ref_weights=None, \
                           flo_weights=None
                           )


        ## Add pyramid levels
        if modelname == 'alphaamd':
            # Learning-rate / Step lengths [[start1, end1], [start2, end2] ...] (for each pyramid level)
            step_lengths = np.array([[1., 1.], [1., 0.5], [0.5, 0.1]])
            reg.set_step_lengths(step_lengths)
            reg.add_pyramid_levels(factors=[4, 2, 1], sigmas=[5.0, 3.0, 0.0])
            reg.set_sampling_fraction(0.5) #very patchy with 0.1, also tried 0.25
            reg.set_iterations(3000)

        else:
            # I have seen no evidence so far, that pyramid levels lead the search towards the MI maximum.
            reg.add_pyramid_levels(factors=[1,], sigmas=[0.0,])
            # Try with a blurred full-resolution image first (or only)
#            reg.add_pyramid_levels(factors=[1,1], sigmas=[5.0,0.0])

        ## Add initial transform(s), with parameter scaling if required
        if modelname == 'gridsearch':
            reg.add_initial_transform(id_trans)
        else:
            #BFGS and AlphaAMD
            # Estimate an appropriate parameter scaling based on the sizes of the images (not used in grid search).
            diag = transforms.image_diagonal(ref_im) + transforms.image_diagonal(flo_im)
            diag = 2.0/diag
            p_scaling = np.array([diag*100, diag*100, 5.0, 5.0])
    #        p_scaling = np.array([diag*20., diag*20., 1.0, 1.0])
            reg.add_initial_transform(id_trans, param_scaling=p_scaling)
    
#            #OPTION: in addition to the ID transform, add a bunch of random starting points
#            add_multiple_startpts(reg, count=20, p_scaling=p_scaling)

#            #OPTION: Starting from gridmax already found
#            if not (slide, roi_idx) in grid_params:
#                print(f'Grid results not found for slide {slide}, region {roi_idx}')
#                continue
#            starting_params = grid_params[(slide, roi_idx)]
#            s_trans = transforms.ScalingTransform(2, uniform=True)
#            s_trans.set_params(starting_params[0])
#            r_trans = transforms.Rigid2DTransform()
#            r_trans.set_params(starting_params[1:4])
#            starting_trans = transforms.CompositeTransform(2, [s_trans, r_trans])
#            reg.add_initial_transform(starting_trans, param_scaling=p_scaling)


        reg.set_report_freq(500)
    
        # Create output directory
        directory = os.path.dirname("./tmp/")
        if not os.path.exists(directory):
            os.makedirs(directory)
    
        # Start the pre-processing
        reg.initialize("./tmp/", norm=norm)
        
        # Start the registration
        reg.run(verbose=True)

        # Get the results and find the best one (for the case when there
        # was more than one starting point)    
        out_transforms, values = reg.get_outputs()
        transform = out_transforms[np.argmin(values)]
        value = np.min(values)
        successFlag = reg.get_flags()
        if len(successFlag) == 0:
            successFlag = 'N/A'
        else:
            #Use the optimizer flag for the best output transform found. SuccessFlag
            #has one result for each pyramid level, just take the last level.
            successFlag = successFlag[np.argmin(values)][-1]
    
        ### Warp final image
        c = transforms.make_image_centered_transform(transform, ref_im, flo_im)
    
#       # Print out transformation parameters
#        print('Transformation parameters: %s.' % str(transform.get_params()))
    
        # Create the output image
        im_warped = np.zeros(ref_im.shape)
    
        # Transform the floating image into the reference image space by applying transformation 'c'
        c.warp(In = flo_im, Out = im_warped, mode='nearest', bg_value = 0.0)
    
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
    

def create_dLDP_illustration():
    id_trans = transforms.CompositeTransform(2, [transforms.ScalingTransform(2, uniform=True), \
                                          transforms.Rigid2DTransform()])

    skip = 24
    limit = 1
    norm = True
    blur = 3.0
    modelname = 'dLDP' #Choose from ['alphaAMD', 'MI', 'MSE', 'dLDP']
    modelparams = {'interpolation':'nearest'}
    optname = 'gridsearch' #Choose from ['gd', 'adam', 'gridsearch', 'bfgs']
    optparams = {'bounds':gridBounds(id_trans, 0, 0, 0), 'steps':[1,1,1,1]}

    for slide, roi_idx, ref_im, flo_im in getNextMPMPair(verbose=True, \
                                                         server=False, \
                                                         norm=norm, \
                                                         blur=blur, \
                                                         rotate=True, \
                                                         quantize=32):
        if skip > 0:
            skip -= 1
            continue
        limit -= 1
        if limit < 0:
            break

        reg = Register(2)
        reg.set_model(modelname, **modelparams)
        reg.set_optimizer(optname, **optparams)
        reg.set_image_data(ref_im, \
                           flo_im, \
                           ref_mask=np.ones(ref_im.shape, 'bool'), \
                           flo_mask=np.ones(flo_im.shape, 'bool'), \
                           ref_weights=None, \
                           flo_weights=None
                           )
        reg.add_pyramid_levels(factors=[1,], sigmas=[0.0,])
        reg.add_initial_transform(id_trans)
        # Start the pre-processing
        reg.initialize("./tmp/")
        
        # The initialization sets up the distance measure, now we can use it
        dist = reg.distances[-1]
        
        shg_dLDP, shg_mask = dist.create_LDP(ref_im)
        tpef_dLDP, tpef_mask = dist.create_LDP(flo_im)
        
        #show what the dLDPs look like for ref image
        fig = plt.figure(figsize=(15,10))
        plt.subplot(2,3,1)
        plt.imshow(ref_im, cmap='gray', vmin=0, vmax=1)
        plt.title('a)', loc='left')
        plt.axis('off')
        #show (d)LDP images for two pairs of directions
        i=0 # i=0 #0 degrees for LDP; i=1 #0, 90 for dLDP
        shg_dLDP_im1 = dist.dLDP_as_image(shg_dLDP[...,(8*i):(8*(i+1))]) #Turn the next 8 bits into an image
        i=2 # i=2 #90 degrees for LDP; i=4 #45, 135 for dLDP
        shg_dLDP_im2 = dist.dLDP_as_image(shg_dLDP[...,(8*i):(8*(i+1))]) #Turn the next 8 bits into an image
        plt.subplot(2,3,2)
        plt.imshow(shg_dLDP_im1, cmap='gray', vmin=0, vmax=1)
        plt.title('b)', loc='left')
        plt.axis('off')
        plt.subplot(2,3,3)
        plt.imshow(shg_dLDP_im2, cmap='gray', vmin=0, vmax=1)
        plt.title('c)', loc='left')
        plt.axis('off')

        #same for floating image
        plt.subplot(2,3,4)
        plt.imshow(flo_im, cmap='gray', vmin=0, vmax=1)
        plt.title('d)', loc='left')
        plt.axis('off')
        #show dLDP images for two pairs of directions
        i=0 # i=0 #0 degrees for LDP; i=1 #0, 90 for dLDP
        tpef_dLDP_im1 = dist.dLDP_as_image(tpef_dLDP[...,(8*i):(8*(i+1))]) #Turn the next 8 bits into an image
        i=2 # i=2 #90 degrees for LDP; i=4 #45, 135 for dLDP
        tpef_dLDP_im2 = dist.dLDP_as_image(tpef_dLDP[...,(8*i):(8*(i+1))]) #Turn the next 8 bits into an image
        plt.subplot(2,3,5)
        plt.imshow(tpef_dLDP_im1, cmap='gray', vmin=0, vmax=1)
        plt.title('e)', loc='left')
        plt.axis('off')
        plt.subplot(2,3,6)
        plt.imshow(tpef_dLDP_im2, cmap='gray', vmin=0, vmax=1)
        plt.title('f)', loc='left')
        plt.axis('off')



        fig.subplots_adjust(hspace=0.1, wspace=0.05)
        plt.show()
    
    
def create_surface(server=False):
    
    init_t = transforms.CompositeTransform(2, [transforms.ScalingTransform(2, uniform=True), \
                                          transforms.Rigid2DTransform()])
#    rigT = transforms.Rigid2DTransform()
#    rigT.set_params([0.35,0.5,0.5])
#    rigT.set_params([0.,0.,0.])
#    init_t = transforms.CompositeTransform(2, [transforms.ScalingTransform(2, uniform=True), rigT])
    
    ##### Running parameters to update each time #####
    modelname = 'alphaAMD' #Choose from ['alphaAMD', 'MI', 'MSE', 'dLDP']
#    modelparams = {}
    modelparams = {'alpha_levels':7, 'symmetric_measure':True, 'squared_measure':False}
#    modelparams = {'mutual_info_fun':'norm'}

    optname = 'gridsearch' #Choose from ['gd', 'adam', 'gridsearch', 'bfgs']
#    optparams = {'gradient_magnitude_threshold':1e-9, 'epsilon':0.02}
    optparams = {'bounds':gridBounds(init_t, 0.05, 5, 0), 'steps':[41,41,1,1]}
#    optparams = {'gradient_magnitude_threshold':1e-6}
    
    norm = True
    blur = 0.0
    skip = 24
    limit = 3
    folder = local_sr_folder


    ##### End running parameters #####
    
    for slide, roi_idx, ref_im, flo_im in getNextSRPair(folder, order=True, verbose=True, server=server, norm=norm, blur=blur):
#    for slide, roi_idx, ref_im, flo_im in getNextMPMPair(verbose=True, server=server, norm=norm, blur=blur):
#    slide = 'cilia'
#    roi_idx = 'none'
#    ref_im, _ = OpenAndPreProcessImage('./test_images/reference_example.png', norm=norm, blur=blur, copyOrig=False)
#    flo_im, _ = OpenAndPreProcessImage('./test_images/floating_example.png', norm=norm, blur=blur, copyOrig=False)
#    if True: #to save re-indenting after temporarily removing above loop
        if skip > 0:
            print("Skipping %s_%s"%(slide, roi_idx))
            skip -= 1
            continue
    
        limit -= 1
        if limit < 0:
            break

        print("Creating %s surface for sample %s, region %s"%(modelname, slide, roi_idx))
        reg = Register(2)
        reg.set_model(modelname, **modelparams)

        # Choose an optimzer, apply basic parameters specified above
        reg.set_optimizer(optname, **optparams)


        reg.set_image_data(ref_im, \
                           flo_im, \
                           ref_mask=np.ones(ref_im.shape, 'bool'), \
                           flo_mask=np.ones(flo_im.shape, 'bool'), \
                           ref_weights=None, \
                           flo_weights=None
                           )

        reg.add_pyramid_levels(factors=[1,], sigmas=[0.0,])
        reg.set_sampling_fraction(0.25)
#        #Adam, GD
#        step_lengths = np.array([[0.5, 0.1]])
#        reg.set_step_lengths(step_lengths)
#
        reg.add_initial_transform(init_t, param_scaling=[1/500., 1/500., 1., 1.])
        reg.set_report_freq(500)
    
        # Create output directory
        directory = os.path.dirname("./tmp/")
        if not os.path.exists(directory):
            os.makedirs(directory)
    
        # Start the pre-processing
        reg.initialize("./tmp/")
        
        # Start the registration
        reg.run()
        
        values = np.array(reg.get_value_history(0, 0))

        if True: #Show 2d surface
            axes = [0,1]
            values.resize([optparams['steps'][a] for a in axes])
            gridpts = [np.linspace(*optparams['bounds'][i], optparams['steps'][i]) \
                       for i in range(len(optparams['bounds']))]
            
            #Convert radians to degrees for display
            gridpts[1] *= 180/np.pi

            #Determine what aspect is needed for a square image
            aspect = (gridpts[axes[1]][-1]-gridpts[axes[1]][0])/(gridpts[axes[0]][-1]-gridpts[axes[0]][0])
            
            min_loc = np.unravel_index(np.argmin(values), values.shape)
            min_loc = [gridpts[i][min_loc[idx]] for idx, i in enumerate(axes)]
            plt.figure(figsize=(10,10))
            ax = plt.gca()
            im = ax.imshow(values, extent=[gridpts[axes[1]][0], gridpts[axes[1]][-1], \
                                       gridpts[axes[0]][0], gridpts[axes[0]][-1]], \
                       origin='lower', aspect=aspect, cmap='inferno_r')
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.title(f"{modelname} surface as scale and rotation change\n" \
                      +f"(translation fixed at zero, image{'' if blur else ' not'} smoothed)")
#            plt.title(f"{modelname} surface as translation changes\n" \
#                      +f"(scale 1.0, rotation 0, image{'' if blur else ' not'} smoothed)")
            min_loc = list(reversed(min_loc))
            plt.annotate('Min=%.4f at (%.1f, %.2f)'%(np.min(values),*min_loc), xy=min_loc, xycoords='data',
                         xytext=(0.6, 0.04), textcoords='figure fraction',
                         arrowprops=dict(arrowstyle="->"))

            axisnames=['Scale (%)', 'Rotation (degrees)', 'x translation (px)', 'y translation (px)']
            plt.xlabel(axisnames[axes[1]])
            plt.ylabel(axisnames[axes[0]])
            plt.show()

        
if __name__ == '__main__':
    start_time = time.time()
    print('Start time:', time.strftime('%Y-%m-%d %H:%M:%S'))
    server_paths = False
    if len(sys.argv) > 1:
        server_paths = True
#    register_pairs(server_paths)
    create_surface(server_paths)
#    create_dLDP_illustration()
    end_time = time.time()
    print("Elapsed time: %.1f seconds"%(end_time-start_time))