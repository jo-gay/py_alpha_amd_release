#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 16:30:03 2019

@author: jo

A script to create structural representations of images
"""

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

import sys, os, time
import pickle

import filters
from PIL import Image

import re #for matching filenames to a required pattern

sys.path.append('../PCANet')
from pcanet_based import PCANetBasedSR as net

local_bf_folder = '../data/aligned_190508/'
local_mpm_folder = '../data/processed/'
server_bf_folder = '/data/jo/MPM Skin Deep Learning Project/aligned_190508/'
server_mpm_folder = '/data/jo/MPM Skin Deep Learning Project/processed/'

local_output_folder = '../data/struct_reprs/pca/'
server_output_folder = '/data/jo/MPM Skin Deep Learning Project/struct_reprs/pca/'

def load_brightfield_images(verbose=False, server=False, blur=3.0, limit=None):
    images = []
    count=0
    pattern = re.compile("^final\.tif aligned to ([0-9]+)+\.tif$", re.IGNORECASE)
    if server:
        bf_folder = server_bf_folder
    else:
        bf_folder = local_bf_folder
    bf_subdirs = ['dysplastic/', 'malignant/', 'healthy/']
    for sd in bf_subdirs:
        (_, slidedirs, _) = next(os.walk(bf_folder+sd)) #get list of subdirs within this one

        for slide in slidedirs:
            (_, _, files) = next(os.walk(bf_folder+sd+slide)) #get list of files within this folder

            for f in files:
                if limit and count>=limit:
                    break
                else:
                    #Check each file in the subdir. If the file is an aligned file it will match the pattern above
                    m = pattern.match(f)
                    if m:
                        #If the pattern matches then read the aligned image file
                        bf_path = bf_folder + sd + slide + '/' + f
    #                    roi_idx = m.group(1)
                        
                        images.append(preProcessImage(bf_path, blur=blur))
                        count+=1
    if verbose:
        print('Loaded %d brightfield images'%len(images))

    return np.asarray(images)

def load_MPM_images(verbose=False, server=False, blur=3.0, limit=None):
    """Collate MPM images from a hard-coded directory structure into a single ndarray
    """
    images = []
    count=0
    pattern = re.compile("^([0-9]+)_([0-9]+)_gs.tif$", re.IGNORECASE)
    if server:
        mpm_folder = server_mpm_folder
    else:
        mpm_folder = local_mpm_folder

    (_, _, files) = next(os.walk(mpm_folder)) #get list of files
    for f in files:
        if limit and count>=limit:
            break
        else:
            m = pattern.match(f)
            if m:
                mpm_path = mpm_folder + f
                images.append(preProcessImage(mpm_path, blur=blur))
                count += 1
    if verbose:
        print('Loaded %d MPM images'%len(images))

    return np.asarray(images)

def getNextPair(verbose=False, server=False):
    """Generator function to get next pair of matching images out of a complicated directory structure.
    """
    al_pattern = re.compile("^final\.tif aligned to ([0-9]+)+\.tif$", re.IGNORECASE)
    if server:
        aligned_folder = server_bf_folder
        mpm_folder = server_mpm_folder
    else:
        aligned_folder = local_bf_folder
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
                    
def preProcessImage(path, blur=None, norm=True):
    im = Image.open(path).convert('L')
    im = np.asarray(im)/255. #images are uint8 - convert to float
    
    # Apply gaussian blur with sigma given by 'blur' parameter
    if blur:
        im = filters.gaussian_filter(im, blur)
    
    # Normalize
    if norm:
        im = filters.normalize(im, 0.0, None)
    
    return im


def create_SR_pairs(mpm_net, bf_net, server=False):

    # Create output directory
    if server:
        outpath = server_output_folder
    else:
        outpath = local_output_folder
            
    directory = os.path.dirname(outpath)
    if not os.path.exists(directory):
        os.makedirs(directory)

    for slide, roi_idx, mpm_path, bf_path in getNextPair(verbose=False, server=server):
        mpm = preProcessImage(mpm_path, blur=3.0, norm=True)
        mpmPSR = mpm_net.create_PSR(mpm)
        bf = preProcessImage(bf_path, blur=3.0, norm=True)
        bfPSR = bf_net.create_PSR(bf)
        # Show the images we ended up with
#        print("Structural representations for sample %s, region %s"%(slide, roi_idx))
        plt.figure(figsize=(12,12))
        plt.subplot(221)
        plt.imshow(bf, cmap='gray')
        plt.colorbar()
        plt.title("Brightfield image")
        plt.subplot(222)
        plt.imshow(mpm, cmap='gray')
        plt.colorbar()
        plt.title("MPM image")
        plt.subplot(223)
        plt.imshow(bfPSR[0], cmap='gray')
        plt.colorbar()
        plt.title("SR of brightfield image")
        plt.subplot(224)
        plt.imshow(mpmPSR[0], cmap='gray')
        plt.colorbar()
        plt.title("SR of MPM image")
        plt.show()
        
        #Save the PSRs to file
        mpm = np.rint(mpmPSR[0]*255.).astype('uint8')
        bf = np.rint(bfPSR[0]*255.).astype('uint8')
        Image.fromarray(mpm).save(outpath+'%s_%s_mpm_psr.tif'%(slide,roi_idx))
        Image.fromarray(bf).save(outpath+'%s_%s_bf_psr.tif'%(slide,roi_idx))


def train_PCANet(images, MPM=False, server=False):
    print('Training PCANet-based structural representation model')
    pcanet = net(
        image_shape=images[0].shape,
        filter_shape_l1=3, step_shape_l1=1, n_l1_output=8,
        filter_shape_l2=3, step_shape_l2=1, n_l2_output=8,
        filter_shape_pooling=2, step_shape_pooling=2
    )
    
    if MPM:
        pcanet.c1 = 0.02 #orig 0.8, this is better for MPM
        pcanet.c2 = 0.005 #orig 0.6, this is better for MPM

    pcanet.validate_structure()
    pcanet.fit(images)
    
    if False:
        print('Showing first five structural represenations:')
        for i in range(5):
            imagePSR = pcanet.create_PSR(images[i])
    
            plt.imshow(imagePSR[0], cmap='gray')
            plt.show()
        
    return pcanet

        
if __name__ == '__main__':
    start_time = time.time()
    use_server_paths = False
    if len(sys.argv) > 1:
        use_server_paths = True

    train=True    
    if train:
        mpm_net = train_PCANet(load_MPM_images(verbose=True, server=use_server_paths, blur=3.0, limit=25), MPM=True)
        bf_net = train_PCANet(load_brightfield_images(verbose=True, server=use_server_paths, blur=3.0, limit=25))
        filepath = local_output_folder
        if use_server_paths:
            filepath = server_output_folder
        with open(filepath+'mpm_net', "wb") as f:
            pickle.dump(mpm_net, f)
        with open(filepath+'bf_net', "wb") as f:
            pickle.dump(bf_net, f)
    
    create_SR_pairs(mpm_net, bf_net, server=use_server_paths)
    end_time = time.time()
    print("Elapsed time: %.1f s"%(end_time-start_time,))