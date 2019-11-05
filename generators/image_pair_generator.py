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
# Find pairs of images from given directory structure
#

import re #for matching filenames to a required pattern
import os #for traversing filesystem

def default_AB_converter(a):
    return a.replace('A', 'B')
    
def _nextFile(root='./', subfolders=[], deepestOnly=True):
    """Generator to get the path to the next file recursively from nested directories
    """

    if len(subfolders) == 0 or not deepestOnly:
        # We are at the lowest directory required or we want files from every level. 
        # Yield each filename from this directory
        _, _, files = next(os.walk(root))
        for f in files:
            yield root+f

    sfs = subfolders.copy()
    if len(subfolders) > 0:
        if subfolders[0] == ['*']:
            _, subdirs, _ = next(os.walk(root)) #get list of all subdirs within this one
            
            sfs[0] = subdirs
        for sub in sfs[0]:
            if sub[-1] != '/':
                sub = sub+'/'
            if len(sfs)>1:
                for f in _nextFile(root=root+sub, subfolders=subfolders[1:], deepestOnly=deepestOnly):
                    yield f
            elif deepestOnly: #if not true, we have already processed the files in the current dir
                for f in _nextFile(root=root+sub):
                    yield f
    return


def getImagePair(folder_root='./', \
                subfoldersA=[], \
                deepestOnly=True, \
                patternA="^([0-9]+)_([0-9]+)+\.tif$", \
                ABconverter=default_AB_converter, \
                verbose=False):
    """Generator function to find next pair of matching images within a specified 
    directory structure.

    For each file that matches a given pattern (type A), look for the corresponding
    (type B) file. It is assumed that, given the path to the A file, the B path
    can be calculated, by a supplied function. If both files are found, yield their paths.
    
    Arguments:
        folder_root: (string) full or relative path where to find the data, 
                     terminated with /
        subfoldersA: (list of lists) Each element is a list of subdirectories to 
                     search in, relative to the previous element, e.g. 
                     [[sub1/, sub2/], ['*']] searches every subdirectory
                     of sub1 and sub2. Not case sensitive.
        patternA:    (string suitable for regex compilation) An expression by 
                     which to recognise files of type A.
        deepestOnly: (bool) Only look at files in the deepest level of the given 
                     directory structure
        ABconverter: (function) Given a (string) path to a type A file, return
                     the path to the corresponding type B file
        verbose: (bool) Print a message for any A file that has no matching B file
    Yields:
        pathA: (string) path to image A
        pathB: (string) path to image B
    """

    patternA = re.compile(patternA, re.IGNORECASE)
    for fileA in _nextFile(folder_root, subfoldersA, deepestOnly):
        #Check each file in the subdir to see if it matches the pattern given
        matchA = patternA.match(fileA)
        if matchA:
            #If so, find the corresponding paired image
            fileB = ABconverter(fileA)
            if fileB:
                if os.path.isfile(fileB):
                    yield fileA, fileB
                    continue
            
            if verbose:
                print(f"Unable to find matching image for {fileA}")
    return


if __name__ == '__main__':
    
    ### Test case 1 ###
    print('Finding files of type A: Brightfield; paired with type B: Grayscale MPM')
    count = 0
    patt = '.*/(.+)/final.tif aligned to (.+).tif'
    def converter(bfPath):
        bfPattern = re.compile(patt)
        m = bfPattern.match(bfPath)
        if not m:
            return
        slide = m.group(1)
        region = m.group(2)
        return f'../../data/reprocessed/{slide}_{region}_gs.tif'
        
    pairGenerator = getImagePair(folder_root='../../data/aligned_190508/', \
                        subfoldersA=[['dysplastic', 'malignant', 'healthy'],['*']], \
                        patternA=patt, \
                        ABconverter=converter, \
                        verbose=True)
    while count < 10:
        try:
            files = next(pairGenerator)
            print(files)
        except StopIteration:
            print(f'A total of {count} type A files were found')
            break
        count += 1
        
    ### Test case 2 ###
    print('Finding files of type A: SHG; paired with type B: TPEF1')
    count = 0
    patt = '.*/(.+)-shg.tif'
    
    pairGenerator = getImagePair(folder_root='../../data/unprocessed/', \
                        subfoldersA=[['Dysplastic', 'Malignant', 'Healthy'],['*'],['med']], \
                        patternA=patt, \
                        ABconverter=lambda x:x.replace('shg', 'tpef1'), \
                        verbose=True)
    while count < 10:
        try:
            files = next(pairGenerator)
            print(files)
        except StopIteration:
            print(f'A total of {count} type A files were found')
            break
        count += 1
        

