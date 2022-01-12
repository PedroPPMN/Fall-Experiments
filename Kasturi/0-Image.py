from skimage.measure import label, regionprops
from glob import glob
import pandas as pd 
import numpy as np 
import cv2 as cv
import math
import os
import pickle

dataset_root_folder = './URFallDataset/'
kernel_open = (14,14)
kernel_gradient = (12,12)
mask_thresh = 130
display_debug = True

videos = list(glob(dataset_root_folder + '/Camera1/' + 'fall*/'))
#videos = videos + list(glob(dataset_root_folder + 'adl*/'))
videos = [glob(video + '*.png') for video in videos]

csvs = glob(dataset_root_folder + '*.csv')
csvs = [pd.read_csv(file, usecols=[0,1,2]) for file in csvs]
#csvs = np.concatenate([csvs[0], csvs[1]])
csvs = np.concatenate([csvs[1]])
csvs = {(row[0], row[1]): row[2] for row in csvs}

features_list = []

def createmask(frame, algorithm):
    return algorithm.apply(frame)

def improveMask(mask, kernel_open, kernel_gradient):
    mask[mask < mask_thresh] = 0
    opening = cv.morphologyEx(mask, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, kernel_open))
    gradient = cv.morphologyEx(opening, cv.MORPH_GRADIENT, cv.getStructuringElement(cv.MORPH_ELLIPSE, kernel_gradient))
    return gradient

def findRectangles(mask):
    regions = regionprops(mask)
    if(len(regions) > 0):
        i = np.argmax([region.area for region in regions])
        r = regions[i]  
        if display_debug:
            debug(frame, mask, r.bbox) 
        return r.bbox[0], r.bbox[1], r.bbox[2], r.bbox[3], r.centroid[0], r.centroid[1], r.area, r.orientation, r.extent
    else:
        return [0,0,0,0,0,0,0,0,0]

def debug(frame, mask, rectangle):
    mask = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)
    mask = cv.rectangle(mask, (rectangle[1], rectangle[0]), (rectangle[3], rectangle[2]), (0, 255, 0), 3)   
    print
    cv.imshow("Debug", np.hstack([frame, mask]))
    cv.waitKey(1)

features_list = []
filename = 'featureskasturi.pickle'

for v, frames in enumerate(videos):
    background_removal = cv.createBackgroundSubtractorMOG2()
    for i, frame_path in enumerate(frames):
        name = os.path.basename(frame_path[:-17])
        state = csvs.get((name, i + 1),None)
        if state != None:
            frame = cv.imread(frame_path)
            mask = createmask(frame, background_removal)
            mask = improveMask(mask, kernel_open, kernel_gradient)
            features = findRectangles(mask)
            features_list.append(features)

            if display_debug:
                print(name +' - ' +  str(i) + '- state: ' + str(state))
                print(features)

print(features_list)
print()

picklefile = open(filename, 'wb')
pickle.dump(features_list, picklefile)
picklefile.close()

