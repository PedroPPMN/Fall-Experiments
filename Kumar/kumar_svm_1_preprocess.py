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
min_area = 3000
display_debug = False


def createmask(frame, algorithm):
    return algorithm.apply(frame)

def improveMask(mask, kernel_open, kernel_gradient):
    mask[mask < mask_thresh] = 0
    opening = cv.morphologyEx(mask, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, kernel_open))
    gradient = cv.morphologyEx(opening, cv.MORPH_GRADIENT, cv.getStructuringElement(cv.MORPH_ELLIPSE, kernel_gradient))
    return gradient

def findRectangles(mask, min_area):
    return [region.bbox for region in regionprops(label(mask)) if region.area >= min_area]

def findEllipses(rectangles):
    ellipses_list = []
    for (minl, mint, maxl, maxt) in rectangles:
        center = int((mint + maxt)/2), int((minl + maxl)/2)
        radius = int((maxt - mint)/2), int((maxl - minl)/2)
        ellipse_angle = math.degrees(np.arctan(radius[1] / radius[0]))

        ellipses_list.append((center, radius, ellipse_angle))
        
    return ellipses_list

def extract_features(rectangle, ellipsis):
    aspect_ratio = (rectangle[3] - rectangle[1]) / (rectangle[2] - rectangle[0])
    angle = ellipsis[2]
    #porque divide por quatro?
    area = math.pi * ellipsis[1][0] * ellipsis[1][1] / 4
    return aspect_ratio, angle, area

def is_falling(aspect_ratio, angle, area):
    if (aspect_ratio > 1):
        if (4000 <= area <= 12000):
            if (5 <= angle <= 40 or 70 <= angle <= 100):
                return 1

    return -1
    
def debug(frame, mask, rectangles, ellipses):
    mask = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)
    for rect in rectangles:
        mask = cv.rectangle(mask, (rect[1], rect[0]), (rect[3], rect[2]), (0, 255, 0), 3)

    for ellipsis in ellipses:
        mask = cv.ellipse(mask, ellipsis[0], ellipsis[1], 0, 0, 360, (0, 255, 255), 3)

    cv.imshow("Debug", np.hstack([frame, mask]))
    cv.waitKey(1)


videos = list(glob(dataset_root_folder + 'fall*/'))
videos = videos + list(glob(dataset_root_folder + 'adl*/'))
videos = [glob(video + '*.png') for video in videos]

features_list = []
filename = 'features.pickle'

for v, frames in enumerate(videos):
    background_removal = cv.createBackgroundSubtractorMOG2()
    for i, frame_path in enumerate(frames):
        frame = cv.imread(frame_path)

        mask = createmask(frame, background_removal)
        mask = improveMask(mask, kernel_open, kernel_gradient)
        rectangles = findRectangles(mask, min_area)
        ellipses = findEllipses(rectangles)

        fall = -1
        for rect, ellipsis in zip(rectangles, ellipses):
            aspect_ratio, angle, area = extract_features(rect, ellipsis)
            current_feature = (aspect_ratio, angle, area, is_falling(aspect_ratio, angle, area))
            features_list.append(current_feature)
        if display_debug:
            debug(frame, mask, rectangles, ellipses)



picklefile = open(filename, 'wb')
pickle.dump(features_list, picklefile)
picklefile.close()

print(features_list)
