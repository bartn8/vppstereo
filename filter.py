import cv2
import numpy as np
from numba import njit
import time
import os

@njit
def left_warp(dmap):
    """
    Warp left disparity map to right view.
    Original values are preserved.
    Interpolation is not applied, only round.
    Uniqueness check: if a point collide then save max value.
    
    Parameters
    ----------
    dmap: HxW np.ndarray
        Disparity map in the left view    
    
    Returns
    -------
    omap: HxW np.ndarray
        Original disparity map warped to right view (occlusion map)
    conf_map: HxW np.ndarray
        Binary confidence map (1 for rejected points)
    filtered_i: int
        Number of points filtered by uniqueness check
    """
    h,w = dmap.shape[:2]
    omap = np.zeros(dmap.shape, dtype=dmap.dtype)

    #Verbose info
    warping_filtered_i = 0

    #Warp left dmap in occlusion dmap
    for y in range(h):
        for x in range(w):
            if dmap[y,x] > 0:
                d = int(round(dmap[y,x]))
                xd = x-d
                if 0 <= xd and xd <= w-1:
                    if omap[y,xd] < dmap[y,x]:
                        if omap[y,xd] != 0:
                            warping_filtered_i += 1

                        omap[y,xd] = dmap[y,x]
    
    return omap, warping_filtered_i

@njit
def left_unwarp(omap):
    """
    Unwarp the occlusion map to left view.
    Original values are preserved.
    Interpolation is not applied, only round.

    Parameters
    ----------
    omap: HxW np.ndarray
        Original disparity map warped to right view (occlusion map)
    
    Returns
    -------
    dmap_rst: HxW np.ndarray
        Disparity map in the left view
    """    
    h,w = omap.shape[:2]
    dmap_rst = np.zeros(omap.shape, dtype=omap.dtype)

    #Warp occlusion dmap in left dmap
    for y in range(h):
        for x in range(w):
            if omap[y,x] > 0:
                d = int(round(omap[y,x]))
                xd = x+d
                if 0 <= xd and xd <= w-1:
                    dmap_rst[y,xd] = omap[y,x]
    
    return dmap_rst

@njit
def conf_unwarp(conf, dmap):
    """
    Unwarp the confidence map to left view.
    Original values are preserved.
    Interpolation is not applied, only round.

    Parameters
    ----------
    conf: HxW np.ndarray
        Confidence map to unwarp.
    dmap: HxW np.ndarray
        Disparity map for warping operation.
    
    Returns
    -------
    conf_rst: HxW np.ndarray
        Unwarped confidence map
    """    
    h,w = dmap.shape[:2]
    conf_rst = np.ones(conf.shape, dtype=conf.dtype)

    #Warp occlusion dmap in left dmap
    for y in range(h):
        for x in range(w):
            if dmap[y,x] > 0:
                d = int(round(dmap[y,x]))
                xd = x+d
                if 0 <= xd and xd <= w-1:
                    conf_rst[y,xd] = conf[y,x]
    
    return conf_rst

@njit
def weighted_conf(dmap, rx=9, ry=7, l=2, g=0.4375, th=1.1):
    """
    Return a confidence map based on weighted distance.
    Points that are too close to foreground pixel are rejected (conf=1)
    
    Parameters
    ----------
    dmap: HxW np.ndarray
        Disparity map used to extract confidence map.
    rx: int
        Horizontal search radius (1,3,5,...)
    ry: int
        Vertical search radius (1,3,5,...)
    th: float
        Threshold for absolute difference

    Returns
    -------
    conf_rst: HxW np.ndarray
        Binary confidence map (1 for rejected points)
    """      

    h,w = dmap.shape[:2]
        
    #Confidence map between 0 and 1 (binary)
    conf_map = np.zeros(dmap.shape, dtype=np.uint8)

    rx = rx//2  
    ry = ry//2    

    for y in range(h):
        for x in range(w):
            if dmap[y,x] > 0:
                for xw in range(-rx,rx+1):               
                    for yw in range(-ry-1,ry+1):                     
                        if 0 <= y+yw and y+yw <= h-1 and 0 <= x+xw and x+xw <= w-1:
                            if dmap[y+yw, x+xw] > 0:
                                #Check that's a "background point"
                                #For slanted surfaces: check later with a threshold
                                if dmap[y+yw, x+xw] < dmap[y,x]:
                                    #Use Manhattan distance to keep in mind y-shifts
                                    #Reject a point if foreground disparity is greather than distance between fg and bg
                                    #if (dmap[y,x]-dmap[y+yw, x+xw]) - 2*((rx/(rx+ry))*abs(yw)+(ry/(rx+ry))*abs(xw)) > th:
                                    if (dmap[y,x]-dmap[y+yw, x+xw]) - l*(g*abs(xw)+(1-g)*abs(yw)) > th:
                                        conf_map[y+yw, x+xw] = 1                                            
                    
            else:
                conf_map[y,x] = 1
                    
    return conf_map


@njit
def filter(dmap,conf_map,th):
    """
    Drop points from a disparity map based on a confidence map.
    
    Parameters
    ----------
    dmap: HxW np.ndarray
        Disparity map to modify: there is side-effect.    
    conf_map: HxW np.ndarray
        Confidence map to use for filtering (1 if point is filtered).
    th: float
        Threshold for filtering

    Returns
    -------
    filtered_i: int
        Number of points filtered
    """
    h,w = dmap.shape[:2]
    filtered_i = 0
    for y in range(h):
        for x in range(w):
            if dmap[y,x] > 0:
                if conf_map[y,x] > th:
                    dmap[y,x] = 0
                    filtered_i += 1
    return filtered_i

@njit
def interpolate_disparity(dmap, n=3, th=1):
    """
    Try to fill gaps caused by warping operations using linear interpolation

    Parameters
    ----------
    dmap: HxW np.ndarray
        Disparity map to modify: there is side-effect.    
    n: int
        Horizontal window size (3,5,...).
    th: float
        Threshold for filtering.
    """
    h,w = dmap.shape[:2]
    n = n//2

    for y in range(h):
        for x in range(w):
            if dmap[y,x] == 0:
                #Find left-right neighbours
                n_left = 0
                n_leftx = 0
                n_right = 0
                n_rightx = 0
                
                for xw in range(-1,-n-1,-1):
                    if dmap[y,x+xw] > 0:
                        n_left = dmap[y,x+xw]
                        n_leftx = xw
                        break

                for xw in range(1, n+1):
                    if dmap[y,x+xw] > 0:
                        n_right = dmap[y,x+xw]
                        n_rightx = xw    
                        break
                
                #If two neighbours are found
                if n_left > 0 and n_right > 0:
                    #If left-right diff is under threshold
                    if abs(n_left-n_right) < th:
                        #Fill between zeros with linear interpolation
                        m = (n_right - n_left) / (n_rightx-n_leftx)
                        q = n_left - m * n_leftx
                        
                        for xw in range(n_leftx,n_rightx+1):
                            dmap[y,x+xw] = m * xw + q


def occlusion_heuristic(dmap, rx=9, ry=7, l=2, g=0.4375, th_conf=1, th_filter=0.1):

    """
    Occlusion filter based on a weighted window.

    Parameters
    ----------
    dmap: HxW np.ndarray
        Disparity map to modify: there is side-effect.  
    rx: int 
        x-axis radius of the window
    ry: int 
        y-axis radius of the window
    th_conf: float
        confidence threshold: used to classify a occluded point
    th_filter: float
        If confidence is binary (and weighted is) every value (0.0,1.0) is fine

    Return
    ------
    dmap: HxW np.ndarray
        Filtered disparity map
    conf_map: HxW np.ndarray
        Binary confidence map: 0 for no occlusion

    Usage
    -----
    ```python
    from filter import occlusion_heuristic
    gts = sample_gt(gt)
    gts_filtered = occlusion_heuristic(gts)[0]
    ```
    """

    omap, _ = left_warp(dmap)
    conf_map = weighted_conf(omap,rx=rx, ry=ry, l=l, g=g, th=th_conf)  

    #Filter omap
    _ = filter(omap, conf_map, th_filter)

    dmap = left_unwarp(omap)
    conf_map = conf_unwarp(conf_map, omap)
    
    #Interpolate dmap
    interpolate_disparity(dmap,3)
    
    return dmap, conf_map
