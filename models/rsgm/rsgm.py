import cv2
import numpy as np
from numba import njit


from pyrSGM import census5x5_SSE, costMeasureCensus5x5_xyd_SSE, aggregate_SSE, matchWTA_SSE, matchWTARight_SSE, subPixelRefine, median3x3_SSE

def _census_transform(left,right):
    #Grayscale conversion HxWxC -> HxW
    if len(left.shape) == 3 and left.shape[2] == 3:
        left = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
        right = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)
    elif len(left.shape) == 3 and left.shape[2] == 1:
        left = np.squeeze(left)
        right = np.squeeze(right)

    h,w = left.shape[:2]

    if w % 16 != 0:
        raise Exception(f"Invalid width ({w}): width % 16 != 0")

    ctl = np.zeros(left.shape, dtype=np.uint32)
    ctr = np.zeros(right.shape, dtype=np.uint32)

    census5x5_SSE(left,ctl,int(w), int(h))
    census5x5_SSE(right,ctr,int(w), int(h))

    return ctl, ctr

def _hamming_matching(ctl,ctr,dmax=192):
    if dmax % 8 != 0:
        raise Exception(f"Invalid dmax ({dmax}): dmax % 8 != 0")
    
    if dmax > 256:
        raise Exception(f"Invalid dmax ({dmax}): dmax > 256")

    h,w = ctl.shape[:2]

    if w % 16 != 0:
        raise Exception(f"Invalid width ({w}): width % 16 != 0")

    dsi = np.zeros((h, w, dmax), dtype=np.uint16)

    costMeasureCensus5x5_xyd_SSE(ctl, ctr, dsi, w, h, dmax, 1)#1 -> Single thread

    return dsi

def _aggregate_dsi(left, dsi, p1 = 11, p2min = 17, alpha = 0.5, gamma = 35):
    h,w,dmax = dsi.shape

    if w % 16 != 0:
        raise Exception(f"Invalid width ({w}): width % 16 != 0")
    
    if dmax % 8 != 0:
        raise Exception(f"Invalid dmax ({dmax}): dmax % 8 != 0")
    
    if dmax > 256:
        raise Exception(f"Invalid dmax ({dmax}): dmax > 256")

    dsiAgg = np.zeros((h,w,dmax), dtype=np.uint16)
    aggregate_SSE(left, dsi, dsiAgg, w, h, dmax, p1, p2min, alpha, gamma)

    return dsiAgg


@njit
def _linear_interpolate(dmap, n=3, th=1):
    """
    Try to fill gaps using linear interpolation

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
            if dmap[y,x] <= 0:
                #Find left-right neighbours
                n_left = 0
                n_leftx = 0
                n_right = 0
                n_rightx = 0
                
                for xw in range(-1,-n-1,-1):
                    if 0 <= x+xw < w and dmap[y,x+xw] > 0:
                        n_left = dmap[y,x+xw]
                        n_leftx = xw
                        break

                for xw in range(1, n+1):
                    if 0 <= x+xw < w and dmap[y,x+xw] > 0:
                        n_right = dmap[y,x+xw]
                        n_rightx = xw    
                        break
                
                #If two neighbours are found
                if n_left > 0 and n_right > 0:
                    #If left-right diff is under threshold
                    if abs(n_left-n_right) < th:
                        #Fill invalid points with linear interpolation
                        m = (n_right - n_left) / (n_rightx-n_leftx)
                        q = n_left - m * n_leftx
                        
                        for xw in range(n_leftx,n_rightx+1):
                            dmap[y,x+xw] = m * xw + q

@njit
def _guided_dsi(dsi, hints, validhints, k = 10, c = 1):
    h,w,dmax = dsi.shape
    dsi_dtype = dsi.dtype
    dsi = dsi.astype(np.float64)

    for y in range(h):
        for x in range(w):
            if validhints[y,x] > 0:
                tmprange = k * (1-np.exp((-(hints[y,x]-np.arange(dmax))**2)/(2*c**2)))
                dsi[y,x,:] *= tmprange

    return dsi.astype(dsi_dtype)

def _disparity_computation(dsi, uniqueness=0.95):
    h,w,dmax = dsi.shape
    if w % 16 != 0:
        raise Exception(f"Invalid width ({w}): width % 16 != 0")
    
    if dmax % 8 != 0:
        raise Exception(f"Invalid dmax ({dmax}): dmax % 8 != 0")
    
    if dmax > 256:
        raise Exception(f"Invalid dmax ({dmax}): dmax > 256")
    
    dispImg = np.zeros((h,w), dtype=np.float32)
    matchWTA_SSE(dsi, dispImg, w,h,dmax,float(uniqueness))
    subPixelRefine(dsi, dispImg, w,h,dmax,0)#0-> equiangular method

    dispImgfiltered = np.zeros((h,w), dtype=np.float32)
    median3x3_SSE(dispImg, dispImgfiltered, w, h)

    #Invalid pixels interpolation
    _linear_interpolate(dispImgfiltered,15,3)

    #Clipping
    dispImgfiltered = np.clip(dispImgfiltered, 0, None)

    return dispImgfiltered

def _right_disparity_computation(dsi, uniqueness=0.95):
    h,w,dmax = dsi.shape
    if w % 16 != 0:
        raise Exception(f"Invalid width ({w}): width % 16 != 0")
    
    if dmax % 8 != 0:
        raise Exception(f"Invalid dmax ({dmax}): dmax % 8 != 0")
    
    if dmax > 256:
        raise Exception(f"Invalid dmax ({dmax}): dmax > 256")
    
    if uniqueness > 1.0 or uniqueness <= 0.0:
         raise Exception(f"Invalid uniqueness ({uniqueness}): uniqueness in ]0,1]")
    
    dispImgRight = np.zeros((h,w), dtype=np.float32)
    matchWTARight_SSE(dsi, dispImgRight, w,h,dmax,float(uniqueness))

    dispImgRightfiltered = np.zeros((h,w), dtype=np.float32)
    median3x3_SSE(dispImgRight, dispImgRightfiltered, w, h)

    #Invalid pixels interpolation
    _linear_interpolate(dispImgRightfiltered,15,3)

    #Clipping
    dispImgRightfiltered = np.clip(dispImgRightfiltered, 0, None)

    return dispImgRightfiltered

#https://github.com/simonmeister/motion-rcnn/blob/master/devkit/cpp/io_disp.h
@njit
def _interpolate_background(dmap):
    h,w = dmap.shape[:2]

    for v in range(h):
        count = 0
        for u in range(w):
            if dmap[v,u] > 0:
                if count >= 1:#at least one pixel requires interpolation
                    u1,u2 = u-count,u-1#first and last value for interpolation
                    if u1>0 and u2<w-1:#set pixel to min disparity
                        d_ipol = min(dmap[v,u1-1], dmap[v,u2+1])
                        for u_curr in range(u1,u2+1):
                            dmap[v,u_curr] = d_ipol
                count = 0
            else:
                count +=1
        
        #Border interpolation(left,right): first valid dmap value is used as filler
        for u in range(w):
            if dmap[v,u] > 0:
                for u2 in range(u):
                    dmap[v,u2] = dmap[v,u]
                break

        for u in range(w-1,-1,-1):
            if dmap[v,u] > 0:
                for u2 in range(u+1,w):
                    dmap[v,u2] = dmap[v,u]
                break
        
    #Border interpolation(top,bottom): first valid dmap value is used as filler
    for u in range(w):
        for v in range(h):
            if dmap[v,u] > 0:
                for v2 in range(v):
                    dmap[v2,u] = dmap[v,u]
                break
        
        for v in range(h-1,-1,-1):
            if dmap[v,u] > 0:
                for v2 in range(v+1,h):
                    dmap[v2,u] = dmap[v,u]
                break

@njit
def _left_right_check(dmapl,dmapr, th = 1):
    mask_occ = np.zeros(dmapl.shape, np.uint8)
    h,w = dmapl.shape[:2]

    for y in range(h):
        for x in range(w):
            if dmapl[y,x] > 0:
                d = int(round(dmapl[y,x]))
                xd = x-d
                if 0 <= xd and xd <= w-1:
                    if dmapr[y,xd] > 0:
                        if abs(dmapl[y,x]-dmapr[y,xd]) > th:
                            mask_occ[y,x] = 128
                        else:
                            mask_occ[y,x] = 255
                else:
                    mask_occ[y,x] = 128
    
    return mask_occ

def compute_rsgm(left, left_vpp, right_vpp, hints=None, validhints=None, dmax=192, p1 = 11, p2min = 17, alpha = 0.5, gamma = 35, uniqueness=0.95):
    ht,wt = left.shape[:2]

    #padding here all images then remove it after disparity computation
    pad_ht = (((ht // 16) + 1) * 16 - ht) % 16
    pad_wd = (((wt // 16) + 1) * 16 - wt) % 16
    _pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]

    left = cv2.copyMakeBorder(left, _pad[2], _pad[3], _pad[0], _pad[1], cv2.BORDER_REFLECT)
    left_vpp = cv2.copyMakeBorder(left_vpp, _pad[2], _pad[3], _pad[0], _pad[1], cv2.BORDER_REFLECT)
    right_vpp = cv2.copyMakeBorder(right_vpp, _pad[2], _pad[3], _pad[0], _pad[1], cv2.BORDER_REFLECT)

    ctl,ctr = _census_transform(left_vpp,right_vpp)
    dsi = _hamming_matching(ctl,ctr,dmax)

    if hints is not None and validhints is not None:
        hints = cv2.copyMakeBorder(hints, _pad[2], _pad[3], _pad[0], _pad[1], cv2.BORDER_CONSTANT, value=0)
        validhints = cv2.copyMakeBorder(validhints, _pad[2], _pad[3], _pad[0], _pad[1], cv2.BORDER_CONSTANT, value=0)
        dsi = _guided_dsi(dsi, hints, validhints)

    dsi = _aggregate_dsi(left,dsi,p1,p2min,alpha,gamma)

    fdmap = _disparity_computation(dsi,uniqueness)
    fdmap_right = _right_disparity_computation(dsi,uniqueness)

    hd, wd = fdmap.shape[-2:]
    c = [_pad[2], hd-_pad[3], _pad[0], wd-_pad[1]]
    fdmap = fdmap[c[0]:c[1], c[2]:c[3]]
    fdmap_right = fdmap_right[c[0]:c[1], c[2]:c[3]]
    
    maskocc = _left_right_check(fdmap, fdmap_right, 1)
    fdmap[maskocc == 128] = 0
    fdmap = fdmap.astype(np.uint8)
    cv2.filterSpeckles(fdmap,0,200,10)
    fdmap = fdmap.astype(np.float32)
    
    _interpolate_background(fdmap)

    return fdmap

