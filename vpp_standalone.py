import numpy as np
import cv2
import math
from numba import njit

@njit
def _get_patch_size_based_on_distance(d_ref,d_min,d_max,patch_size,gamma=0.3):
    gamma_weight = ((d_ref-d_min)/(d_max-d_min)) ** (1/gamma) # [0,1]
    wsize = round((gamma_weight * (patch_size-1)) + 1)
    n = ((wsize -1) // 2)
    return n,n,n,n

@njit
def _virtual_projection_scan_max_dist(l, r, g, filled_g, uniform_color, wsize, wsize_agg_x, wsize_agg_y, direction, c, c_occ, g_occ, discard_occ, interpolate, use_distance_patch, use_bilateral_patch, dmin, dmax, distance_gamma):

    """
    Virtual projection using sparse disparity.

    Parameters
    ----------
    l: np.numpy [H,W,C] np.uint8
        Left original image
    r: np.numpy [H,W,C] np.uint8
        Right original image        
    g: np.numpy [H,W] np.float32
        Sparse disparity
    filled_g: np.numpy [H,W] np.float32
        Sparse disparity filled with bilateral model       
    wsize: int
        Max projection patch size (Default 5)     
    wsize_agg: int
        Window size for color computation (Default 5)                      
    direction: int mod 2
        Projection direction (1: left->right or 0: right->left) (Default 0)
    c: float
        alpha blending factor
    c_occ: float
        alpha blending factor in occluded areas
    g_occ: np.numpy [H,W] np.uint8
        Occlusion mask (If not present use np.zeros(l.shape, dtype=np.uint8))
    discard_occ: bool
        Use "NO" occlusion strategy -- ie, discard occluded points
    interpolate: bool
        Use weighted splatting of patterns in target view
    use_distance_patch: bool
        Use patch size based on distance
    use_bilateral_patch: bool
        Use adaptive patch based on bilateral filling
    dmin: float
        Minimum disparity value found in the frame
    dmax: float
        Maximum disparity value found in the frame
    distance_gamma: float
        Distance patch hyperparameter

    Returns
    -------
    sample_i:
        number of points projected
    """

    sample_i = 0

    height, width, channels = l.shape[:3]
    
    #Window size factor (wsize=2*n+1 -> n = (wsize-1)/2)
    n = ((wsize -1) // 2)
    #Window size factor (wsize=2*n+1 -> n = (wsize-1)/2)
    n_agg_x = ((wsize_agg_x -1) // 2)
    n_agg_y = ((wsize_agg_y -1) // 2)

    used_bins = np.zeros(256,dtype=np.uint8)
    n_bins = 256
    min_bin = 0
    min_bin_value = 1000000
    
    for y in range(height):
        x = width-1 if direction == 0 else 0
        #x = 0 if direction == 0 else width-1
        while (direction != 0 and x < width) or (direction == 0 and x>=0):
            if g[y,x] > 0:
                d = round(g[y,x])
                d0 = math.floor(g[y,x]) #5.43 -> 5
                d1 = math.ceil(g[y,x])  #5.43 -> 6
                d1_blending = g[y,x]-d0   #0.43 -> d_blending = 1-0.43 = 0.57

                #Warping right (negative disparity hardcoded)
                xd = x-d
                xd0 = x-d0
                xd1 = x-d1

                #Get adaptive patch based on distance
                if use_distance_patch:
                    min_n_y, max_n_y, min_n_x, max_n_x = _get_patch_size_based_on_distance(g[y,x], dmin, dmax, wsize, distance_gamma)
                else:
                    min_n_y, max_n_y, min_n_x, max_n_x =  n,n,n,n
                
                for j in range(channels):

                    if uniform_color:
                        pa = 0
                        pb = 255
                        n_bins = 256
                        for k in range(256):
                            used_bins[k] = 0

                        for yw_agg in range(-n_agg_y,n_agg_y+1):
                            for xw_agg in range(-n_agg_x,n_agg_x+1):
                                if 0 <= y+yw_agg and y+yw_agg <= height-1 and 0 <= x+xw_agg and x+xw_agg <= width-1:
                                    #Left Histogram analysis   
                                    #No occluded point or left side occlusion
                                    if g_occ[y,x] == 0 or not (0 <= xd+xw_agg and xd+xw_agg <= width-1):       
                                        if l[y+yw_agg, x+xw_agg, j] > pa and l[y+yw_agg, x+xw_agg, j] < pb:#if l inside [pa,pb]
                                            if l[y+yw_agg, x+xw_agg, j] - pa > pb - l[y+yw_agg, x+xw_agg, j]:#if d(pa,l) > d(pb,l)
                                                pb = l[y+yw_agg, x+xw_agg, j]#Discard the min distance
                                            elif l[y+yw_agg, x+xw_agg, j] - pa < pb - l[y+yw_agg, x+xw_agg, j]:#if d(pa,l) < d(pb,l)
                                                pa = l[y+yw_agg, x+xw_agg, j]#Discard the min distance

                                        if l[y+yw_agg, x+xw_agg, j] == 0:
                                            n_bins -=1
                                        used_bins[l[y+yw_agg, x+xw_agg, j]] += 1
                                    
                                    #Right Histogram analysis
                                    #Search right only if point is not left side occluded
                                    if 0 <= xd+xw_agg and xd+xw_agg <= width-1:
                                        if r[y+yw_agg, xd+xw_agg, j] > pa and r[y+yw_agg, xd+xw_agg, j] < pb:#if r inside [pa,pb]
                                            if r[y+yw_agg, xd+xw_agg, j] - pa > pb - r[y+yw_agg, xd+xw_agg, j]:#if d(pa,r) > d(pb,r)
                                                pb = r[y+yw_agg, xd+xw_agg, j]#Discard the min distance
                                            elif r[y+yw_agg, xd+xw_agg, j] - pa < pb - r[y+yw_agg, xd+xw_agg, j]:#if d(pa,r) < d(pb,r)
                                                pa = r[y+yw_agg, xd+xw_agg, j]#Discard the min distance
                                        
                                        if r[y+yw_agg, xd+xw_agg, j] == 0:
                                            n_bins -=1
                                        used_bins[r[y+yw_agg, xd+xw_agg, j]] += 1
                        
                        if n_bins == 0:
                            min_bin_value = used_bins[0]
                            min_bin = 0
                            for k in range(256):
                                if min_bin_value > used_bins[k]:
                                    min_bin=k
                                    min_bin_value=used_bins[k]
                            pa=min_bin
                            pb=min_bin

                    #Project patch in left and right images (1)
                    #Also in left side occlusion maintain uniformity (2)

                    for yw in range(-min_n_y,max_n_y+1):
                        for xw in range(-min_n_x,max_n_x+1):
                            if 0 <= y+yw and y+yw <= height-1 and 0 <= x+xw and x+xw <= width-1:     

                                #0) skip projection if bilateral filtering is on and current disparity does not match filled disparity value
                                if not use_bilateral_patch or abs(g[y,x] - filled_g[y+yw,x+xw]) < 0.1:

                                    #1)Pattern color part
                                    #Search for the best color to blend in the image
                                    if not uniform_color:
                                        pa = 0
                                        pb = 255
                                        n_bins = 256
                                        for k in range(256):
                                            used_bins[k] = 0

                                        for yw_agg in range(-n_agg_y,n_agg_y+1):
                                            for xw_agg in range(-n_agg_x,n_agg_x+1):
                                                if 0 <= y+yw+yw_agg and y+yw+yw_agg <= height-1 and 0 <= x+xw+xw_agg and x+xw+xw_agg <= width-1:
                                                    #Left Histogram analysis   
                                                    #No occluded point or left side occlusion
                                                    if g_occ[y,x] == 0 or not (0 <= xd+xw+xw_agg and xd+xw+xw_agg <= width-1):       
                                                        if l[y+yw+yw_agg, x+xw+xw_agg, j] > pa and l[y+yw+yw_agg, x+xw+xw_agg, j] < pb:#if l inside [pa,pb]
                                                            if l[y+yw+yw_agg, x+xw+xw_agg, j] - pa > pb - l[y+yw+yw_agg, x+xw+xw_agg, j]:#if d(pa,l) > d(pb,l)
                                                                pb = l[y+yw+yw_agg, x+xw+xw_agg, j]#Discard the min distance
                                                            elif l[y+yw+yw_agg, x+xw+xw_agg, j] - pa < pb - l[y+yw+yw_agg, x+xw+xw_agg, j]:#if d(pa,l) < d(pb,l)
                                                                pa = l[y+yw+yw_agg, x+xw+xw_agg, j]#Discard the min distance
                                                        
                                                        if l[y+yw+yw_agg, x+xw+xw_agg, j] == 0:
                                                            n_bins -=1
                                                        used_bins[l[y+yw+yw_agg, x+xw+xw_agg, j]] += 1

                                                    #Right Histogram analysis
                                                    #Search right only if point is not left side occluded
                                                    if 0 <= xd+xw+xw_agg and xd+xw+xw_agg <= width-1:
                                                        if r[y+yw+yw_agg, xd+xw+xw_agg, j] > pa and r[y+yw+yw_agg, xd+xw+xw_agg, j] < pb:#if r inside [pa,pb]
                                                            if r[y+yw+yw_agg, xd+xw+xw_agg, j] - pa > pb - r[y+yw+yw_agg, xd+xw+xw_agg, j]:#if d(pa,r) > d(pb,r)
                                                                pb = r[y+yw+yw_agg, xd+xw+xw_agg, j]#Discard the min distance
                                                            elif r[y+yw+yw_agg, xd+xw+xw_agg, j] - pa < pb - r[y+yw+yw_agg, xd+xw+xw_agg, j]:#if d(pa,r) < d(pb,r)
                                                                pa = r[y+yw+yw_agg, xd+xw+xw_agg, j]#Discard the min distance
                                                        
                                                        if r[y+yw+yw_agg, xd+xw+xw_agg, j] == 0:
                                                            n_bins -=1
                                                        used_bins[r[y+yw+yw_agg, xd+xw+xw_agg, j]] += 1
                                        
                                        if n_bins == 0:
                                            min_bin_value = used_bins[0]
                                            min_bin = 0
                                            for k in range(256):
                                                if min_bin_value > used_bins[k]:
                                                    min_bin=k
                                                    min_bin_value=used_bins[k]
                                            pa=min_bin
                                            pb=min_bin

                                            

                                    if  0 <= xd0+xw and xd0+xw <= width-1:#  (1)
                                        #Occlusion check
                                        if g_occ[y,x] == 0:#Not occluded point  
                                            l[y+yw,x+xw,j] = (((pa+pb)/2) * c + l[y+yw,x+xw,j] * (1-c))
                                            if interpolate:
                                                r[y+yw,xd0+xw,j] = (((((pa+pb)/2) * c + r[y+yw,xd0+xw,j] * (1-c)) * (1-d1_blending)) + r[y+yw,xd0+xw,j] * d1_blending)
                                                if 0 <= xd1+xw and xd1+xw <= width-1:# Linear interpolation only if inside the border
                                                    r[y+yw,xd1+xw,j] = (((((pa+pb)/2) * c + r[y+yw,xd1+xw,j] * (1-c)) * d1_blending) + r[y+yw,xd1+xw,j] * (1-d1_blending))
                                            else:
                                                r[y+yw,xd+xw,j] = ((pa+pb)/2) * c + r[y+yw,xd+xw,j] * (1-c)
                                        elif not discard_occ:# Occluded point: Foreground point should be projected before occluded point
                                            if interpolate:
                                                r[y+yw,xd0+xw,j] = (((((pa+pb)/2) * c_occ + r[y+yw,xd0+xw,j] * (1-c_occ)) * (1-d1_blending)) + r[y+yw,xd0+xw,j] * d1_blending)
                                                if 0 <= xd1+xw and xd1+xw <= width-1:
                                                    r[y+yw,xd1+xw,j] = (((((pa+pb)/2) * c_occ + r[y+yw,xd1+xw,j] * (1-c_occ)) * d1_blending) + r[y+yw,xd1+xw,j] * (1-d1_blending))
                                                l[y+yw,x+xw,j] = ((r[y+yw,xd0+xw,j]*(1-d1_blending)+r[y+yw,xd1+xw,j]*d1_blending) * c + l[y+yw,x+xw,j] * (1-c))             
                                            else:
                                                r[y+yw,xd+xw,j] = ((pa+pb)/2) * c_occ + r[y+yw,xd+xw,j] * (1-c_occ)
                                                l[y+yw,x+xw,j] = r[y+yw,xd+xw,j] * c + l[y+yw,x+xw,j] * (1-c)
                                    else:#Left side occlusion (known) (2)
                                        l[y+yw,x+xw,j] = (((pa+pb)/2) * c + l[y+yw,x+xw,j] * (1-c))        
                                                    
                sample_i +=1

            x = x-1 if direction == 0 else x+1
    
    return sample_i

@njit
def random_int_256():
    return np.random.randint(0, 256)

@njit
def random_int_max(n):
    return np.random.randint(0, n)

@njit
def _virtual_projection_scan_rnd(l, r, g, filled_g, uniform_color, wsize, direction, c, c_occ, g_occ, discard_occ, interpolate, use_distance_patch, use_bilateral_patch, dmin, dmax, distance_gamma):

    """
    Virtual projection using sparse disparity.

    Parameters
    ----------
    l: np.numpy [H,W,C] np.uint8
        Left original image
    r: np.numpy [H,W,C] np.uint8
        Right original image        
    g: np.numpy [H,W] np.float32
        Sparse disparity
    filled_g: np.numpy [H,W] np.float32
        Sparse disparity filled with bilateral model         
    wsize: int
        Max projection patch size (Default 5)                
    direction: int mod 2
        Projection direction (left->right or right->left) (Default 0)
    c: float
        alpha blending factor
    c_occ: float
        alpha blending factor in occluded areas
    g_occ: np.numpy [H,W] np.uint8
        Occlusion mask (If not present use np.zeros(l.shape, dtype=np.uint8))
    discard_occ: bool
        Use "NO" occlusion strategy -- ie, discard occluded points
    interpolate: bool
        Use weighted splatting of patterns in target view
    use_distance_patch: bool
        Use patch size based on distance
    use_bilateral_patch: bool
        Use adaptive patch based on bilateral filling
    dmin: float
        Minimum disparity value found in the frame
    dmax: float
        Maximum disparity value found in the frame
    distance_gamma: float
        Distance patch hyperparameter
        
    Returns
    -------
    sample_i:
        number of points projected
    """

    sample_i = 0

    height, width, channels = l.shape[:3]
    
    #Window size factor (wsize=2*n+1 -> n = (wsize-1)/2)
    n = ((wsize -1) // 2)
    #Window size factor (wsize=2*n+1 -> n = (wsize-1)/2)
    # n_agg_x = ((wsize_agg_x -1) // 2)
    # n_agg_y = ((wsize_agg_y -1) // 2)
    
    for y in range(height):
        x = width-1 if direction == 0 else 0
        #x = 0 if direction == 0 else width-1
        while (direction != 0 and x < width) or (direction == 0 and x>=0):
            if g[y,x] > 0:
                d = round(g[y,x])
                d0 = math.floor(g[y,x]) #5.43 -> 5
                d1 = math.ceil(g[y,x])  #5.43 -> 6
                d1_blending = g[y,x]-d0   #0.43 -> d_blending = 1-0.43 = 0.57

                #Warping right (negative disparity hardcoded)
                xd = x-d
                xd0 = x-d0
                xd1 = x-d1

                #Get adaptive patch based on distance
                if use_distance_patch:
                    min_n_y, max_n_y, min_n_x, max_n_x = _get_patch_size_based_on_distance(g[y,x], dmin, dmax, wsize, distance_gamma)
                else:
                    min_n_y, max_n_y, min_n_x, max_n_x =  n,n,n,n

                for j in range(channels):

                    #1)Pattern color part
                    #Search for the best color to blend in the image
                    if uniform_color:
                        rvalue = random_int_256()

                    #Project patch in left and right images (1)
                    #Also in left side occlusion maintain uniformity (2)

                    for yw in range(-min_n_y,max_n_y+1):
                        for xw in range(-min_n_x,max_n_x+1):
                            if 0 <= y+yw and y+yw <= height-1 and 0 <= x+xw and x+xw <= width-1:     

                                #0) skip projection if bilateral filtering is on and current disparity does not match filled disparity value
                                if not use_bilateral_patch or abs(g[y,x] - filled_g[y+yw,x+xw]) < 0.1:

                                    #1)Pattern color part
                                    #Search for the best color to blend in the image
                                    if not uniform_color:
                                        rvalue = random_int_256()

                                    if  0 <= xd0+xw and xd0+xw <= width-1:#  (1)
                                        #Occlusion check
                                        if g_occ[y,x] == 0:#Not occluded point  
                                            l[y+yw,x+xw,j] = (rvalue * c + l[y+yw,x+xw,j] * (1-c))
                                            if interpolate:
                                                r[y+yw,xd0+xw,j] = (((rvalue * c + r[y+yw,xd0+xw,j] * (1-c)) * (1-d1_blending)) + r[y+yw,xd0+xw,j] * d1_blending)
                                                if 0 <= xd1+xw and xd1+xw <= width-1:# Linear interpolation only if inside the border
                                                    r[y+yw,xd1+xw,j] = (((rvalue * c + r[y+yw,xd1+xw,j] * (1-c)) * d1_blending) + r[y+yw,xd1+xw,j] * (1-d1_blending))
                                            else:
                                                r[y+yw,xd+xw,j] = rvalue * c + r[y+yw,xd+xw,j] * (1-c)
                                        elif not discard_occ:# Occluded point: Foreground point should be projected before occluded point
                                            if interpolate:
                                                r[y+yw,xd0+xw,j] = (((rvalue * c_occ + r[y+yw,xd0+xw,j] * (1-c_occ)) * (1-d1_blending)) + r[y+yw,xd0+xw,j] * d1_blending)
                                                if 0 <= xd1+xw and xd1+xw <= width-1:
                                                    r[y+yw,xd1+xw,j] = (((rvalue * c_occ + r[y+yw,xd1+xw,j] * (1-c_occ)) * d1_blending) + r[y+yw,xd1+xw,j] * (1-d1_blending))
                                                l[y+yw,x+xw,j] = ((r[y+yw,xd0+xw,j]*(1-d1_blending)+r[y+yw,xd1+xw,j]*d1_blending) * c + l[y+yw,x+xw,j] * (1-c))             
                                            else:
                                                r[y+yw,xd+xw,j] = rvalue * c_occ + r[y+yw,xd+xw,j] * (1-c_occ)
                                                l[y+yw,x+xw,j] = r[y+yw,xd+xw,j] * c + l[y+yw,x+xw,j] * (1-c)
                                                
                                    else:#Left side occlusion (known) (2)
                                        l[y+yw,x+xw,j] = (rvalue * c + l[y+yw,x+xw,j] * (1-c))        
                                                    
                sample_i +=1

            x = x-1 if direction == 0 else x+1
    
    return sample_i

@njit
def _bilateral_filling(dmap, img, n, o_xy = 2, o_i= 1, th=.001):
    h,w = img.shape[:2]
    assert dmap.shape == img.shape
    cmap = np.zeros_like(dmap)
    aug_dmap = dmap.copy()

    for y in range(h):
        for x in range(w):
            i_ref = img[y,x]
            d_ref = dmap[y,x]
            if d_ref > 0:
                for yw in range(-n,n+1):
                    for xw in range(-n,n+1):
                        if 0 <= y+yw and y+yw <= h-1 and 0 <= x+xw and x+xw <= w-1:
                            weight = math.exp(-(((yw)**2+(xw)**2)/(2*(o_xy**2)) + ((img[y+yw,x+xw]-i_ref)**2)/(2*(o_i**2))))

                            if cmap[y+yw,x+xw] < weight:
                                cmap[y+yw,x+xw] = weight
                                aug_dmap[y+yw,x+xw] = d_ref

    aug_dmap = np.where(cmap>th,aug_dmap,0)

    return aug_dmap

def vpp(left, right, gt, wsize = 3, wsizeAgg_x = 64, wsizeAgg_y = 3, left2right = True, blending = 0.4, use_distance_patch = False, use_bilateral_patch = False, distance_gamma = 0.3, bilateral_o_xy = 2, bilateral_o_i= 1, bilateral_th = .001, uniform_color = False, method="rnd", c_occ = 0.00, g_occ = None, discard_occ = False, interpolate = True):
    lc,rc = np.copy(left), np.copy(right)
    gt = gt.astype(np.float32)

    assert method in ["rnd", "maxDistance"]
    direction = 1 if left2right else 0

    if len(lc.shape) < 3:
        lc,rc = np.expand_dims(lc, axis=-1), np.expand_dims(rc, axis=-1)

    #No projection if no points are provided
    if np.count_nonzero(gt) == 0:
        return lc,rc
        
    dmin = gt[gt>0].min()
    dmax = gt[gt>0].max()

    #Convert rgb to gray if needed
    if len(lc.shape) == 3 and lc.shape[2] == 3:
        gray_context = cv2.cvtColor(lc, cv2.COLOR_BGR2GRAY)
    else:
        gray_context = np.squeeze(lc)

    if use_bilateral_patch:
        filled_gt = _bilateral_filling(gt, gray_context, (wsize-1)//2, bilateral_o_xy, bilateral_o_i, th=bilateral_th)
    else:
        filled_gt = gt.copy()        

    if g_occ is None:
        g_occ = np.zeros_like(gt)

    if method == "maxDistance":
        _virtual_projection_scan_max_dist(lc,rc,gt,filled_gt,uniform_color,wsize,wsizeAgg_x,wsizeAgg_y,direction, blending, c_occ, g_occ, discard_occ,interpolate,use_distance_patch,use_bilateral_patch,dmin,dmax,distance_gamma)
    elif method == "rnd":
        _virtual_projection_scan_rnd(lc,rc,gt,filled_gt,uniform_color,wsize,direction, blending, c_occ, g_occ, discard_occ, interpolate,use_distance_patch,use_bilateral_patch,dmin,dmax,distance_gamma)

    return lc,rc
