# cython: language_level=3, boundscheck=False

from cpython cimport array
import array

import numpy as np
cimport numpy as np

from libc.stdint cimport (uint8_t, uint16_t, uint32_t, uint64_t,
                          int8_t, int16_t, int32_t, int64_t, UINT32_MAX)

from libc.stdlib cimport srand, rand

from libc.math cimport round, sqrt, log, M_PI, cos, ceil, floor

from posix.time cimport clock_gettime, timespec, CLOCK_REALTIME

np.import_array()

"""
Return a seed for random initialization 
"""
def get_seed():
    cdef timespec ts
    cdef double current
    clock_gettime(CLOCK_REALTIME, &ts)
    current = ts.tv_sec + (ts.tv_nsec / 1000000000.)
    return current

"""
Initialize random module
"""
def init_rand(_seed=0):
    cdef uint64_t seed = <uint64_t> _seed
    srand(<int>_seed)

cdef inline int clipping(int x, int min, int max):
    if x < min:
        return min
    elif x > max:
        return max
    else:
        return x

cdef inline clipping_float(float x, float min, float max):
    if x < min:
        return min
    elif x > max:
        return max
    else:
        return x

cpdef int virtual_projection_scan_rnd(uint8_t[:,:,:] l, uint8_t[:,:,:] r, float[:,:] g, 
 int width, int height, int channels, bint uniform_color, int wsize, int direction, float c, float c_occ, uint8_t[:,:] g_occ, bint discard_occluded, bint interpolate):

    cdef int sample_i = 0
    cdef int x,y,j,xd1,d1,d0,xd0,xd,d
    cdef float d1_blending
    #Window size factor (wsize=2*n+1 -> n = (wsize-1)/2)
    cdef int n = <int> ((wsize -1) // 2)
    # cdef int n_agg_x = <int> ((wsize_agg_x -1) // 2)
    # cdef int n_agg_y = <int> ((wsize_agg_y -1) // 2)

    #Window coords
    #cdef int xw_agg,yw_agg
    cdef int xw,yw
    #Random values in x,y and xd,y
    cdef uint8_t rvalue = 0
    #adaptive patch size
    cdef int min_n_y,max_n_y,min_n_x,max_n_x

    min_n_y = n
    max_n_y = n
    min_n_x = n
    max_n_x = n

    for y in range(height):
        x = width-1 if direction == 0 else 0
        #x = 0 if direction == 0 else width-1
        while (direction != 0 and x < width) or (direction == 0 and x>=0):
            if g[y,x] > 0:
                d = <int>(round(g[y,x]))
                d0 = <int>(floor(g[y,x])) #5.43 -> 5
                d1 = <int>(ceil(g[y,x]))  #5.43 -> 6
                d1_blending = g[y,x]-d0   #0.43 -> d_blending = 1-0.43 = 0.57

                xd = x-d
                xd0 = x-d0
                xd1 = x-d1
                for j in range(channels):

                    if uniform_color:
                        rvalue = rand() % 256
                    
                    #Use a certain window
                    #if adapt_win_size > 1:
                    for yw in range(-min_n_y,max_n_y+1):
                        for xw in range(-min_n_x,max_n_x+1):
                            if 0 <= y+yw and y+yw <= height-1 and 0 <= x+xw and x+xw <= width-1:

                                if not uniform_color:
                                    rvalue = rand() % 256

                                if 0 <= xd0+xw and xd0+xw <= width-1:
                                    #Apply to right patch only if point is not occluded
                                    if g_occ[y,x] == 0:#Not Occluded point
                                        l[y+yw,x+xw,j] = <uint8_t>(rvalue * c + l[y+yw,x+xw,j] * (1-c))
                                        if interpolate:
                                            r[y+yw,xd0+xw,j] = <uint8_t>(((rvalue * c + r[y+yw,xd0+xw,j] * (1-c)) * (1-d1_blending)) + r[y+yw,xd0+xw,j] * d1_blending)
                                            if 0 <= xd1+xw and xd1+xw <= width-1:
                                                r[y+yw,xd1+xw,j] = <uint8_t>(((rvalue * c + r[y+yw,xd1+xw,j] * (1-c)) * d1_blending) + r[y+yw,xd1+xw,j] * (1-d1_blending))
                                        else:
                                            r[y+yw,xd+xw,j] = <uint8_t>(rvalue * c + r[y+yw,xd+xw,j] * (1-c))
                                    elif not discard_occluded: # Occluded point: Foreground point should be projected before occluded point (Only if activated)
                                        if interpolate:
                                            r[y+yw,xd0+xw,j] = <uint8_t>(((rvalue * c_occ + r[y+yw,xd0+xw,j] * (1-c_occ)) * (1-d1_blending)) + r[y+yw,xd0+xw,j] * d1_blending)
                                            if 0 <= xd1+xw and xd1+xw <= width-1:
                                                r[y+yw,xd1+xw,j] = <uint8_t>(((rvalue * c_occ + r[y+yw,xd1+xw,j] * (1-c_occ)) * d1_blending) + r[y+yw,xd1+xw,j] * (1-d1_blending))
                                            l[y+yw,x+xw,j] = <uint8_t>((r[y+yw,xd0+xw,j]*(1-d1_blending)+r[y+yw,xd1+xw,j]*d1_blending) * c + l[y+yw,x+xw,j] * (1-c)) 
                                        else:
                                            r[y+yw,xd+xw,j] = <uint8_t>(rvalue * c_occ + r[y+yw,xd+xw,j] * (1-c_occ))
                                            l[y+yw,x+xw,j] = <uint8_t>(r[y+yw,xd+xw,j] * c + l[y+yw,x+xw,j] * (1-c))
                                else:#Left side occlusion (known)
                                    l[y+yw,x+xw,j] = <uint8_t>(rvalue * c + l[y+yw,x+xw,j] * (1-c))  

                
                sample_i +=1

            x = x-1 if direction == 0 else x+1

    return sample_i

cpdef int virtual_projection_scan_max_dist(uint8_t[:,:,:] l, uint8_t[:,:,:] r, float[:,:] g, 
 int width, int height, int channels, bint uniform_color, int wsize, int wsize_agg_x, int wsize_agg_y, int direction, float c, float c_occ, uint8_t[:,:] g_occ, bint discard_occluded, bint interpolate):

    """
    Virtual projection using sparse disparity.
    Different color for each pixel of wsize window

    Parameters
    ----------
    l: np.numpy [H,W,C] np.uint8
        Left original image
    r: np.numpy [H,W,C] np.uint8
        Right original image        
    g: np.numpy [H,W] np.float32
        Sparse disparity
    width: int
        Image width (l.shape[1])
    height: int
        Image height (l.shape[0])      
    channels: int
        Image channels (l.shape[3]) 
    wsize: int
        Max projection patch size (Default 5)     
    wsize_agg: int
        Window size for color computation (Default 5)                      
    direction: int mod 2
        Projection direction (left->right or right->left) (Default 0)
    c: float
        alpha blending factor
    c_occ: float
        alpha blending factor in occluded areas
    g_occ: np.numpy [H,W] np.uint8
        Occlusion mask (If not present use np.zeros(l.shape, dtype=np.uint8))

    Returns
    -------
    sample_i:
        number of points projected
    """

    cdef int sample_i = 0
    cdef int x,y,j,d,xd,xd1,d1,d0,xd0,k
    cdef float d1_blending
    #Window size factor (wsize=2*n+1 -> n = (wsize-1)/2)
    cdef int n = <int> ((wsize -1) // 2)
    #Window size factor (wsize=2*n+1 -> n = (wsize-1)/2)
    cdef int n_agg_x = <int> ((wsize_agg_x -1) // 2)
    cdef int n_agg_y = <int> ((wsize_agg_y -1) // 2)
    #Distance parameters
    cdef uint16_t pa,pb
    #Window coords
    cdef int xw,yw,xw_agg,yw_agg
    cdef int min_n_y,max_n_y,min_n_x,max_n_x, adapt_win_size

    #TODO finishsk
    cdef int used_bins[256]
    cdef int n_bins = 256
    cdef uint8_t min_bin = 0
    cdef int min_bin_value = 1000000
    
    min_n_y = n
    max_n_y = n
    min_n_x = n
    max_n_x = n
    pa = 0
    pb = 255

    for y in range(height):
        x = width-1 if direction == 0 else 0
        #x = 0 if direction == 0 else width-1
        while (direction != 0 and x < width) or (direction == 0 and x>=0):
            if g[y,x] > 0:
                d = <int>(round(g[y,x]))
                d0 = <int>(floor(g[y,x])) #5.43 -> 5
                d1 = <int>(ceil(g[y,x]))  #5.43 -> 6
                d1_blending = g[y,x]-d0   #0.43 -> d_blending = 1-0.43 = 0.57
                
                xd = x-d
                xd0 = x-d0
                xd1 = x-d1
                
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

                    #Confidence weight: less confidence -> less distinctness
                    #Use a certain window
                    #if adapt_win_size > 1:
                    for yw in range(-min_n_y,max_n_y+1):
                        for xw in range(-min_n_x,max_n_x+1):
                            if 0 <= y+yw and y+yw <= height-1 and 0 <= x+xw and x+xw <= width-1:        

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

                                if  0 <= xd0+xw and xd0+xw <= width-1:
                                    #Occlusion check
                                    if g_occ[y,x] == 0:#Not occluded point           
                                        l[y+yw,x+xw,j] = <uint8_t>(((pa+pb)/2) * c + l[y+yw,x+xw,j] * (1-c))
                                        if interpolate:
                                            r[y+yw,xd0+xw,j] = <uint8_t>(((((pa+pb)/2) * c + r[y+yw,xd0+xw,j] * (1-c)) * (1-d1_blending)) + r[y+yw,xd0+xw,j] * d1_blending)
                                            if 0 <= xd1+xw and xd1+xw <= width-1:
                                                r[y+yw,xd1+xw,j] = <uint8_t>(((((pa+pb)/2) * c + r[y+yw,xd1+xw,j] * (1-c)) * d1_blending) + r[y+yw,xd1+xw,j] * (1-d1_blending))
                                        else:
                                            r[y+yw,xd+xw,j] = <uint8_t>(((pa+pb)/2) * c + r[y+yw,xd+xw,j] * (1-c))
                                    elif not discard_occluded: # Occluded point: Foreground point should be projected before occluded point (Only if activated)
                                        if interpolate:
                                            r[y+yw,xd0+xw,j] = <uint8_t>(((((pa+pb)/2) * c_occ + r[y+yw,xd0+xw,j] * (1-c_occ)) * (1-d1_blending)) + r[y+yw,xd0+xw,j] * d1_blending)
                                            if 0 <= xd1+xw and xd1+xw <= width-1:
                                                r[y+yw,xd1+xw,j] = <uint8_t>(((((pa+pb)/2) * c_occ + r[y+yw,xd1+xw,j] * (1-c_occ)) * d1_blending) + r[y+yw,xd1+xw,j] * (1-d1_blending))
                                            l[y+yw,x+xw,j] = <uint8_t>((r[y+yw,xd0+xw,j]*(1-d1_blending)+r[y+yw,xd1+xw,j]*d1_blending) * c + l[y+yw,x+xw,j] * (1-c))             
                                        else:
                                            r[y+yw,xd+xw,j] = <uint8_t>(((pa+pb)/2) * c_occ + r[y+yw,xd+xw,j] * (1-c_occ))
                                            l[y+yw,x+xw,j] = <uint8_t>(r[y+yw,xd+xw,j] * c + l[y+yw,x+xw,j] * (1-c))
                                else:#Left side occlusion (known)
                                    l[y+yw,x+xw,j] = <uint8_t>(((pa+pb)/2) * c + l[y+yw,x+xw,j] * (1-c))        
                                                    
                sample_i +=1 

            x = x-1 if direction == 0 else x+1
    
    return sample_i

"""
Reshape GT (HxW) to (Nx4) where:
- N: points of GT
- 0: x coord of a point
- 1: y coord of a point
- 2: GT value in (x,y)
- 3: GT confidence value (always 1)
Points with GT=0 are considered invalid so are filtered out
"""
def gt_reshape(_gt):
    cdef float[:,:] gt = _gt
    cdef int height = <int>_gt.shape[0]
    cdef int width = <int>_gt.shape[1]

    cdef int i = 0
    cdef int y,x
    cdef np.ndarray gtp = np.zeros((width*height, 4), dtype=np.float32)
    cdef float[:,:] gtp_ptr = gtp

    for y in range(height):
        for x in range(width):
            if gt[y,x] > 0:
                gtp_ptr[i,0] = x
                gtp_ptr[i,1] = y
                gtp_ptr[i,2] = gt[y,x]
                gtp_ptr[i,3] = 1
                i = i+1

    return gtp[:i]


#From: https://cse.usf.edu/~kchriste/tools/gennorm.c
cdef float norm(float std):
    cdef float u,r,theta    #Variables for Box-Muller method
    cdef float x            #Normal(0, 1) rv
    cdef float norm_rv      #The adjusted normal rv

    u = 0.0
    while u == 0.0:
        u = <float>rand() / UINT32_MAX
    
    r = sqrt(-2.0 * log(u))

    theta = 0.0
    while theta == 0.0:
        theta = 2.0 * M_PI * (<float>rand() / UINT32_MAX)

    x = r * cos(theta)
    norm_rv = (x * std)

    return norm_rv

cdef float uniform(float std):
    cdef float u
    cdef float result
    u = <float>rand() / UINT32_MAX
    result = u * std
    return result
