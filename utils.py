import cv2
import numpy as np

_color_map_errors_kitti = np.array([
        [ 0,       0.1875, 149,  54,  49],
        [ 0.1875,  0.375,  180, 117,  69],
        [ 0.375,   0.75,   209, 173, 116],
        [ 0.75,    1.5,    233, 217, 171],
        [ 1.5,     3,      248, 243, 224],
        [ 3,       6,      144, 224, 254],
        [ 6,      12,       97, 174, 253],
        [12,      24,       67, 109, 244],
        [24,      48,       39,  48, 215],
        [48,  np.inf,       38,   0, 165]
]).astype(float)

def color_error_image_kitti(errors, scale=1, mask=None, BGR=True, dilation=1):
    errors_flat = errors.flatten()
    colored_errors_flat = np.zeros((errors_flat.shape[0], 3))
    
    for col in _color_map_errors_kitti:
        col_mask = np.logical_and(errors_flat>=col[0]/scale, errors_flat<=col[1]/scale)
        colored_errors_flat[col_mask] = col[2:]
        
    if mask is not None:
        colored_errors_flat[mask.flatten() == 0] = 0

    if not BGR:
        colored_errors_flat = colored_errors_flat[:, [2, 1, 0]]

    colored_errors = colored_errors_flat.reshape(errors.shape[0], errors.shape[1], 3).astype(np.uint8)

    if dilation>0:
        kernel = np.ones((dilation, dilation))
        colored_errors = cv2.dilate(colored_errors, kernel)
    return colored_errors


def guided_visualize(disp, gt, valid, scale=1, dilation=7):
    H,W = disp.shape[:2]

    error = np.abs(disp-gt)
    error[valid==0] = 0
    
    bad1 = np.zeros((H,W,3), dtype=np.uint8)
    bad1[error > 1.,:] = (49, 54,  149)    
    bad1[error <= 1.,:] = (165,  0, 38)   
    bad1[valid==0,:] = (0,0,0)

    bad2 = np.zeros((H,W,3), dtype=np.uint8)
    bad2[error > 2.,:] = (49, 54,  149)       
    bad2[error <= 2.,:] = (165,  0, 38)   
    bad2[valid==0,:] = (0,0,0)

    bad3 = np.zeros((H,W,3), dtype=np.uint8)
    bad3[error > 3.,:] = (49, 54,  149)       
    bad3[error <= 3.,:] = (165,  0, 38) 
    bad3[valid==0,:] = (0,0,0)

    bad4 = np.zeros((H,W,3), dtype=np.uint8)
    bad4[error > 4.,:] = (49, 54,  149)     
    bad4[error <= 4.,:] = (165,  0, 38)  
    bad4[valid==0,:] = (0,0,0)
 
    if dilation>0:
        kernel = np.ones((dilation, dilation))
        bad1 = cv2.dilate(bad1, kernel)
        bad2 = cv2.dilate(bad2, kernel)
        bad3 = cv2.dilate(bad3, kernel)
        bad4 = cv2.dilate(bad4, kernel)

    avgerr = color_error_image_kitti(error, scale=scale, mask=valid, dilation=dilation)
    
    rms = (disp-gt)**2
    rms = np.sqrt(rms)

    rms = color_error_image_kitti(rms, scale=scale, mask=valid, dilation=dilation)
    
    return {'bad 1.0':bad1, 'bad 2.0':bad2, 'bad 3.0': bad3, 'bad 4.0':bad4, 'avgerr':avgerr, 'rms':rms}
