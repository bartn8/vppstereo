import numpy as np 
import cv2 
import pathlib

SCRIPT_PATH = pathlib.Path(__file__).parent.resolve()
DOT_PATTERN = cv2.imread(f'{SCRIPT_PATH}/kinect-pattern_3x3.png', 0)

# https://github.com/ankurhanda/simkinect/blob/master/add_noise.py
def add_gaussian_shifts(depth, std=1/2.0):

    rows, cols = depth.shape[:2]
    gaussian_shifts = np.random.normal(0, std, size=(rows, cols, 2))
    gaussian_shifts = gaussian_shifts.astype(np.float32)

    # creating evenly spaced coordinates  
    xx = np.linspace(0, cols-1, cols)
    yy = np.linspace(0, rows-1, rows)

    # get xpixels and ypixels 
    xp, yp = np.meshgrid(xx, yy)

    xp = xp.astype(np.float32)
    yp = yp.astype(np.float32)

    xp_interp = np.minimum(np.maximum(xp + gaussian_shifts[:, :, 0], 0.0), cols)
    yp_interp = np.minimum(np.maximum(yp + gaussian_shifts[:, :, 1], 0.0), rows)

    depth_interp = cv2.remap(depth, xp_interp, yp_interp, cv2.INTER_LINEAR)

    return depth_interp

def resize_images_to_match(image1, image2):
    """
    Resize or pad image1 to match the dimensions of image2 using cv2.BORDER_WRAP.

    Args:
    image1: The first input image.
    image2: The second input image.

    Returns:
    First image is resized/padded to match the dimensions of the second image.
    """

    # Get the dimensions of both images
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]

    # If both images have the same dimensions, return them as is
    if height1 == height2 and width1 == width2:
        return image1

    # Determine whether to pad or crop image1 to match the dimensions of image2
    if height1 < height2 or width1 < width2:
        # Pad image1 to match the dimensions of image2
        pad_height = max(height2 - height1, 0)
        pad_width = max(width2 - width1, 0)
        border_type = cv2.BORDER_WRAP
        padded_image = cv2.copyMakeBorder(image1, 0, pad_height, 0, pad_width, border_type)
        return padded_image
    else:
        # Crop image1 to match the dimensions of image2
        crop_height = min(height1, height2)
        crop_width = min(width1, width2)
        cropped_image = image1[:crop_height, :crop_width]
        return cropped_image
    
# https://github.com/ankurhanda/simkinect/blob/master/add_noise.py
def filterDisp(disp, dot_pattern_, invalid_disp_):

    size_filt_ = 9

    xx = np.linspace(0, size_filt_-1, size_filt_)
    yy = np.linspace(0, size_filt_-1, size_filt_)

    xf, yf = np.meshgrid(xx, yy)

    xf = xf - int(size_filt_ / 2.0)
    yf = yf - int(size_filt_ / 2.0)

    sqr_radius = (xf**2 + yf**2)
    vals = sqr_radius * 1.2**2 

    vals[vals==0] = 1 
    weights_ = 1 /vals  

    fill_weights = 1 / ( 1 + sqr_radius)
    fill_weights[sqr_radius > 9] = -1.0 

    disp_rows, disp_cols = disp.shape 
    dot_pattern_rows, dot_pattern_cols = dot_pattern_.shape

    lim_rows = np.minimum(disp_rows - size_filt_, dot_pattern_rows - size_filt_)
    lim_cols = np.minimum(disp_cols - size_filt_, dot_pattern_cols - size_filt_)

    center = int(size_filt_ / 2.0)

    window_inlier_distance_ = 0.1

    out_disp = np.ones_like(disp) * invalid_disp_

    interpolation_map = np.zeros_like(disp)

    for r in range(0, lim_rows):
        for c in range(0, lim_cols):
            if dot_pattern_[r+center, c+center] > 0:
                                
                # c and r are the top left corner 
                window  = disp[r:r+size_filt_, c:c+size_filt_]
                dot_win = dot_pattern_[r:r+size_filt_, c:c+size_filt_] 

                valid_dots = dot_win[window < invalid_disp_]
                                
                n_valids = np.sum(valid_dots) / 255.0 
                n_thresh = np.sum(dot_win) / 255.0 

                if n_valids > n_thresh / 1.2: 

                    mean = np.mean(window[window < invalid_disp_])

                    diffs = np.abs(window - mean)
                    diffs = np.multiply(diffs, weights_)

                    cur_valid_dots = np.multiply(np.where(window<invalid_disp_, dot_win, 0), 
                                                np.where(diffs < window_inlier_distance_, 1, 0))

                    n_valids = np.sum(cur_valid_dots) / 255.0

                    if n_valids > n_thresh / 1.2: 
                    
                        accu = window[center, center] 

                        assert(accu < invalid_disp_)

                        out_disp[r+center, c + center] = round((accu)*8.0) / 8.0

                        interpolation_window = interpolation_map[r:r+size_filt_, c:c+size_filt_]
                        disp_data_window     = out_disp[r:r+size_filt_, c:c+size_filt_]

                        substitutes = np.where(interpolation_window < fill_weights, 1, 0)
                        interpolation_window[substitutes==1] = fill_weights[substitutes ==1 ]

                        disp_data_window[substitutes==1] = out_disp[r+center, c+center]

    return out_disp

def add_kinectv1_noise(depth, focal_length=480.0, baseline_m=0.075, invalid_disp_=99999999.9):
    """
    Add Kinect V1-like noise to a depth image.

    Args:
    depth: The input depth image.
    focal_length: The focal length of the camera (default is 480.0).
    baseline_m: The baseline distance (default is 0.075 meters).
    invalid_disp_: The value representing invalid disparities (default is 99999999.9).

    Returns:
    A noisy depth image with Kinect V1-like noise added.
    """
    
    # Resize the dot pattern to match the size of the depth image
    correct_size_dot_pattern = resize_images_to_match(DOT_PATTERN, depth)

    # Add Gaussian shifts to the depth image
    depth_interp = add_gaussian_shifts(depth)

    # Calculate disparity and round to 1/8th pixel accuracy
    disp_ = focal_length * baseline_m / (depth_interp + 1e-10)
    depth_f = np.round(disp_ * 8.0) / 8.0
    
    # Filter the disparity map
    out_disp = filterDisp(depth_f, correct_size_dot_pattern, invalid_disp_)
    
    # Calculate depth from disparity and handle invalid disparities
    depth = focal_length * baseline_m / out_disp
    depth[out_disp == invalid_disp_] = 0
    
    # Scale depth to centimeters, add noise, and convert back to meters
    scale_factor = 100  # Formula is in cm: convert depth to cm
    noisy_depth = depth.copy()
    noisy_depth[depth>0] = (35130 / np.round((35130 / np.round(depth[depth>0] * scale_factor)) + np.random.normal(size=depth.shape[:2])[depth>0] * (1.0/6.0) + 0.5)) / scale_factor
    
    return noisy_depth
