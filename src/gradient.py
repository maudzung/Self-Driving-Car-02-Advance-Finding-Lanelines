import os
from glob import glob
import sys

import cv2
import numpy as np


def abs_sobel_thresh(image, orient, sobel_kernel, thresh):
    # Calculate directional gradient
    if orient == 'x':
        grad = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient == 'y':
        grad = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Apply threshold

    grad = ((grad / np.max(grad)) * 255).astype(np.uint8)

    grad_binary = np.zeros_like(grad)
    grad_binary[(grad > thresh[0]) * (grad < thresh[1])] = 1
    return grad_binary


def mag_thresh(image, sobel_kernel, thresh):
    # Calculate gradient magnitude
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    # Apply threshold
    mag = ((mag / np.max(mag)) * 255).astype(np.uint8)
    mag_binary = np.zeros_like(mag)
    mag_binary[(mag > thresh[0]) * (mag < thresh[1])] = 1
    return mag_binary


def dir_threshold(image, sobel_kernel, thresh):
    # Calculate gradient direction
    grad_x = np.absolute(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    grad_y = np.absolute(cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

    direction = np.arctan2(grad_y, grad_x)
    dir_binary = np.zeros_like(direction, dtype=np.uint8)
    # Apply threshold
    dir_binary[(direction > thresh[0]) * (direction < thresh[1])] = 1
    return dir_binary


def get_binary_gradient_img(img, thresh_gradx, thresh_grady, thresh_mag, thresh_dir):
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    mask_gradx = abs_sobel_thresh(gray_img, orient='x', sobel_kernel=3, thresh=thresh_gradx)
    mask_grady = abs_sobel_thresh(gray_img, orient='y', sobel_kernel=3, thresh=thresh_grady)
    mask_mag = mag_thresh(gray_img, sobel_kernel=3, thresh=thresh_mag)
    mask_direction = dir_threshold(gray_img, sobel_kernel=15, thresh=thresh_dir)

    binary_output = np.zeros_like(mask_gradx)
    binary_output[((mask_gradx == 1) & (mask_grady == 1)) | ((mask_mag == 1) & (mask_direction == 1))] = 1

    return binary_output


def get_binary_s_channel_img(img, thresh_s_channel):
    hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls_img[:, :, 2]
    binary_output = np.zeros_like(s_channel, dtype=np.uint8)
    binary_output[(s_channel > thresh_s_channel[0]) & (s_channel < thresh_s_channel[1])] = 1

    return binary_output


def get_binary_img(img, thresh_gradx, thresh_grady, thresh_mag, thresh_dir, thresh_s_channel):
    grad_binary_output = get_binary_gradient_img(img, thresh_gradx, thresh_grady, thresh_mag, thresh_dir)
    s_channel_binary_output = get_binary_s_channel_img(img, thresh_s_channel)
    binary_output = grad_binary_output | s_channel_binary_output

    return binary_output


if __name__ == '__main__':
    # sys.path.append('./')
    from undistort_img import undistort, calibrate

    output_images_dir = '../output_images'
    output_binary_img = os.path.join(output_images_dir, 'binary_test_images')
    if not os.path.isdir(output_binary_img):
        os.makedirs(output_binary_img)

    ret, mtx, dist, rvecs, tvecs = calibrate(is_save=True)

    img_paths = glob('../test_images/*.jpg')
    thresh_gradx = (20, 100)
    thresh_grady = (20, 100)
    thresh_mag = (30, 100)
    thresh_dir = (0.7, 1.3)
    thresh_s_channel = (170, 255)
    for idx, img_path_ in enumerate(img_paths):
        img_fn = os.path.basename(img_path_)[:-4]
        img = cv2.cvtColor(cv2.imread(img_path_), cv2.COLOR_BGR2RGB)  # BGR --> RGB
        undistorted_img = undistort(img, mtx, dist)
        # grad_binary_output = get_binary_gradient_img(img, thresh_gradx, thresh_grady, thresh_mag, thresh_dir)
        # s_channel_binary_output = get_binary_s_channel_img(img, thresh_s_channel)
        binary_output = get_binary_img(undistorted_img, thresh_gradx, thresh_grady, thresh_mag, thresh_dir,
                                       thresh_s_channel)

        cv2.imwrite(os.path.join(output_binary_img, 'binary_{}.jpg'.format(img_fn)), binary_output * 255) # Gray img
