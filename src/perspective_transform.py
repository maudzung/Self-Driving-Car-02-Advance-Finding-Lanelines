import os
from glob import glob
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_transform_matrix(src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    return M, Minv


def warped_birdview(img, M):
    h, w = img.shape[:2]
    warped = cv2.warpPerspective(img, M, (w, h))

    return warped


if __name__ == '__main__':
    from undistort_img import undistort, calibrate
    from gradient import get_binary_img

    output_images_dir = '../output_images'
    output_warped_img = os.path.join(output_images_dir, 'warped_test_images')
    if not os.path.isdir(output_warped_img):
        os.makedirs(output_warped_img)

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

        h, w = binary_output.shape[:2]
        src = np.float32([
            [(w / 2) - 55, h / 2 + 100],
            [((w / 6) - 10), h],
            [(w * 5 / 6) + 60, h],
            [(w / 2 + 55), h / 2 + 100]
        ])
        dst = np.float32([
            [(w / 4), 0],
            [(w / 4), h],
            [(w * 3 / 4), h],
            [(w * 3 / 4), 0]
        ])
        M, Minv = get_transform_matrix(src, dst)

        warped = warped_birdview(undistorted_img, M)
        binary_warped = warped_birdview(binary_output, M)

        draw_img = np.copy(undistorted_img)
        draw_img = cv2.polylines(draw_img, [src.astype(np.int32)], True, (255, 0, 0), thickness=10)
        warped = cv2.polylines(warped, [dst.astype(np.int32)], True, (255, 0, 0), thickness=10)

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
        ax1.imshow(draw_img)
        ax1.set_title('Undistorted Image', fontsize=20)
        ax2.imshow(warped)
        ax2.set_title('Perspective Transform Image', fontsize=20)
        plt.savefig(os.path.join(output_warped_img, 'color_{}.jpg'.format(img_fn)))

        ax1.imshow(binary_output, cmap='gray')
        ax1.set_title('Undistorted Image', fontsize=20)
        ax2.imshow(binary_warped, cmap='gray')
        ax2.set_title('Perspective Transform Image', fontsize=20)
        plt.savefig(os.path.join(output_warped_img, 'binary_{}.jpg'.format(img_fn)), cmap='gray')
