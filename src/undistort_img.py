from glob import glob
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def calibrate(is_save=False):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    img_paths = glob('../camera_cal/calibration*.jpg')
    for img_path_ in img_paths:
        img_fn = os.path.basename(img_path_)[:-4]
        img = cv2.imread(img_path_)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            if is_save:
                output_images_dir = '../output_images'
                output_chessboard_coners = os.path.join(output_images_dir, 'chessboard_conners')
                if not os.path.isdir(output_chessboard_coners):
                    os.makedirs(output_chessboard_coners)
                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
                cv2.imwrite(os.path.join(output_chessboard_coners, '{}.jpg'.format(img_fn)), img)

    w, h = 1280, 720
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)

    return ret, mtx, dist, rvecs, tvecs


def undistort(img, mtx, dist):
    undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)

    return undistorted_img


if __name__ == '__main__':
    output_images_dir = '../output_images'
    output_undistorted_img = os.path.join(output_images_dir, 'undistorted')
    if not os.path.isdir(output_undistorted_img):
        os.makedirs(output_undistorted_img)

    ret, mtx, dist, rvecs, tvecs = calibrate(is_save=True)

    img_paths = glob('../camera_cal/calibration*.jpg')
    for img_path_ in img_paths:
        img_fn = os.path.basename(img_path_)[:-4]
        img = mpimg.imread(img_path_)
        undistorted_img = undistort(img, mtx, dist)

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=20)
        ax2.imshow(undistorted_img)
        ax2.set_title('Undistorted Image', fontsize=20)
        plt.savefig(os.path.join(output_undistorted_img, '{}.jpg'.format(img_fn)))

    output_undistorted_img = os.path.join(output_images_dir, 'undistorted_test_images')
    if not os.path.isdir(output_undistorted_img):
        os.makedirs(output_undistorted_img)

    img_paths = glob('../test_images/*.jpg')
    for img_path_ in img_paths:
        img_fn = os.path.basename(img_path_)[:-4]
        img = mpimg.imread(img_path_)
        undistorted_img = undistort(img, mtx, dist)

        plt.cla()
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=20)
        ax2.imshow(undistorted_img)
        ax2.set_title('Undistorted Image', fontsize=20)
        plt.imsave(os.path.join(output_undistorted_img, 'undistorted_{}.jpg'.format(img_fn)), undistorted_img)
        plt.savefig(os.path.join(output_undistorted_img, 'both_{}.jpg'.format(img_fn)))
