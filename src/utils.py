import os
from glob import glob
import sys
import collections

import cv2
import numpy as np
import matplotlib.pyplot as plt


# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self, buffer_len=20):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.last_fit_pixel = None
        self.last_fit_meter = None
        self.recent_fits = collections.deque(maxlen=buffer_len)
        # radius of curvature of the line in some units
        self.curvature_in_meter = 0
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

    def reset(self):
        self.recent_fits.clear()

    def update_lane(self, new_curve_fit_pixel, new_curve_fit_meter, detected, new_lane_x, new_lane_y):
        self.detected = detected
        self.last_fit_pixel = new_curve_fit_pixel
        self.last_fit_meter = new_curve_fit_meter
        self.recent_fits.append(new_curve_fit_pixel)
        self.allx = new_lane_x
        self.ally = new_lane_y

    def cal_curvature(self, h, ym_per_pix):
        y_eval = (h - 1) * ym_per_pix # bottom image
        self.curvature_in_meter = np.sqrt(
            (1 + (2 * self.last_fit_meter[0] * y_eval + self.last_fit_meter[1]) ** 2) ** 3) / np.absolute(
            2 * self.last_fit_meter[0])

    def average_fit(self):
        return np.mean(self.recent_fits, axis=0)


# def transform_to_the_road(undistorted_img, Minv, left_lane, right_lane):
def transform_to_the_road(undistorted_img, Minv, left_fit_x, right_fit_x, ploty):
    h, w = undistorted_img.shape[:2]

    road_warped = np.zeros_like(undistorted_img, dtype=np.uint8)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fit_x, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fit_x, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(road_warped, np.int_([pts]), (0, 255, 0))

    # Draw 2 curves
    left_fit_x = left_fit_x.astype(np.int32)
    right_fit_x = right_fit_x.astype(np.int32)
    ploty = ploty.astype(np.int32)

    for idx in range(len(ploty) - 2):
        cv2.line(road_warped, (left_fit_x[idx], ploty[idx]), (left_fit_x[idx + 1], ploty[idx + 1]), (255,0,0), 20)
        cv2.line(road_warped, (right_fit_x[idx], ploty[idx]), (right_fit_x[idx + 1], ploty[idx + 1]), (255,0,0), 20)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    road_unwarped = cv2.warpPerspective(road_warped, Minv, (w, h))  # Warp back to original image space

    blend_img = cv2.addWeighted(undistorted_img, 1., road_unwarped, 0.8, 0)

    return blend_img


if __name__ == '__main__':
    from undistort_img import undistort, calibrate
    from gradient import get_binary_img
    from perspective_transform import get_transform_matrix, warped_birdview
    from detect_lanelines import find_lane_sliding_window

    nwindows = 9
    margin = 100
    minpix = 50
    thresh_gradx = (20, 100)
    thresh_grady = (20, 100)
    thresh_mag = (30, 100)
    thresh_dir = (0.7, 1.3)
    thresh_s_channel = (170, 255)

    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    output_images_dir = '../output_images'
    output_detectedline_img = os.path.join(output_images_dir, 'onroad_test_images')
    if not os.path.isdir(output_detectedline_img):
        os.makedirs(output_detectedline_img)

    img_paths = glob('../test_images/*.jpg')

    ret, mtx, dist, rvecs, tvecs = calibrate(is_save=True)

    for idx, img_path_ in enumerate(img_paths):
        img_fn = os.path.basename(img_path_)[:-4]
        img = cv2.cvtColor(cv2.imread(img_path_), cv2.COLOR_BGR2RGB)  # BGR --> RGB
        undistorted_img = undistort(img, mtx, dist)
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

        warped = warped_birdview(img, M)
        binary_birdview = warped_birdview(binary_output, M)

        lane_left, lane_right = Line(buffer_len=20), Line(buffer_len=20)

        out_img, left_fit, right_fit, left_fit_x, right_fit_x, ploty = find_lane_sliding_window(binary_birdview,
                                                                                                nwindows, margin,
                                                                                                minpix, lane_left,
                                                                                                lane_right, ym_per_pix,
                                                                                                xm_per_pix)

        blend_img = transform_to_the_road(undistorted_img, Minv, left_fit_x, right_fit_x, ploty)

        plt.cla()
        # plt.plot(left_fit_x, ploty, color='yellow')
        # plt.plot(right_fit_x, ploty, color='yellow')
        plt.imshow(blend_img)
        plt.savefig(os.path.join(output_detectedline_img, '{}.jpg'.format(img_fn)))
    #   plt.imshow(binary_warped, cmap='gray')
