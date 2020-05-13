import os
from glob import glob
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt


# Using histogram --> finding peak at left and right
# Using sliding window method to find the curve
def find_lane_sliding_window(binary_birdview, nwindows, margin, minpix, lane_left, lane_right, ym_per_pix, xm_per_pix):
    # # Clear the 2 lane buffers
    # lane_left.reset()
    # lane_right.reset()

    h, w = binary_birdview.shape[:2]
    step_y = int(h / nwindows)
    nonzeroxy = binary_birdview.nonzero()
    nonzerox = nonzeroxy[1]
    nonzeroy = nonzeroxy[0]

    out_img = np.dstack((binary_birdview, binary_birdview, binary_birdview)) * 255

    half_below = binary_birdview[int(h / 2):, :]
    hist = np.sum(half_below, axis=0)
    left_x_peak = np.argmax(hist[:int(w / 2)])
    right_x_peak = np.argmax(hist[int(w / 2):]) + int(w / 2)

    left_x_cur = left_x_peak
    right_x_cur = right_x_peak

    left_lane_idexes = []
    right_lane_indexes = []
    for window_idx in range(nwindows):
        low_y = h - step_y * (window_idx + 1)
        high_y = h - step_y * window_idx
        low_left_x = left_x_cur - margin
        high_left_x = left_x_cur + margin
        low_right_x = right_x_cur - margin
        high_right_x = right_x_cur + margin

        cv2.rectangle(out_img, (low_left_x, low_y), (high_left_x, high_y), (0, 255, 0), 5)
        cv2.rectangle(out_img, (low_right_x, low_y), (high_right_x, high_y), (0, 255, 0), 5)

        left_lane_idx = \
            ((nonzerox >= low_left_x) & (nonzerox < high_left_x) & (nonzeroy >= low_y) & (nonzeroy < high_y)).nonzero()[
                0]
        right_lane_idx = \
            ((nonzerox >= low_right_x) & (nonzerox < high_right_x) & (nonzeroy >= low_y) & (
                    nonzeroy < high_y)).nonzero()[0]

        if len(left_lane_idx) > minpix:
            left_x_cur = np.int(np.mean(nonzerox[left_lane_idx]))
        if len(right_lane_idx) > minpix:
            right_x_cur = np.int(np.mean(nonzerox[right_lane_idx]))
        left_lane_idexes.append(left_lane_idx)
        right_lane_indexes.append(right_lane_idx)

    try:
        left_lane_idexes = np.concatenate(left_lane_idexes)
        right_lane_indexes = np.concatenate(right_lane_indexes)
    except ValueError:
        pass

    left_lane_x = nonzerox[left_lane_idexes]
    left_lane_y = nonzeroy[left_lane_idexes]
    right_lane_x = nonzerox[right_lane_indexes]
    right_lane_y = nonzeroy[right_lane_indexes]

    detected = True
    if len(left_lane_x) == 0:
        left_fit_pixel = lane_left.last_fit_pixel
        left_fit_meter = lane_left.left_fit_meter
        detected = False
    else:
        left_fit_pixel = np.polyfit(left_lane_y, left_lane_x, 2)
        left_fit_meter = np.polyfit(left_lane_y * ym_per_pix, left_lane_x * xm_per_pix, 2)

    if len(right_lane_x) == 0:
        right_fit_pixel = lane_right.last_fit_pixel
        right_fit_meter = lane_left.right_fit_meter
        detected = False
    else:
        right_fit_pixel = np.polyfit(right_lane_y, right_lane_x, 2)
        right_fit_meter = np.polyfit(right_lane_y * ym_per_pix, right_lane_x * xm_per_pix, 2)

    lane_left.update_lane(left_fit_pixel, left_fit_meter, detected, left_lane_x, left_lane_y)
    lane_right.update_lane(right_fit_pixel, right_fit_meter, detected, right_lane_x, right_lane_y)

    # Take average of previous frames
    left_fit_pixel = lane_left.average_fit()
    right_fit_pixel = lane_right.average_fit()

    ploty = np.linspace(0, h - 1, h)
    left_fit_x = left_fit_pixel[0] * ploty ** 2 + left_fit_pixel[1] * ploty + left_fit_pixel[2]
    right_fit_x = right_fit_pixel[0] * ploty ** 2 + right_fit_pixel[1] * ploty + right_fit_pixel[2]

    out_img[left_lane_y, left_lane_x] = [255, 0, 0]
    out_img[right_lane_y, right_lane_x] = [0, 0, 255]

    return out_img, lane_left, lane_right, left_fit_x, right_fit_x, ploty


def find_lane_based_on_previous_frame(binary_birdview, margin, lane_left, lane_right, ym_per_pix, xm_per_pix):
    h, w = binary_birdview.shape[:2]
    nonzeroxy = binary_birdview.nonzero()
    nonzerox = nonzeroxy[1]
    nonzeroy = nonzeroxy[0]

    left_fit_pixel = lane_left.last_fit_pixel
    right_fit_pixel = lane_right.last_fit_pixel

    left_fit_x = left_fit_pixel[0] * nonzeroy ** 2 + left_fit_pixel[1] * nonzeroy + left_fit_pixel[2]
    right_fit_x = right_fit_pixel[0] * nonzeroy ** 2 + right_fit_pixel[1] * nonzeroy + right_fit_pixel[2]

    left_lane_idx = (nonzerox >= left_fit_x - margin) & (nonzerox < left_fit_x + margin)
    right_lane_idx = (nonzerox >= right_fit_x - margin) & (nonzerox < right_fit_x + margin)

    left_lane_x = nonzerox[left_lane_idx]
    left_lane_y = nonzeroy[left_lane_idx]
    right_lane_x = nonzerox[right_lane_idx]
    right_lane_y = nonzeroy[right_lane_idx]

    detected = True
    if len(left_lane_x) == 0:
        left_fit_pixel = lane_left.last_fit_pixel
        left_fit_meter = lane_left.left_fit_meter
        detected = False
    else:
        left_fit_pixel = np.polyfit(left_lane_y, left_lane_x, 2)
        left_fit_meter = np.polyfit(left_lane_y * ym_per_pix, left_lane_x * xm_per_pix, 2)

    if len(right_lane_x) == 0:
        right_fit_pixel = lane_right.last_fit_pixel
        right_fit_meter = lane_left.right_fit_meter
        detected = False
    else:
        right_fit_pixel = np.polyfit(right_lane_y, right_lane_x, 2)
        right_fit_meter = np.polyfit(right_lane_y * ym_per_pix, right_lane_x * xm_per_pix, 2)

    lane_left.update_lane(left_fit_pixel, left_fit_meter, detected, left_lane_x, left_lane_y)
    lane_right.update_lane(right_fit_pixel, right_fit_meter, detected, right_lane_x, right_lane_y)

    # Take average of previous frames
    left_fit_pixel = lane_left.average_fit()
    right_fit_pixel = lane_right.average_fit()

    ploty = np.linspace(0, h - 1, h)
    left_fit_x = left_fit_pixel[0] * ploty ** 2 + left_fit_pixel[1] * ploty + left_fit_pixel[2]
    right_fit_x = right_fit_pixel[0] * ploty ** 2 + right_fit_pixel[1] * ploty + right_fit_pixel[2]

    out_img = np.dstack((binary_birdview, binary_birdview, binary_birdview)) * 255
    out_img[left_lane_y, left_lane_x] = [255, 0, 0]
    out_img[right_lane_y, right_lane_x] = [0, 0, 255]

    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fit_x - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fit_x + margin,
                                                                    ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fit_x - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fit_x + margin,
                                                                     ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    return out_img, lane_left, lane_right, left_fit_x, right_fit_x, ploty


if __name__ == '__main__':
    from undistort_img import undistort, calibrate
    from gradient import get_binary_img
    from perspective_transform import get_transform_matrix, warped_birdview
    from utils import Line

    nwindows = 9
    margin = 100
    minpix = 50

    output_images_dir = '../output_images'
    output_detectedline_img = os.path.join(output_images_dir, 'detected_lane_test_images')
    if not os.path.isdir(output_detectedline_img):
        os.makedirs(output_detectedline_img)

    img_paths = glob('../test_images/*.jpg')

    ret, mtx, dist, rvecs, tvecs = calibrate(is_save=True)

    thresh_gradx = (20, 100)
    thresh_grady = (20, 100)
    thresh_mag = (30, 100)
    thresh_dir = (0.7, 1.3)
    thresh_s_channel = (170, 255)
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
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

        lane_left = Line(buffer_len=20)
        lane_right = Line(buffer_len=20)

        #     left_lane_x, left_lane_y, right_lane_x, right_lane_y, out_img = find_lane_boundary(binary_birdview)
        out_img, lane_left, lane_right, left_fit_x, right_fit_x, ploty = find_lane_sliding_window(binary_birdview,
                                                                                                  nwindows, margin,
                                                                                                  minpix, lane_left,
                                                                                                  lane_right,
                                                                                                  ym_per_pix,
                                                                                                  xm_per_pix)
        plt.cla()
        plt.plot(left_fit_x, ploty, color='yellow')
        plt.plot(right_fit_x, ploty, color='yellow')
        plt.imshow(out_img)
        plt.savefig(os.path.join(output_detectedline_img, 'n{}.jpg'.format(img_fn)))
    #   plt.imshow(binary_birdview, cmap='gray')
