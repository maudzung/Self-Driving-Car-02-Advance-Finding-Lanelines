import time
import os
import numpy as np

import cv2
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip

from undistort_img import undistort, calibrate
from gradient import get_binary_img
from perspective_transform import get_transform_matrix, warped_birdview
from detect_lanelines import find_lane_sliding_window, find_lane_based_on_previous_frame
from utils import transform_to_the_road, Line


def calculate_distance_from_lane_center(lane_left, lane_right, w, h, xm_per_pix):
    distance_meter = -100
    if lane_left.detected and lane_right.detected:
        center_bottom_left_x = np.mean(lane_left.allx[lane_left.ally > 0.9 * h])
        center_bottom_right_x = np.mean(lane_right.allx[lane_right.ally > 0.9 * h])
        lane_width = center_bottom_right_x - center_bottom_left_x
        distance_pixel = abs((center_bottom_left_x + lane_width / 2) - w / 2)
        distance_meter = distance_pixel * xm_per_pix
    return distance_meter


def compose_final_output(onroad_img, out_binary_birdview, w, h, mean_curvature_in_meter, distance_meter):
    off_x, off_y = 100, 30
    thumb_ratio = 0.2
    thumb_h, thumb_w = int(thumb_ratio * h), int(thumb_ratio * w)

    # add a gray rectangle to highlight the upper area
    topmask = np.copy(onroad_img)
    topmask = cv2.rectangle(topmask, pt1=(0, 0), pt2=(w, thumb_h + off_y * 2), color=(0, 0, 0), thickness=cv2.FILLED)
    blend_on_road = cv2.addWeighted(src1=onroad_img, alpha=1., src2=topmask, beta=0.2, gamma=0)

    # add thumbnail of binary image
    thumb_binary = cv2.resize(out_binary_birdview, dsize=(thumb_w, thumb_h))
    # thumb_binary = np.dstack([thumb_binary, thumb_binary, thumb_binary]) * 255
    blend_on_road[off_y:thumb_h + off_y, off_x:off_x + thumb_w, :] = thumb_binary

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blend_on_road, 'Birdview', (int(off_x + thumb_w / 4), int(thumb_h + 2 * off_y)), font,
                0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(blend_on_road, 'Lane curvature: {:.2f} m'.format(mean_curvature_in_meter), (700, 60), font,
                0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(blend_on_road, 'Distance from lane center: {:.2f} m'.format(distance_meter), (700, 130), font, 0.9,
                (255, 255, 255), 2, cv2.LINE_AA)

    return blend_on_road


def process_image(image):
    global lane_left, lane_right, frame_idx
    frame_idx += 1
    h, w = image.shape[:2]
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

    undistorted_img = undistort(image, mtx, dist)
    binary_output = get_binary_img(undistorted_img, thresh_gradx, thresh_grady, thresh_mag, thresh_dir,
                                   thresh_s_channel)

    M, Minv = get_transform_matrix(src, dst)
    binary_birdview = warped_birdview(binary_output, M)

    # diff_curvature_in_meter = abs(lane_right.curvature_in_meter - lane_left.curvature_in_meter)

    # if (frame_idx > 0) and lane_left.detected and lane_right.detected and (diff_curvature_in_meter < 5000.):
    if (frame_idx > 0) and lane_left.detected and lane_right.detected:
        out_binary_birdview, lane_left, lane_right, left_fit_x, right_fit_x, ploty = find_lane_based_on_previous_frame(
            binary_birdview, margin, lane_left, lane_right, ym_per_pix, xm_per_pix)
    else:
        out_binary_birdview, lane_left, lane_right, left_fit_x, right_fit_x, ploty = find_lane_sliding_window(
            binary_birdview,
            nwindows, margin,
            minpix, lane_left,
            lane_right, ym_per_pix, xm_per_pix)

    lane_left.cal_curvature(h, xm_per_pix)
    lane_right.cal_curvature(h, xm_per_pix)

    distance_meter = calculate_distance_from_lane_center(lane_left, lane_right, w, h, xm_per_pix)
    mean_curvature_in_meter = (lane_left.curvature_in_meter + lane_right.curvature_in_meter)/2

    onroad_img = transform_to_the_road(undistorted_img, Minv, left_fit_x, right_fit_x, ploty)
    # onroad_img = cv2.polylines(onroad_img, [src.astype(np.int32)], True, (255, 0, 0), 5)

    blend_on_road = compose_final_output(onroad_img, out_binary_birdview, w, h, mean_curvature_in_meter, distance_meter)
    return blend_on_road


def main():
    video_output_dir = '../test_videos_output'
    if not os.path.isdir(video_output_dir):
        os.makedirs(video_output_dir)

    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    ##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
    video_fn = 'project_video.mp4'
    # video_fn = 'challenge_video.mp4'
    # video_fn = 'harder_challenge_video.mp4'
    video_output_path = os.path.join(video_output_dir, video_fn)
    # clip1 = VideoFileClip(os.path.join('../', video_fn)).subclip(0,2)
    clip1 = VideoFileClip(os.path.join('../', video_fn))
    white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
    white_clip.write_videofile(video_output_path, audio=False)


if __name__ == '__main__':
    lane_left, lane_right = Line(buffer_len=20), Line(buffer_len=20)
    frame_idx = -1
    nwindows = 9
    margin = 100
    minpix = 50
    thresh_gradx = (20, 100)
    thresh_grady = (20, 100)
    thresh_mag = (30, 100)
    thresh_dir = (0.7, 1.3)
    thresh_s_channel = (170, 255)
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    ret, mtx, dist, rvecs, tvecs = calibrate(is_save=False)
    main()
