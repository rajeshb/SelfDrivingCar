import os
import cv2
import glob
import numpy as np
import glob
import matplotlib.image as mpimg
from image_functions import *
import matplotlib.pyplot as plt
from Line import Line

# project class
class AdvancedLaneLine:
    def __init__(self, session=random_prefix(), debug=False):
        # calibrate camera
        images = glob.glob("./camera_cal/calibration*.jpg")
        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = calibrate_camera(images)

        # previous lines
        self.left_line = Line()
        self.right_line = Line()

        # perspective transform coordinates
        self.src, self.dst = get_src_dst_coordinates()
        
        # debug purposes
        self.frame_number = 0
        self.session = session
        self.debug = debug
        
        # counters
        self.sliding_window_count = 0
        self.from_existing_count = 0
        return

    # get the session/frame/file name for the given file name suffix
    def get_session_frame_file_name(self, file_suffix):
        return "{}_{}_{}".format(self.session, self.frame_number, file_suffix)

    # save intermediate images, debug purposes 
    def save_image(self, img, suffix="out", out_folder="output"):
        if self.debug == True:
            outfile = "./{}/{}.jpg".format(out_folder, self.get_session_frame_file_name(suffix))
            mpimg.imsave(outfile, img)
        return

    def find_lane_lines(self, unwarp):
        if self.left_line.detected == False or self.right_line.detected == False:
            sliding_windows_list, leftx, lefty, rightx, righty = sliding_windows2(unwarp)
            slide_win_img = sliding_windows_image(unwarp, sliding_windows_list)
            self.save_image(slide_win_img, "slide")
            self.sliding_window_count += 1
        else:
            leftx, lefty, rightx, righty = find_lane_lines_from_existing(unwarp, self.left_line.best_fit, self.right_line.best_fit)
            self.from_existing_count += 1

        left_fit, right_fit = (None, None)
        # remove outliers
        if len(leftx) > 0:
            leftx, lefty = outlier_removal(leftx, lefty)
            if len(leftx) > 2:
                left_fit = np.polyfit(lefty, leftx, 2)
        if len(rightx) > 0:
            rightx, righty = outlier_removal(rightx, righty)
            if len(rightx) > 2:
                right_fit = np.polyfit(righty, rightx, 2)
        if left_fit is not None or right_fit is not None:
            ploty = np.linspace(0, unwarp.shape[0]-1, unwarp.shape[0])
            left_fitx, right_fitx = (None, None)
            if left_fit is not None:
                left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            if right_fit is not None:
                right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        if left_fitx is not None and right_fitx is not None:
            # Create an output image to draw on and  visualize the result
            lines_out = np.dstack((unwarp, unwarp, unwarp))*255
            ploty = np.linspace(0, unwarp.shape[0]-1, unwarp.shape[0])

            # Recast the x and y points into usable format for cv2.fillPoly()
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))
            # Draw the lane onto the warped blank image
            cv2.polylines(lines_out, np.int32([pts]), 1, (255,255,0))
            self.save_image(lines_out, "lines")
            
            # check if the fits are looking valid, if not for the first frame
            if self.left_line.best_fit is None or self.right_line.best_fit is None or (is_parallel(left_fit, right_fit) and is_distance_in_range(left_fitx, right_fitx)):
                self.left_line.add(left_fit)
                self.right_line.add(right_fit)
                return

        # reset the flags so that next time, it will do sliding windows
        self.left_line.detected = False
        self.right_line.detected = False
        return
    
    def overlay_lane_lines(self, undist, unwarp, Minv):
        out = overlay_image(undist, unwarp, Minv, self.left_line.best_fit, self.right_line.best_fit)
        return out
    
    def image_pipeline(self, image):
        self.save_image(image, "org")
        # 1. undistort image
        undist = undistort_image(image, self.mtx, self.dist)
        self.save_image(undist, "undist")
        # 2. color and gradient threshold
        threshold = color_gradient_threshold(undist)
        self.save_image(threshold,"thres")
        # 3. perspective transform
        unwarp, M, Minv = unwarp_image(threshold, self.src, self.dst)
        self.save_image(unwarp, "unwarp")
        # 4. find lane lines
        self.find_lane_lines(unwarp)
        # 5. overlay lane lines on image
        out_img = self.overlay_lane_lines(undist, unwarp, Minv)
        self.save_image(out_img, "out")
        # increment the frame counter
        self.frame_number += 1
        return out_img


