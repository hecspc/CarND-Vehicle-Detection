import numpy as np
import cv2

class LaneFinder:
    def __init__(self, l_line, r_line, camera_calibration, src_points = None, dst_points = None):
        self.l_line = l_line
        self.r_line = r_line
        self.camera_calibration = camera_calibration
        self.src_points = src_points
        self.dst_points = dst_points

    def pipeline(self, img):
        l_line, r_line = self.l_line, self.r_line
        if (l_line is None or r_line is None ):
            raise NotImplementedError
        # Undistort
        img_undistort = self.undistort(img)

        img_unwarp, M, Minv = self.unwarp(img_undistort)

        # HLS L-channel Threshold (using default parameters)
        img_LThresh = self.hls_lthresh(img_unwarp)

        # Lab B-channel Threshold (using default parameters)
        img_BThresh = self.lab_bthresh(img_unwarp)

        # Combine HLS and Lab B channel thresholds
        img_bin = np.zeros_like(img_BThresh)
        img_bin[(img_LThresh == 1) | (img_BThresh == 1)] = 1

         # if both left and right lines were detected last frame, use polyfit_using_prev_fit, otherwise use sliding window
        if not l_line.detected or not r_line.detected:
            l_fit, r_fit, l_lane_inds, r_lane_inds, _ = self.sliding_window_search(img_bin)
        else:
            l_fit, r_fit, l_lane_inds, r_lane_inds = self.polyfit_using_prev_fit(img_bin, l_line.best_fit, r_line.best_fit)

        # invalidate both fits if the difference in their x-intercepts isn't around 350 px (+/- 100 px)
        if l_fit is not None and r_fit is not None:
            # calculate x-intercept (bottom of image, x=image_height) for fits
            h = img.shape[0]
            l_fit_x_int = l_fit[0]*h**2 + l_fit[1]*h + l_fit[2]
            r_fit_x_int = r_fit[0]*h**2 + r_fit[1]*h + r_fit[2]
            x_int_diff = abs(r_fit_x_int-l_fit_x_int)
            if abs(350 - x_int_diff) > 100:
                l_fit = None
                r_fit = None

        l_line.add_fit(l_fit, l_lane_inds)
        r_line.add_fit(r_fit, r_lane_inds)

        # draw the current best fit if it exists
        if l_line.best_fit is not None and r_line.best_fit is not None:
            img_out1 = self.draw_lane(img, img_bin, l_line.best_fit, r_line.best_fit, Minv)
            rad_l, rad_r, d_center = self.calc_curv_rad_and_center_dist(img_bin, l_line.best_fit, r_line.best_fit,
                                                                   l_lane_inds, r_lane_inds)
            img_out = self.draw_data(img_out1, (rad_l+rad_r)/2, d_center)
        else:
            img_out = img

        return img_out


    def undistort(self, img):
        mtx, dist = self.camera_calibration["mtx"], self.camera_calibration["dist"]
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        return undist

    def unwarp(self,img):
        h,w = img.shape[:2]
        if self.src_points == None:
            self.src_points = np.float32([(575,464),
                              (707,464),
                              (258,682),
                              (1049,682)])

        if self.dst_points == None:
            self.dst_points = np.float32([(450,0),
                              (w-450,0),
                              (450,h),
                              (w-450,h)])

        # use cv2.getPerspectiveTransform() to get M, the transform matrix, and Minv, the inverse
        M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        Minv = cv2.getPerspectiveTransform(self.dst_points, self.src_points)
        # use cv2.warpPerspective() to warp your image to a top-down view
        warped = cv2.warpPerspective(img, M, (w,h), flags=cv2.INTER_LINEAR)
        return warped, M, Minv

    def hls_lthresh(self, img, thresh=(220, 255)):
        # 1) Convert to HLS color space
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        hls_l = hls[:,:,1]
        hls_l = hls_l*(255/np.max(hls_l))
        # 2) Apply a threshold to the L channel
        binary_output = np.zeros_like(hls_l)
        binary_output[(hls_l > thresh[0]) & (hls_l <= thresh[1])] = 1
        # 3) Return a binary image of threshold result
        return binary_output

    def lab_bthresh(self, img, thresh=(190,255)):
        # 1) Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
        lab_b = lab[:,:,2]
        # don't normalize if there are no yellows in the image
        if np.max(lab_b) > 175:
            lab_b = lab_b*(255/np.max(lab_b))
        # 2) Apply a threshold to the L channel
        binary_output = np.zeros_like(lab_b)
        binary_output[((lab_b > thresh[0]) & (lab_b <= thresh[1]))] = 1
        # 3) Return a binary image of threshold result
        return binary_output

    def sliding_window_search(self, img):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        quarter_point = np.int(midpoint//2)
        # Previously the left/right base was the max of the left/right half of the histogram
        # this changes it so that only a quarter of the histogram (directly to the left/right) is considered
        leftx_base = np.argmax(histogram[quarter_point:midpoint]) + quarter_point
        rightx_base = np.argmax(histogram[midpoint:(midpoint+quarter_point)]) + midpoint

        # Choose the number of sliding windows
        nwindows = 10
        # Set height of windows
        window_height = np.int(img.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 80
        # Set minimum number of pixels found to recenter window
        minpix = 40
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        # Rectangle data for visualization
        rectangle_data = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = img.shape[0] - (window+1)*window_height
            win_y_high = img.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            rectangle_data.append((win_y_low, win_y_high, win_xleft_low, win_xleft_high, win_xright_low, win_xright_high))
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        left_fit, right_fit = (None, None)
        # Fit a second order polynomial to each
        if len(leftx) != 0:
            left_fit = np.polyfit(lefty, leftx, 2)
        if len(rightx) != 0:
            right_fit = np.polyfit(righty, rightx, 2)

        visualization_data = (rectangle_data, histogram)

        return left_fit, right_fit, left_lane_inds, right_lane_inds, visualization_data

    def polyfit_using_prev_fit(self, binary_warped, left_fit_prev, right_fit_prev):
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 80
        left_lane_inds = ((nonzerox > (left_fit_prev[0]*(nonzeroy**2) + left_fit_prev[1]*nonzeroy + left_fit_prev[2] - margin)) &
                          (nonzerox < (left_fit_prev[0]*(nonzeroy**2) + left_fit_prev[1]*nonzeroy + left_fit_prev[2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit_prev[0]*(nonzeroy**2) + right_fit_prev[1]*nonzeroy + right_fit_prev[2] - margin)) &
                           (nonzerox < (right_fit_prev[0]*(nonzeroy**2) + right_fit_prev[1]*nonzeroy + right_fit_prev[2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        left_fit_new, right_fit_new = (None, None)
        if len(leftx) != 0:
            # Fit a second order polynomial to each
            left_fit_new = np.polyfit(lefty, leftx, 2)
        if len(rightx) != 0:
            right_fit_new = np.polyfit(righty, rightx, 2)
        return left_fit_new, right_fit_new, left_lane_inds, right_lane_inds


    def draw_lane(self, original_img, binary_img, l_fit, r_fit, Minv):
        new_img = np.copy(original_img)
        if l_fit is None or r_fit is None:
            return original_img
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(binary_img).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        h,w = binary_img.shape
        ploty = np.linspace(0, h-1, num=h)# to cover same y-range as image
        left_fitx = l_fit[0]*ploty**2 + l_fit[1]*ploty + l_fit[2]
        right_fitx = r_fit[0]*ploty**2 + r_fit[1]*ploty + r_fit[2]

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255,0,255), thickness=15)
        cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0,255,255), thickness=15)

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (w, h))
        # Combine the result with the original image
        result = cv2.addWeighted(new_img, 1, newwarp, 0.5, 0)
        return result


    def calc_curv_rad_and_center_dist(self, bin_img, l_fit, r_fit, l_lane_inds, r_lane_inds):
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 3.048/100 # meters per pixel in y dimension, lane line is 10 ft = 3.048 meters
        xm_per_pix = 3.7/378 # meters per pixel in x dimension, lane width is 12 ft = 3.7 meters
        left_curverad, right_curverad, center_dist = (0, 0, 0)
        # Define y-value where we want radius of curvature
        # I'll choose the maximum y-value, corresponding to the bottom of the image
        h = bin_img.shape[0]
        ploty = np.linspace(0, h-1, h)
        y_eval = np.max(ploty)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = bin_img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Again, extract left and right line pixel positions
        leftx = nonzerox[l_lane_inds]
        lefty = nonzeroy[l_lane_inds]
        rightx = nonzerox[r_lane_inds]
        righty = nonzeroy[r_lane_inds]

        if len(leftx) != 0 and len(rightx) != 0:
            # Fit new polynomials to x,y in world space
            left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
            right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
            # Calculate the new radii of curvature
            left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
            right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
            # Now our radius of curvature is in meters

        # Distance from center is image x midpoint - mean of l_fit and r_fit intercepts
        if r_fit is not None and l_fit is not None:
            car_position = bin_img.shape[1]/2
            l_fit_x_int = l_fit[0]*h**2 + l_fit[1]*h + l_fit[2]
            r_fit_x_int = r_fit[0]*h**2 + r_fit[1]*h + r_fit[2]
            lane_center_position = (r_fit_x_int + l_fit_x_int) /2
            center_dist = (car_position - lane_center_position) * xm_per_pix
        return left_curverad, right_curverad, center_dist

    def draw_data(self, original_img, curv_rad, center_dist):
        new_img = np.copy(original_img)
        h = new_img.shape[0]
        font = cv2.FONT_HERSHEY_DUPLEX
        text = 'Radius of Curvature = ' + '{:04.2f}'.format(curv_rad) + 'm'
        cv2.putText(new_img, text, (40,70), font, 1.5, (255,204,0), 2, cv2.LINE_AA)
        direction = ''
        if center_dist > 0:
            direction = 'right'
        elif center_dist < 0:
            direction = 'left'
        abs_center_dist = abs(center_dist)
        text = 'Vehicle is {:03.2f}'.format(abs_center_dist) + 'm ' + direction + ' of center'
        cv2.putText(new_img, text, (40,120), font, 1.5, (255,204,0), 2, cv2.LINE_AA)
        return new_img
