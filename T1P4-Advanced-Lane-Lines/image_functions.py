import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

def get_src_dst_coordinates():
	# source points
	top_left = (588, 454)
	top_right = (692, 454)
	bottom_left = (203, 720)
	bottom_right = (1105, 720)

	src = np.float32([top_right,
	                  bottom_right, 
	                  bottom_left, 
	                  top_left])

	# desired destination
	dst_top_left = (200, 0)
	dst_top_right = (1000, 0)
	dst_bottom_right = (1000, 720)
	dst_bottom_left = (200, 720)

	dst = np.float32([dst_top_right,
	                  dst_bottom_right, 
	                  dst_bottom_left, 
	                  dst_top_left])
	return src, dst

# get the color mask
def color_mask(img, low, high):
    return cv2.inRange(img, low, high)

# undistort image
def undistort_image(img, mtx, dist):
    return cv2.undistort(img, mtx, dist, None, mtx)

# unwarp image
def unwarp_image(img, src, dst):
    img_height, img_width = img.shape[:2]
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, (img_width, img_height), flags=cv2.INTER_LINEAR)
    return warped, M, Minv

# calibrate camera
def calibrate_camera(images, nx=9, ny=6):

    objpoints = [] #3D object points
    imgpoints = [] #2D image points

    # prepare object points
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2) 

    for idx, fname in enumerate(images):

        # read image, BGR format in cv2
        img = cv2.imread(fname)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, add to the list of imgpoints and objpoints
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            
    return cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# gets rgb images 
def get_samples_rgb(image_filename_list):
    out_images = []
    for image_filename in image_filename_list:
        img = cv2.imread(image_filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out_images.append(img)
    return out_images

# get unwarp images for given rgb images
def get_samples_unwarp(rgb_images, mtx, dist, src, dst):
    out_unwarp = []
    out_undist = []
    out_M = []
    out_Minv = []
    for image in rgb_images:
        undist_img = undistort_image(image, mtx, dist)
        unwarp_img, M, Minv = unwarp_image(undist_img, src, dst)
        out_unwarp.append(unwarp_img)
        out_undist.append(undist_img)
        out_M.append(M)
        out_Minv.append(Minv)
    return out_unwarp, out_undist, out_M, out_Minv

def yellow_lines(img):
    image = np.copy(img)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    #yellow_hsv_low  = np.array([ 15,  100,  100])
    #yellow_hsv_high = np.array([ 35, 255, 255])
    yellow_hsv_low  = np.array([ 10,  70,  100])
    yellow_hsv_high = np.array([ 35, 255, 255])
    mask = color_mask(hsv, yellow_hsv_low, yellow_hsv_high)
    result = cv2.bitwise_and(image, image, mask= mask)
    return result, mask

def white_lines(img):
    image = np.copy(img)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    #white_hsv_low  = np.array([ 0,   0,   205])
    #white_hsv_high = np.array([ 255,  30, 255])
    white_hsv_low  = np.array([ 0,   0,   180])
    white_hsv_high = np.array([ 180,  25, 255])
    mask = color_mask(hsv, white_hsv_low, white_hsv_high)
    result = cv2.bitwise_and(image, image, mask= mask)
    return result, mask

def build_lookup_table(inv_gamma):
    return np.array([((i / 255) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)

def gamma_corrected(img, gamma):
    if gamma == 0:
        return np.zeros(img.shape)
    lookup_table = build_lookup_table(1 / gamma)
    return cv2.LUT(img, lookup_table)

def abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(0, 255)):
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
        
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output

def apply_filter(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    out = cv2.GaussianBlur(gray,(5,5),0)
    #kernel = np.ones((5,5),np.float32)/25
    #out = gamma_corrected(gray, 1.5)
    #out = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    #out = cv2.medianBlur(gray,3)
    #out = cv2.filter2D(gray,-1,kernel)
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #out = clahe.apply(gray)
    #out = cv2.equalizeHist(gray)
    #out = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #out = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,23,4)
    return out

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    absgraddir = np.arctan2(abs_sobely, abs_sobelx)
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return binary_output

def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def find_window_centroids(warped, window_width, window_height, margin):
    
    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions
    
    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 
    
    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(warped[int(3*warped.shape[0]/4):,:int(warped.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    r_sum = np.sum(warped[int(3*warped.shape[0]/4):,int(warped.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(warped.shape[1]/2)
    
    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))
    
    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(warped.shape[0]/window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),:], axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width/2
        l_min_index = int(max(l_center+offset-margin,0))
        l_max_index = int(min(l_center+offset+margin,warped.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center+offset-margin,0))
        r_max_index = int(min(r_center+offset+margin,warped.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
        # Add what we found for that layer
        window_centroids.append((l_center,r_center))
        
    return window_centroids

def add_overlay(warped, undist, Minv, left_fitx, right_fitx, ploty):

	# Create an image to draw the lines on
	warp_zero = np.zeros_like(warped).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	cv2.drawContours(color_warp, np.int_([pts_left]), -1, (255,0,0), thickness=30)
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	cv2.drawContours(color_warp, np.int_([pts_right]), -1, (0,0,255), thickness=30)
	pts = np.hstack((pts_left, pts_right))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))

	# Combine the result with the original image
	result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

	return result, color_warp, newwarp

def add_selection_window(binary_warped, nonzerox, nonzeroy, left_lane_inds, right_lane_inds, left_fitx, right_fitx, ploty, margin):
	# Create an image to draw on and an image to show the selection window
	out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
	window_img = np.zeros_like(out_img)
	# Color in left and right line pixels
	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

	# Generate a polygon to illustrate the search window area
	# And recast the x and y points into usable format for cv2.fillPoly()
	left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
	left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
	left_line_pts = np.hstack((left_line_window1, left_line_window2))
	right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
	right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
	right_line_pts = np.hstack((right_line_window1, right_line_window2))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
	cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))

	result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
	return result

def sliding_windows(binary_warped, out_img, nonzerox, nonzeroy, leftx_current, rightx_current, margin=100, minpix=50, nwindows=9):
	# Set height of windows
	window_height = np.int(binary_warped.shape[0]/nwindows)
	left_lane_inds = []
	right_lane_inds = []

	# Step through the windows one by one
	for window in range(nwindows):
	    # Identify window boundaries in x and y (and right and left)
	    win_y_low = binary_warped.shape[0] - (window+1)*window_height
	    win_y_high = binary_warped.shape[0] - window*window_height
	    win_xleft_low = leftx_current - margin
	    win_xleft_high = leftx_current + margin
	    win_xright_low = rightx_current - margin
	    win_xright_high = rightx_current + margin
	    # Draw the windows on the visualization image
	    cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 3) 
	    cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 3) 
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

	return left_lane_inds, right_lane_inds, leftx_current, rightx_current

# plot and save images
def plot_images(rows, cols, images_to_plot, titles_to_plot, out_file, cmap=None):
	fig = plt.figure(figsize=(6 * cols, 4 * rows))
	for idx in range(len(images_to_plot)):
		plt.subplot(rows, cols, idx+1)
		if cmap is not None and cmap[idx] is not None:
			plt.imshow(images_to_plot[idx], cmap=cmap[idx])
		else:
			plt.imshow(images_to_plot[idx])
		plt.title(titles_to_plot[idx])
	plt.tight_layout()
	fig.savefig('./output/' + out_file)
	plt.show()

# color gradient threshold
def color_gradient_threshold(image):
	# Choose a Sobel kernel size
	ksize = 31 # Choose a larger odd number to smooth gradient measurements

	# Apply each of the thresholding functions
	gradx = abs_sobel_thresh(apply_filter(image), orient='x', sobel_kernel=ksize, thresh=(50, 150))
	yellow_lines_binary, yellow_mask = yellow_lines(image)
	white_lines_binary, white_mask = white_lines(image)
	lane_mask = cv2.bitwise_or(yellow_mask, white_mask)
	lane_mask = lane_mask/255.0

	combined = np.zeros_like(gradx)
	combined[((gradx == 1) | (lane_mask == 1))] = 1

	return combined

# check if the left/right fits are parallel within a given threshold
def is_parallel(left_fit, right_fit, threshold = (0.01, 0.5)):
	first_coefficient_diff = np.abs(left_fit[0] - right_fit[0])
	second_coefficient_diff = np.abs(left_fit[1] - right_fit[1])
	#print("first coefficient diff : {} second coefficient diff : {}".format(first_coefficient_diff, second_coefficient_diff))
	return first_coefficient_diff < threshold[0] and second_coefficient_diff < threshold[1]

# get distance between fits
def is_distance_in_range(first_fitx, second_fitx, threshold = (600,900)):
	dist = np.abs(first_fitx[-1] - second_fitx[-1])
	xm_per_pix = 3.7/700 # To be cleaned up
	#print("fit pos : {:.2f} px {:.2f} px".format(first_fitx[-1], second_fitx[-1]))
	#print("fit distance : {:.2f} px {:.2f} m".format(dist, dist * xm_per_pix))
	return dist > threshold[0] and dist < threshold[1]

# removal of outlier samples
def outlier_removal(x, y, q=5):
	x = np.array(x)
	y = np.array(y)

	lower_bound = np.percentile(x, q)
	upper_bound = np.percentile(x, 100 - q)
	selection = (x >= lower_bound) & (x <= upper_bound)
	return x[selection], y[selection]

# sliding window implementation
# for a given image, returns a list of points on the left/right lines
def sliding_windows2(binary_warped):
	# Take a histogram of the bottom half of the image
	histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	midpoint = np.int(histogram.shape[0]/2)
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint
	# Choose the number of sliding windows
	nwindows = 9
	# Set height of windows
	window_height = np.int(binary_warped.shape[0]/nwindows)
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base
	# Set the width of the windows +/- margin
	margin = 100
	# Set minimum number of pixels found to recenter window
	minpix = 50
	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []
	# initialize sliding windows
	sliding_windows_list = []
	# Step through the windows one by one
	for window in range(nwindows):
		# Identify window boundaries in x and y (and right and left)
		win_y_low = binary_warped.shape[0] - (window+1)*window_height
		win_y_high = binary_warped.shape[0] - window*window_height
		win_xleft_low = leftx_current - margin
		win_xleft_high = leftx_current + margin
		win_xright_low = rightx_current - margin
		win_xright_high = rightx_current + margin
		# save the sliding window
		left_window = [(win_xleft_low, win_y_low), (win_xleft_high, win_y_high)]
		right_window = [(win_xright_low, win_y_low), (win_xright_high, win_y_high)]
		sliding_windows_list.append((left_window, right_window))
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
	# Return
	return sliding_windows_list, leftx, lefty, rightx, righty

# from existing polyfit
def find_lane_lines_from_existing(binary_warped, left_fit, right_fit):
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	margin = 100
	left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
	right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

	# Again, extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]

	return leftx, lefty, rightx, righty

def sliding_windows_image(binary_warped, sliding_windows_list, color=(0,255,0), border_thickness=3):
	# Create an output image to draw on and  visualize the result
	out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
	for window in sliding_windows_list:
		win_left = window[0]
		win_right = window[1]
		# Draw the windows on the visualization image
		cv2.rectangle(out_img, win_left[0], win_left[1], color, border_thickness)
		cv2.rectangle(out_img, win_right[0], win_right[1], color, border_thickness)
	return out_img

def overlay_image(undist, binary_warped, Minv, left_fit, right_fit):

	if left_fit is None or right_fit is None:
		return undist

	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	# Create an image to draw the lines on
	warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	cv2.drawContours(color_warp, np.int_([pts_left]), -1, (255,0,0), thickness=30)
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	cv2.drawContours(color_warp, np.int_([pts_right]), -1, (0,0,255), thickness=30)
	pts = np.hstack((pts_left, pts_right))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0])) 
	# Combine the result with the original image
	result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

	# Define conversions in x and y from pixels space to meters
	y_eval = np.max(ploty)

	img_center = binary_warped.shape[1] / 2
	lane_center = (left_fit[-1] + right_fit[-1]) / 2
	diff = (lane_center - img_center) * xm_per_pix
	left_or_right = 'left' if diff < 0 else 'right'

	center_fit = (left_fit + right_fit) / 2
	center_curverad = ((1 + (2 * center_fit[0] * y_eval * ym_per_pix + center_fit[1])**2)**1.5) / np.absolute(2 * center_fit[0])

	# Adding Curvature and Distance information
	result = cv2.putText(
		result,
		'Curvature : {:.2f} m'.format(center_curverad),(50,50), 
		cv2.FONT_HERSHEY_SIMPLEX, 
		1,
		(255,255,255),
		2,
		cv2.LINE_AA)
	result = cv2.putText(
		result,
		'Distance : {:.2f} m {} of the center'.format(abs(diff), left_or_right),
		(50,100), 
		cv2.FONT_HERSHEY_SIMPLEX, 
		1,
		(255,255,255),
		2,
		cv2.LINE_AA)
	return result

# return a random prefix with numbers and alphabets of N length, used as a session id for output files
def random_prefix(N=5):
	import random, string
	return ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(N))
