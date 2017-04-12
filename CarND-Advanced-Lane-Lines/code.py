import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

### Functions to process images
## Camera calibration
def cam_cal(cal_images):
	# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
	objp = np.zeros((6*9,3), np.float32)
	new = np.mgrid[0:9,0:6]
	objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

	# Arrays to store object points and image points from all the images.
	objpoints = [] # 3d points in real world space
	imgpoints = [] # 2d points in image plane.

	# Step through the list and search for chessboard corners
	for fname in cal_images:
		img = cv2.imread(fname)
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		# Find the chessboard corners
		ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
	    	# If found, add object points, image points
		if ret == True:
			objpoints.append(objp)
			imgpoints.append(corners)

	return objpoints, imgpoints


def cam_undist(img, objpoints, imgpoints):
	# Use cv2.calibrateCamera() and cv2.undistort()
	undist = np.copy(img)  # Delete this line
	img_size = (undist.shape[1], undist.shape[0])
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
	undst = cv2.undistort(undist, mtx, dist, None, mtx)
	return undst


## Find gradients
# x/y gradient
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, abs_thresh=(0, 255)):
	# Apply the following steps to img
	# 1) Convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# 2) Take the derivative in x or y given orient = 'x' or 'y'
	if (orient=='x'):
	    s = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
	elif(orient=='y'):
	    s = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
	else:
	    s = gray
	# 3) Take the absolute value of the derivative or gradient
	s_abs = np.absolute(s)
	# 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
	s_scale = np.uint8(255*s_abs/np.max(s_abs))
	# 5) Create a mask of 1's where the scaled gradient magnitude 
	        # is > thresh_min and < thresh_max
	s_bin = np.zeros_like(s_scale)
	s_bin[(s_scale>=abs_thresh[0])&(s_scale<=abs_thresh[1])] = 1
	# 6) Return this mask as your binary_output image
	return s_bin
# magnitude gradient
def mag_sobel_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2) Take the gradient in x and y separately
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude 
    s_mag = np.sqrt(np.square(sx) + np.square(sy))
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    s_scale = np.uint8(255*s_mag/np.max(s_mag))
    # 5) Create a binary mask where mag thresholds are met
    s_bin = np.zeros_like(s_scale)
    s_bin[(s_scale>=mag_thresh[0])&(s_scale<=mag_thresh[1])]=1
    # 6) Return this mask as your binary_output image
    return s_bin
# direction gradient
def dir_sobel_thresh(img, sobel_kernel=3, dir_thresh=(0, np.pi/2)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2) Take the gradient in x and y separately
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    sx_abs = np.absolute(sx)
    sy_abs = np.absolute(sy)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    grad_dir = np.arctan2(sy_abs, sx_abs)
    # 5) Create a binary mask where direction thresholds are met
    s_bin = np.zeros_like(grad_dir)
    s_bin[(grad_dir>=dir_thresh[0]) & (grad_dir<=dir_thresh[1])]=1
    # 6) Return this mask as your binary_output image
    return s_bin
# color space threshlding - s color 
def hls_select(img, hls_thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    # 2) Apply a threshold to the S channel
    s = hls[:,:,2]
    s_bin = np.zeros_like(s)
    s_bin[(s>=hls_thresh[0])&(s<=hls_thresh[1])]=1
    # 3) Return a binary image of threshold result
    return s_bin

# change image perspective    
def warp(img, src, dst):
	img_size = (img.shape[1], img.shape[0])
	M = cv2.getPerspectiveTransform(src, dst)
	warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
	
	return warped

## pipeline to proces images for lane finding
def proc_img(img, objpoints, imgpoints, src, dst, ksize=3, abs_thresh=(0, 255), mag_thresh=(0, 255), dir_thresh=(0, np.pi/2), hls_thresh=(0, 255)):
	# undistort image based on camera calibration result matrices
	undistorted = cam_undist(img, objpoints, imgpoints)

	# apply each of the thresholding functions
	gradx = abs_sobel_thresh(undistorted, orient='x', sobel_kernel=ksize, abs_thresh=(20, 100))
	grady = abs_sobel_thresh(undistorted, orient='y', sobel_kernel=ksize, abs_thresh=(20, 100))
	mag_binary = mag_sobel_thresh(undistorted, sobel_kernel=ksize, mag_thresh=(30, 100))
	dir_binary = dir_sobel_thresh(undistorted, sobel_kernel=ksize, dir_thresh=(0.7, 1.3))
	hls_binary = hls_select(undistorted, hls_thresh=(160, 220))
	# combine thresholded images
	combined = np.zeros_like(dir_binary)
	combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | hls_binary==1] = 1
	# warp image
	warpped_comb = warp(combined, src, dst)

	return undistorted, combined, warpped_comb

### Functions to find lanes
# blind search of lines
def find_lines(image, window_width, window_height, margin):

	img_height = image.shape[0]
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = image.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	minpix = 50 # pixel threshold to decide whether there's a lane
	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []

	
	# First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
	# Take a histogram of the bottom quarter of the image
	histogram = np.sum(image[int(3*img_height/4):,:], axis=0)
	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	midpoint = int(histogram.shape[0]/2)
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint

	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base

	# Step through the windows one by one
	for window in range(int(img_height/window_height)):
	    # Identify window boundaries in x and y (and right and left)
	    win_y_low = img_height - (window+1)*window_height
	    win_y_high = img_height - window*window_height
	    win_xleft_low = leftx_current - margin
	    win_xleft_high = leftx_current + margin
	    win_xright_low = rightx_current - margin
	    win_xright_high = rightx_current + margin
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

	return leftx, lefty, rightx, righty

# update lines based on previous fitting results
def update_lines(image, left_fit, right_fit, margin):

	nonzero = image.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
	right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  
	
	# Again, extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]
	
	return leftx, lefty, rightx, righty

# fit lines
def fit_lines(leftx, lefty, rightx, righty):

	# Fit a second order polynomial to pixel positions in each lane line
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)

	return left_fit, right_fit

# measure curvature, lane locations
def meas_param(ploty, left_fit, right_fit, xm_per_pix, ym_per_pix):

	left_fitx_pix = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx_pix = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	# based on fitted lines, find the physical curvature
	left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx_pix*xm_per_pix, 2)
	right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx_pix*xm_per_pix, 2)
	y_eval = np.max(ploty)
	# Calculate the new radii of curvature
	left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
	right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

	return left_curverad, right_curverad, left_fitx_pix, right_fitx_pix

### Post processing functions
# unwarp image
def unwarp_lane_img(img_warpped, Minv, ploty, left_fitx_pix, right_fitx_pix):
	# Create an image to draw the lines on
	warp_zero = np.zeros_like(img_warpped).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
	
	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx_pix, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx_pix, ploty])))])
	pts = np.hstack((pts_left, pts_right))
	
	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
	
	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	img_unwarpped = cv2.warpPerspective(color_warp, Minv, (img_warpped.shape[1], img_warpped.shape[0])) 

	return img_unwarpped

# Define a class to receive the characteristics of each line detection
class Lines():
	def __init__(self):
		#was the line detected in the last iteration?
		self.last_det = False  
		#pixel threshold for detecting a lane
		self.lane_det_thresh = 100 
		#missed lane count
		self.miss_det_cnt = 0
		# to reset if missed cont reaches this value
		self.miss_det_target = 12 # roughpy half a second for 25 frame/sec case
		#need to reset the line?
		self.reset = True # it is in reset mode when initialized  
		#number of averages before update
		self.n_avg = 3
		#detected lines counts
		self.det_lines = 0
		#average x values of the fitted line over the last n iterations
		self.avg_left_fitx = None     
		self.avg_right_fitx = None     
		#polynomial coefficients averaged over the last n iterations
		self.avg_left_fit = None  
		self.avg_right_fit = None  
		#polynomial coefficients for the most recent fit
		self.recent_left_fit = None 
		self.recent_right_fit = None 
		#accumulated fit over # of average frames
		self.acc_left_fit=[]
		self.acc_right_fit=[]
		#radius of curvature of the line in some units
		self.left_curverad = None 
		self.right_curverad = None 
		#distance in meters of vehicle center from the lane center
		self.cntr_offset = None 
		#x values for detected line pixels
		self.left_allx = None  
		self.right_allx = None  
		#y values for detected line pixels
		self.left_ally = None
		self.right_ally = None


# Make a list of calibration images
images = glob.glob('./camera_cal/calibration*.jpg')
objpoints, imgpoints = cam_cal(images)

# just read one video frame to get image width and height
video_path = 'project_video.mp4'
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
img_width = frame.shape[1]
img_height = frame.shape[0]

# used for lane line fitting
ploty = np.linspace(0, img_height-1, img_height)
# kernel size for thresholding
ksize = 3
# window settings for finding lanes
window_width = 50 
window_height = 80 # Break image into 9 vertical layers since image height is 720
margin = 30 # How much to slide left and right for searching

# source and dest rectangular for the perspective transform
# clockwise from the top-right corner
src = np.float32(
	[[680, 450],
	 [1120, 720],
	 [200, 720],
	 [590, 450]])

dst = np.float32(
	[[850, 0],
	 [900, 720],
	 [300, 720],
	 [300, 0]])

# meter to pixel mapping
ym_per_pix = 30/(dst[1,1]-dst[0,1])
xm_per_pix = 3.7/(dst[1,0]-dst[2,0])

# inverse transform matrix
Minv = cv2.getPerspectiveTransform(dst, src)

# create a Lines object to keep track of the extracted pair of lines
lane_lines = Lines()

# for video recording
from moviepy.editor import VideoFileClip

# video processing pipeline
def proc_video(image):
	# process image
	img_undst, img_bin, warpped_bin = proc_img(image, objpoints, imgpoints, src, dst, ksize, abs_thresh=(20, 100), mag_thresh=(30, 100), dir_thresh=(0.7, 1.3), hls_thresh=(160, 220)) 	

	# depending on the line status, apply different functions	
	if (lane_lines.reset): # in reset status, find the lane and update parameters for the first time
		lane_lines.left_allx, lane_lines.left_ally, lane_lines.right_allx, lane_lines.right_ally = find_lines(warpped_bin, window_width, window_height, margin) # find lanes
		if ((len(lane_lines.left_allx) >= lane_lines.lane_det_thresh) & (len(lane_lines.right_allx) >= lane_lines.lane_det_thresh)): # if succesfully find lanes
			lane_lines.reset = False
			lane_lines.last_det = True
			lane_lines.det_lines = lane_lines.det_lines + 1 # used to find the average lane coefficients
			# fit lines
			lane_lines.recent_left_fit, lane_lines.recent_right_fit = fit_lines(lane_lines.left_allx, lane_lines.left_ally, lane_lines.right_allx, lane_lines.right_ally)
			# compute line parameter
			lane_lines.left_curvrad, lane_lines.right_curvrad, lane_lines.avg_left_fitx, lane_lines.avg_right_fitx = meas_param(ploty, lane_lines.recent_left_fit, lane_lines.recent_right_fit, xm_per_pix, ym_per_pix)
			lane_cntr_pix = (lane_lines.avg_left_fitx[-1] + lane_lines.avg_right_fitx[-1])/2
			lane_lines.cntr_offset = (img_width/2 - lane_cntr_pix)*xm_per_pix
			# append current fit to an moving average array for filtering
			lane_lines.acc_left_fit.append(lane_lines.recent_left_fit)
			lane_lines.acc_right_fit.append(lane_lines.recent_right_fit)

	else: # in continous update status
		if (lane_lines.det_lines == lane_lines.n_avg): # if reach the max of the moving average window
			# find the average fitting coefficients
			acc_left_fit_arr = np.array(lane_lines.acc_left_fit)
			acc_right_fit_arr = np.array(lane_lines.acc_right_fit)

			cur_avg_left_fit = np.sum(acc_left_fit_arr, axis=0)/lane_lines.n_avg
			cur_avg_right_fit = np.sum(acc_right_fit_arr, axis=0)/lane_lines.n_avg
			
			lane_lines.avg_left_fit = cur_avg_left_fit
			lane_lines.avg_right_fit = cur_avg_right_fit
			
			# compute the line parameters using the average fititng coefficients
			lane_lines.left_curvrad, lane_lines.right_curvrad, lane_lines.avg_left_fitx, lane_lines.avg_right_fitx = meas_param(ploty, lane_lines.avg_left_fit, lane_lines.avg_right_fit, xm_per_pix, ym_per_pix)
			lane_cntr_pix = (lane_lines.avg_left_fitx[-1] + lane_lines.avg_right_fitx[-1])/2
			lane_lines.cntr_offset = (img_width/2 - lane_cntr_pix)*xm_per_pix
			
			# reset the moving average filter
			lane_lines.det_lines = 0
			lane_lines.acc_left_fit = []
			lane_lines.acc_right_fit = []

		# we only need to call the line update to avoid blind search for lines
		lane_lines.left_allx, lane_lines.left_ally, lane_lines.right_allx, lane_lines.right_ally = update_lines(warpped_bin, lane_lines.recent_left_fit, lane_lines.recent_right_fit, margin)
		if ((len(lane_lines.left_allx) >= lane_lines.lane_det_thresh) & (len(lane_lines.right_allx) >= lane_lines.lane_det_thresh)): # if lines are found (i.e. meet threshold)
			if (lane_lines.last_det): # if consecutively detect lines, update the fit and moving average filter elements, else do nothing (reuse the previous finding)
				lane_lines.det_lines = lane_lines.det_lines + 1
				lane_lines.recent_left_fit, lane_lines.recent_right_fit = fit_lines(lane_lines.left_allx, lane_lines.left_ally, lane_lines.right_allx, lane_lines.right_ally)
				lane_lines.acc_left_fit.append(lane_lines.recent_left_fit)
				lane_lines.acc_right_fit.append(lane_lines.recent_right_fit)
				lane_lines.last_det = True
		else: # didn't detec line this time, reset
			lane_lines.lane_det_lines = 0
			lane_lines.last_det = False
			lane_lines.miss_det_cnt = lane_lines.miss_det_cnt + 1
			if (lane_lines.miss_det_cnt >= lane_lines.miss_det_target): # if didn't detect line for a couple of times, reset line fit and enter blind search mode
				self.miss_det_cnt = 0
				self.reset = True
				self.det_lines = 0
				self.acc_left_fit=[]
				self.acc_right_fit=[]

	# combine color and unwarpped images
	img_unwarp = unwarp_lane_img(warpped_bin, Minv, ploty, lane_lines.avg_left_fitx, lane_lines.avg_right_fitx)
	img_weighted = cv2.addWeighted(img_undst, 1, img_unwarp, 0.3, 0)
	
	# annotate text on the video
	curv_text = 'radius of the curvature is {0:.1f}m'.format((lane_lines.left_curvrad+lane_lines.right_curvrad)/2) 

	if (lane_lines.cntr_offset<0):
		cntr_text = 'vehicle is {0:.2f}m left to the center'.format(-lane_lines.cntr_offset)
	else:
		cntr_text = 'vehicle is {0:.2f}m right to the center'.format(lane_lines.cntr_offset)

	overlay = cv2.putText(img_weighted, curv_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
	overlay = cv2.putText(overlay, cntr_text, (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
	return overlay

# output and save video
video_out = 'project_output.mp4'
clip = VideoFileClip(video_path)
proc_clip = clip.fl_image(proc_video)
proc_clip.write_videofile(video_out, audio=False)
