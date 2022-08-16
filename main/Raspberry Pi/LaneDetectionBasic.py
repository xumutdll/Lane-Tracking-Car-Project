import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob


def camera_calibration(cal_path, debug=False):
    '''
    this routine performs camera calibration
    it returns `mtx` and `dist` needed to
    undistort images taken from this camera
    '''
    # list all calibration images paths
    cal_images_names = glob.glob(cal_path)

    # chessboard-specific parameters
    nx = 9
    ny = 6

    # code below is based on classroom example
    objpoints = [] # 3D points
    imgpoints = [] # 2D points

    # (x,y,z): (0,0,0), (1,0,0), etc
    objp = np.zeros((nx * ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) # x, y coordinates, z stays 0

    for fname in cal_images_names:
        # read in image
        img = cv2.imread(fname)
        img = cv2.resize(img, (320, 240), interpolation= cv2.INTER_LINEAR)
        
        # convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # in case chessboard was found successfully
        # it skips 3 images that do not show full chessboard (1, 4 and 5)
        if ret == True:
            # image points will be different for each calibration image
            imgpoints.append(corners)
            # object points are the same for all calibration images
            objpoints.append(objp)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            if debug:
                plt.figure(figsize=(15,10))
                plt.imshow(img)

    # calibration parameters calculation
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, 
                                                       imgpoints, 
                                                       gray.shape[::-1], 
                                                       None, None)

    # will only use `mtx` and `dist` in this project, hence return
    return mtx, dist


def gaussian_blur(image, kernel=5):
    '''
    this routine applies blur to reduce noise in images
    '''
    blurred = cv2.GaussianBlur(image, (kernel,kernel), 0)
    return blurred


def detect_edges(frame):
    # filter for blue lane lines
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([80, 80, 0])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # # gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # # # apply gradient threshold on the horizontal gradient
    # # sx_binary = abs_sobel_thresh(gray, 'x', 12, 200)
    
    # # # apply gradient direction threshold so that only edges closer to vertical are detected.
    # # dir_binary = dir_threshold(gray, thresh=(np.pi/36, np.pi/2))
    
    # # # combine the gradient and direction thresholds.
    # # combined_condition = ((sx_binary == 1) & (dir_binary == 1))

    # # # filter for blue lane lines
    # # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # # lower_blue = np.array([90, 80, 0])
    # # upper_blue = np.array([140, 255, 255])
    # # blue_condition = cv2.inRange(hsv, lower_blue, upper_blue)
    # # bool_blue_condition = (blue_condition == 255)
    
    # # # combine all the thresholds
    # # color_combined = np.zeros_like(gray)
    # # color_combined[(bool_blue_condition & combined_condition)] = 1

    # # # detect edges
    edges = cv2.Canny(mask, 200, 400)

    return edges


# # def abs_sobel_thresh(gray, orient='x', thresh_min=0, thresh_max=255):
# #     if orient == 'x':
# #         sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
# #     else:
# #         sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
# #     abs_sobel = np.absolute(sobel)
# #     max_value = np.max(abs_sobel)
# #     binary_output = np.uint8(255*abs_sobel/max_value)
# #     threshold_mask = np.zeros_like(binary_output)
# #     threshold_mask[(binary_output >= thresh_min) & (binary_output <= thresh_max)] = 1
    
# #     return threshold_mask


# # def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi/2)):
# #     # Take the gradient in x and y separately
# #     sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
# #     sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
# #     # 3) Take the absolute value of the x and y gradients
# #     abs_sobel_x = np.absolute(sobel_x)
# #     abs_sobel_y = np.absolute(sobel_y)
# #     # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
# #     direction = np.arctan2(abs_sobel_y,abs_sobel_x)
# #     direction = np.absolute(direction)
# #     # 5) Create a binary mask where direction thresholds are met
# #     mask = np.zeros_like(direction)
# #     mask[(direction >= thresh[0]) & (direction <= thresh[1])] = 1
# #     # 6) Return this mask as your binary_output image
    
# #     return mask



def region_of_interest(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)

    # only focus bottom half of the screen
    polygon = np.array([[
        (0, height * 1 / 2),
        (width, height * 1 / 2),
        (width, height),
        (0, height),
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)
    return cropped_edges



def detect_line_segments(cropped_edges):
    # tuning min_threshold, minLineLength, maxLineGap is a trial and error process by hand
    rho = 1  # distance precision in pixel, i.e. 1 pixel
    angle = np.pi / 180  # angular precision in radian, i.e. 1 degree
    min_threshold = 10  # minimal of votes
    line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold, 
                                   np.array([]), minLineLength=8, maxLineGap=4)
    
    # rho = pixel cinsinden mesafe hassasiyeti
    # angle = radyan cinsinden derece alır, açısal hassasiyeti belirler.
    # min_threshold is the number of votes needed to be considered a line segment
    #   If a line has more votes, Hough Transform considers them to be more likely to have detected a line segment
    # minLineLength is the minimum length of the line segment in PIXELS. 
    #   Hough Transform won’t return any line segments shorter than this minimum length.
    # maxLineGap is the maximum in pixels that two line segments 
    #   that can be separated and still be considered as a single line segment.
    
    return line_segments





def average_slope_intercept(frame, line_segments):
    """
    This function combines line segments into one or two lane lines
    If all line slopes are < 0: then we only have detected left lane
    If all line slopes are > 0: then we only have detected right lane
    """
    lane_lines = []
    if line_segments is None:
        # logging.info('No line_segment segments detected')
        # print('No line_segment segments detected')
        return lane_lines

    height, width, _ = frame.shape
    left_fit = []
    right_fit = []

    boundary = 1/3
    left_region_boundary = width * (1 - boundary)  # left lane line segment should be on left 2/3 of the screen
    right_region_boundary = width * boundary # right lane line segment should be on left 2/3 of the screen


    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                # logging.info('skipping vertical line segment (slope=inf): %s' % line_segment)
                # print('skipping vertical line segment')
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines.append(make_points(frame, left_fit_average))

    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(frame, right_fit_average))

    # logging.debug('lane lines: %s' % lane_lines)  # [[[316, 720, 484, 432]], [[1009, 720, 718, 432]]]
    # print('lane lines: %s' % lane_lines')
    return lane_lines


def make_points(frame, line):
    height, width, _ = frame.shape
    slope, intercept = line
    y1 = height  # bottom of the frame
    y2 = int(y1 * 1 / 2)  # make points from middle of the frame down

    # bound the coordinates within the frame
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    
    return [[x1, y1, x2, y2]]

    

def display_lines(frame, lines, line_color=(0, 255, 0), line_width=2):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    
    return line_image

