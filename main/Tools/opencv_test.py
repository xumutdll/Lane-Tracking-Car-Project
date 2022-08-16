import cv2
import numpy as np
import matplotlib.pyplot as plt
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
        img = cv2.resize(img, (640, 480), interpolation= cv2.INTER_LINEAR)
        
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

mtx, dist = camera_calibration('./calib_imgs/calibration*.jpg',debug = True)

camera = cv2.VideoCapture(-1)
camera.set(3, 640)
camera.set(4, 480)

while(camera.isOpened()):
    success, image = camera.read()
    if (success is not True):
        print("No frame detected")

    # cv2.imshow('Distorted Image', image)
    
    undistorted_img = cv2.undistort(image, mtx, dist, None, mtx)
    cv2.imshow('Undistorted Image', undistorted_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cv2.destroyAllWindows()
    
    