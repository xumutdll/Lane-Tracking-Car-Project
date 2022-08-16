import cv2
import numpy as np

# We can create two functions for the trackbars. One that initializes the trackbars and the second that get the current value from them.

def nothing(a):
    pass

def initializeTrackbars(intialTracbarVals, wT=480, hT=240):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Width Top", "Trackbars", intialTracbarVals[0],wT//2, nothing)
    cv2.createTrackbar("Height Top", "Trackbars", intialTracbarVals[1], hT, nothing)
    cv2.createTrackbar("Width Bottom", "Trackbars", intialTracbarVals[2],wT//2, nothing)
    cv2.createTrackbar("Height Bottom", "Trackbars", intialTracbarVals[3], hT, nothing)

def valTrackbars(wT=640, hT=480):
    widthTop = cv2.getTrackbarPos("Width Top", "Trackbars")
    heightTop = cv2.getTrackbarPos("Height Top", "Trackbars")
    widthBottom = cv2.getTrackbarPos("Width Bottom", "Trackbars")
    heightBottom = cv2.getTrackbarPos("Height Bottom", "Trackbars")
    points = np.float32([(widthTop, heightTop), (wT-widthTop, heightTop),
                      (widthBottom , heightBottom ), (wT-widthBottom, heightBottom)])
    return points
# Now we can call the initialize function at the start of the code and the valTrackbar in the while loop just before warping the image. Since both functions are written in the utlis file we will write ‘utlis.’ before calling them.

intialTracbarVals = [98,180,0,480]
initializeTrackbars(intialTracbarVals)
	
# Now we will write our warping function that will allow us to get the bird eyes view using the four points that we just tuned.

def warpImg (img, points, inv=False):
    h, w, _ = img.shape
    pts1 = np.float32(points)
    pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    if inv:
        matrix = cv2.getPerspectiveTransform(pts2,pts1)
    else:
        matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgWarp = cv2.warpPerspective(img,matrix,(w,h))
    return imgWarp


def drawPoints(img,points):
    for x in range( 0,4):
        cv2.circle(img,(int(points[x][0]),int(points[x][1])),15,(0,0,255),cv2.FILLED)
    return img
# Now we can call this function to draw the points.


img = cv2.imread('yeni_warp.jpg')
mtx = np.loadtxt('mtx_640x480.csv', delimiter=',')
dist = np.loadtxt('dist_640x480.csv', delimiter=',')

while True:
    resized = cv2.resize(img, (640, 480), interpolation= cv2.INTER_LINEAR)
    undistorted_img =  cv2.undistort(resized, mtx, dist, None, mtx)

    
    # cv2.imshow('Blurred', blurred)
               
    points = valTrackbars()
    imgWarp = warpImg(undistorted_img, points)
    imgWarpPoints = drawPoints(undistorted_img, points)
    cv2.imshow('Warped', imgWarp)

    cv2.imshow('Points',imgWarpPoints)
    

    # combined_treshold = LDetect.get_thresholded_image(blurred)
    # cv2.imshow('Combined', combined_treshold)

    # roi = LDetect.region_of_interest(combined_treshold)
    # cv2.imshow('Region Of Interest', roi)

    # line_segments = LDetect.detect_line_segments(roi)
    # lane_lines =  LDetect.average_slope_intercept(blurred, line_segments)
    # lane_lines_image = LDetect.display_lines(undistorted_img, lane_lines)
    # cv2.imshow("Lane Lines", lane_lines_image)

    if cv2.waitKey(1) == 'q':
        break

cv2.destroyAllWindows()



