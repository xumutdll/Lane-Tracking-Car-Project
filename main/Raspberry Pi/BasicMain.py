import cv2
import numpy as np
import time
import LaneDetectionBasic as LDetect
import SteeringBasic as Steer
import Control


i = 0
flag = 0
new_angle = 90
fps,st,frames_to_count,count = (0,0,10,0)


### Create the specific calibration matrixes and store it in .csv files for further use
# mtx, dist = LDetect.camera_calibration('./calib_imgs/calibration*.jpg', debug = True)
# np.savetxt('mtx.csv', mtx, delimiter=',')
# np.savetxt('dist.csv', dist, delimiter=',')
mtx = np.loadtxt('mtx_320x240.csv', delimiter=',')
dist = np.loadtxt('dist_320x240.csv', delimiter=',')

camera = cv2.VideoCapture(-1)
camera.set(3, 320)
camera.set(4, 240)


while (camera.isOpened()):
    _, img = camera.read()

    undistorted_img =  cv2.undistort(img, mtx, dist)
    # undistorted_img = cv2.putText(undistorted_img,'FPS: '+ str(fps), (5,20), cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),1)
    # cv2.imshow('Undistorted', undistorted_img)

    # blurred = LDetect.gaussian_blur(undistorted_img, kernel=3)
    # cv2.imshow('Blurred', blurred)

    edges = LDetect.detect_edges(undistorted_img)        
    # cv2.imshow('Edges', edges)

    roi = LDetect.region_of_interest(edges)
    line_segments = LDetect.detect_line_segments(roi)
    lane_lines =  LDetect.average_slope_intercept(undistorted_img, line_segments)
    
    lane_lines_image = LDetect.display_lines(undistorted_img, lane_lines)
    # cv2.imshow("Lane Lines", lane_lines_image)

    if count == frames_to_count:
        try:
            fps = round(frames_to_count/(time.time()-st))
            st = time.time()
            count = 0
        except:
            pass
    count += 1
    

    # else:
    # curr_angle = new_angle
    new_angle =  Steer.steering_angle(lane_lines)
    # stabilized_angle = Steer.stabilize_steering_angle(curr_angle, new_angle, np.size(lane_lines)/4)
    # print(str(new_angle) + "  " + str(stabilized_angle))
    
    if i < 12:
        i = i + 1
        # speed_A, speed_B = Steer.angle_to_speed(new_angle, speed= 670)
        speed_A, speed_B = Steer.angle_to_speed(new_angle, speed= 780)
    else:
        # speed_A, speed_B = Steer.angle_to_speed(new_angle, speed= 545)
        speed_A, speed_B = Steer.angle_to_speed(new_angle, speed= 670)
    
    if np.size(lane_lines) == 8:
        print(new_angle, "CIFT")
    elif np.size(lane_lines) == 4:
        print(new_angle, "TEK")
    else:
        print("HATA")

    # print(str(speed_A) + "  " + str(speed_B))
    Control.forward(int(speed_A), int(speed_B))

    # theLine = Steer.display_heading_line(undistorted_img, new_angle)
    # cv2.imshow("Steering Line", theLine)


