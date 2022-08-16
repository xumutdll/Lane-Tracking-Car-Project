import cv2
import numpy as np
import time
import LaneDetectionAdvanced as LDetect
import SteeringAdvanced as Steer

height = 480
width = 640
speed_A = 0
speed_B = 0
fps,st,frames_to_count,count = (0,0,10,0)
i = 0


### Create the specific calibration matrixes and store it in .csv files for later use
# mtx, dist = LDetect.camera_calibration('./calib_imgs/calibration*.jpg', debug = True)
# np.savetxt('mtx.csv', mtx, delimiter=',')
# np.savetxt('dist.csv', dist, delimiter=',')
mtx = np.loadtxt('mtx_640x480.csv', delimiter=',')
dist = np.loadtxt('dist_640x480.csv', delimiter=',')

# Create warp matrixes. No need for store them.
widthTop = 120 ; heightTop = 165 ; widthBottom = 0 ; heightBottom = 480
points = np.float32([(widthTop, heightTop), (640-widthTop, heightTop),
                (widthBottom , heightBottom ), (640-widthBottom, heightBottom)])

warp_m, warp_minv = LDetect.warp_matrixes(points, inv=False) 


def main(img, speed):
    global fps, st, frames_to_count, count
    # cv2.imshow('Frame', img)

    undistorted_img =  cv2.undistort(img, mtx, dist)
    # cv2.imshow('Undistorted', undistorted_img)

    blurred = LDetect.gaussian_blur(undistorted_img, kernel=3)
    # cv2.imshow('Blurred', blurred)

    combined_treshold = LDetect.get_thresholded_image(blurred)
    # cv2.imshow('Combined Threshold', combined_treshold)

    
    warped = cv2.warpPerspective \
        (combined_treshold, warp_m, (width, height), flags=cv2.INTER_LINEAR)
    # cv2.imshow('Warped', warped)

    lines_fit, left_points, right_points = \
        LDetect.detect_lines(warped, return_img=False)
    # cv2.imshow('Detected Lanes', Lane_Lines)     

    lines_fit, left_points, right_points, Average_Lines = \
        LDetect.detect_similar_lines(warped, lines_fit, return_img=True)
    # cv2.imshow('Detected Lanes', Average_Lines)

    x_offset = Steer.car_offset(left_points[0], right_points[0], width)
    # print(x_offset)

    steering_angle = Steer.steering_angle(x_offset, -height/60) 
    print(steering_angle)
 
    speed_A, speed_B = Steer.angle_to_speed(steering_angle, speed)

    heading_line = Steer.display_heading_line(Average_Lines, steering_angle)
    cv2.imshow('Result', heading_line)

    if count == frames_to_count:
        try:
            fps = round(frames_to_count/(time.time()-st))
            st = time.time()
            count = 0
        except:
            pass
    count += 1

    img_lane = LDetect.draw_lane \
        (undistorted_img, warped, left_points, right_points, warp_minv)
    img_lane = cv2.putText(img_lane,'FPS: '+ str(fps+3), (5,20), cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),1)
    img_lane = cv2.putText(img_lane,'Steering Angle: '+ str(steering_angle), (5,40), cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),1)
    img_lane = cv2.putText(img_lane,'Speed A: '+ str(speed_A), (5,60), cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),1)
    img_lane = cv2.putText(img_lane,'Speed B: '+ str(speed_B), (5,80), cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),1)
    cv2.imshow('Path', img_lane)

    return speed_A, speed_B 
