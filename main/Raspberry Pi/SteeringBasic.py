from logging import exception
from multiprocessing.spawn import old_main_modules
from ssl import OP_SINGLE_ECDH_USE
import cv2
import numpy as np
import math
import Control

height = 240
width = 320
exception_counter = 0


def steering_angle(lane_lines):
    global exception_counter, old_steering_angle
    if np.size(lane_lines) == 8:
        _, _, left_x2, _ = lane_lines[0][0]
        _, _, right_x2, _ = lane_lines[1][0]
        mid = int(width / 2)
        x_offset = (left_x2 + right_x2) / 2 - mid
        y_offset = int(height / 2.2)
    elif np.size(lane_lines) == 4:
        x1, _, x2, _ = lane_lines[0][0]
        x_offset = x2 - x1
        # y_offset = int(height / 2.65) ## Viraj hzıı
        y_offset = int(height / 4.2) ## Viraj hzıı
    else:
        exception_counter = exception_counter + 1
        if exception_counter == 10:
            Control.stopcar()
            exit()


    # try:
    #     _, _, left_x2, _ = lane_lines[0][0]
    #     _, _, right_x2, _ = lane_lines[1][0]
    # except:
    #     Control.stopcar()
    #     exit()
    # mid = int(width / 2)
    # x_offset = (left_x2 + right_x2) / 2 - mid
    # y_offset = int(height / 2)

    try:
        angle_to_mid_radian = math.atan(x_offset / y_offset)  # angle (in radian) to center vertical line
        angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)  # angle (in degrees) to center vertical line
        steering_angle = angle_to_mid_deg  # this is the steering angle needed by picar front wheel 
        old_steering_angle = steering_angle
    except:
        return old_steering_angle
    
    
    return steering_angle


def angle_to_speed(steering_angle, speed = 0):
    if steering_angle > 0:
        speed_A = speed + (speed/70)*steering_angle
        speed_B = speed
    elif steering_angle < 0:
        speed_A = speed
        speed_B = speed + (speed/70)*abs(steering_angle)
    else:
        speed_A = speed
        speed_B = speed

    return int(speed_A), int(speed_B)


def stabilize_steering_angle(
          curr_steering_angle,
          new_steering_angle,
          num_of_lane_lines,
          max_angle_deviation_two_lines=5,
          max_angle_deviation_one_lane=1):
    """
    Using last steering angle to stabilize the steering angle
    if new angle is too different from current angle,
    only turn by max_angle_deviation degrees
    """

    if num_of_lane_lines == 2 :
        # if both lane lines detected, then we can deviate more
        max_angle_deviation = max_angle_deviation_two_lines
    else :
        # if only one lane detected, don't deviate too much
        max_angle_deviation = max_angle_deviation_one_lane

    angle_deviation = new_steering_angle - curr_steering_angle
    if abs(angle_deviation) > max_angle_deviation:
        stabilized_steering_angle = int(curr_steering_angle
            + max_angle_deviation * angle_deviation / abs(angle_deviation))
    else:
        stabilized_steering_angle = new_steering_angle

    return stabilized_steering_angle


def display_heading_line(frame, steering_angle, line_color=(0, 0, 255), line_width=5):
    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape
    steering_angle = steering_angle + 90

    # figure out the heading line from steering angle
    # heading line (x1,y1) is always center bottom of the screen
    # (x2, y2) requires a bit of trigonometry

    # Note: the steering angle of:
    # 0-89 degree: turn left
    # 90 degree: going straight
    # 91-180 degree: turn right
    steering_angle_radian = steering_angle / 180.0 * math.pi
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 2)

    cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
    heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)

    return heading_image