import numpy as np
import math


def control_law1(state_estimate_k, x_goal, y_goal):
    
    """  
    alpha is the angle to the goal relative to the heading of the robot
    kp_alpha*alpha drives the robot along a line towards the goal, if kp alpha is positive, the robot will turn right
    
    """

    Kp_alpha = 0.5

    #the current position estimate (x,y and theta) from the filter is defined 
    x = state_estimate_k[0]
    y = state_estimate_k[1]
    theta = state_estimate_k[2]*180/math.pi
    
    #the difference between current position and the next temporary goal
    x_diff = x_goal - x
    y_diff = y_goal - y

    # alpha is the calculated as the difference in angles between the current angle and the angle that it should 
    #  have to reach the next goal. Alpha is restricted between -180 and 180 degrees.
      
    angle_to_follow = math.atan2(y_diff, x_diff)*180/math.pi
    alpha = ( -theta + angle_to_follow + 180) % 360 - 180     
    
    # Linear velocity is set to a constant value in mm/s, which corresponds to 100 thymio units
    v = 31
    
    # The updated angular velocity is calculated based on the angle correction alpha multiplied by a gain.
    w = Kp_alpha * alpha*2*math.pi/(360)

    # The control  vector returns the linear and angular velocities computed which will then be converted to Thymio left and right velocities
    return v,w


