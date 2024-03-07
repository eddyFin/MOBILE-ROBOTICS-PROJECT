# function to impose motor speed
def motors(l_speed=500, r_speed=500, verbose=False):
    """
    Sets the motor speeds of the Thymio 
    l_speed:    left motor speed
    r_speed:    right motor speed
    verbose:    whether to print status messages or not
    """
    # Printing the speeds if requested
    if verbose:
        print("\t\t Setting speed : ", l_speed, r_speed)
    return {
        "motor.left.target": [l_speed],
        "motor.right.target": [r_speed],
    }


def convert_to_motor_speed(v, w, R, L, conv_ratio_right, conv_ratio_left): 
    """
    Find the conversion factor from the robot speed [linear and angular velocity 
    in absolute reference] to wheel speed in Thymio units.
    """ 
    # converting the linear and angular velocities to motor speeds
    vr = (2*v + w*L)/(2*R) # right motor speed
    vl = (2*v - w*L)/(2*R) # left motor speed
    
    # converting the motor speeds to Thymio units
    vr_thymio = vr * conv_ratio_right
    vl_thymio = vl * conv_ratio_left
    
    return vr_thymio, vl_thymio




def convert_to_absolute_speed(vr_thymio, vl_thymio, R, L, conv_ratio_right, conv_ratio_left): 
    """
    Find the conversion factor from wheel speed in Thymio units to the robot 
    speed [linear and angular velocity in absolute reference] 
    """

    # convert Thymio speeds to rad/s velocities
    vr = vr_thymio / conv_ratio_right
    vl = vl_thymio / conv_ratio_left
    
    # convert rad/s velocities to linear and angular velocities
    v = R*(vr + vl)/2   # linear velocity
    w = R*(vr - vl)/L   # angular velocity
    
    return v, w