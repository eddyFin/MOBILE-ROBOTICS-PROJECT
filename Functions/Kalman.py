import numpy as np
import math

def getJacobainA(yaw, deltak, v):
    """
    Calculates and returns the Jacobian matrix for the Extended Kalman Filter, considering a linearized dynamics model
    """
    jacobianA = np.array([
        [1.0, 0.0, -deltak * v * math.sin(yaw), deltak * math.cos(yaw), 0.0],
        [0.0, 1.0, deltak * v * math.cos(yaw), deltak * math.sin(yaw), 0.0],
        [0.0, 0.0, 1.0, 0.0, deltak],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0]
        ])

    return jacobianA

def getB(yaw, deltak):
    """
    Calculates and returns the B matrix, which is time variant
    """
    B = np.array([  [np.cos(yaw)*deltak, 0],
                    [np.sin(yaw)*deltak, 0],
                    [0, deltak],
                    [1, 0],
                    [0, 1]])
    return B

def ekf1(z_k_observation_vector, state_estimate_k_minus_1, 
        control_vector_k_minus_1, P_k_minus_1, dk, camera_is_obstructed):
    """
    Extended Kalman Filter. Implements the Extended Kalman Filter for the Thymio robot.
         
    INPUT
        z_k_observation_vector      The observation from the Camera and from the Odometry

        state_estimate_k_minus_1    State estimate at time k-1
            
        control_vector_k_minus_1    The control vector applied at time k-1

        P_k_minus_1                 The state covariance matrix estimate at time k-1

        dk                          Sampling time in seconds

        camera_obstructed           Boolean value indicating whether the camera is obstructed or not
             
    OUTPUT
        state_estimate_k            State estimate at time k  
        
        P_k                         State covariance_estimate for time k
        
    """
    
    # Process noise
    process_noise_v_k_minus_1 = np.array([0,0,0,0,0])


    # Sensor noise. 
    sensor_noise_w_k = np.array([0,0,0,0,0])

    A_k_minus_1 = np.array([[1.0,  0,   0, 0, 0],
                            [  0,1.0,   0, 0, 0],
                            [  0,  0, 1.0, 0, 0],
                            [  0,  0, 0, 0, 0],
                            [  0,  0, 0, 0, 0]])

    # State model noise covariance matrix Q_k
    var_x = 3.5289
    var_y = 3.5289
    var_theta = 0.0013
    var_v = 0.0339
    var_w = 0.0157

    Q_k = np.array([[var_x,   0.0,     0.0,    0.0, 0.0],
                    [0.0,     var_y,   0.0,    0.0, 0.0],
                    [0.0,     0.0,     var_theta, 0.0, 0.0],
                    [0.0,     0.0,     0.0,    var_v, 0.0],
                    [0.0,     0.0,     0.0,    0.0,   var_w]])
                    
    # Measurement matrix H_k
    H_k = np.array([[1.0,  0.0, 0.0, 0.0, 0.0],
                    [0.0,  1.0, 0.0, 0.0, 0.0],
                    [0.0,  0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0]])
                            
    # Sensor measurement noise covariance matrix R_k
    r_x = 0.003
    r_y = 0.003
    r_theta = 0.001
    r_v =  var_v
    r_w =  var_w
    R_k = np.array([[r_x,   0,    0, 0.0, 0.0],
                    [  0, r_y,    0, 0.0, 0.0],
                    [  0,    0, r_theta, 0.0, 0.0],
                    [0.0,  0.0, 0.0, r_v, 0.0],
                    [0.0, 0.0, 0.0, 0.0, r_w]]) 

    r_obs = np.inf
    R_k_obstr = np.array([[r_obs,   0,    0, 0.0, 0.0],
                    [  0, r_obs,    0, 0.0, 0.0],
                    [  0,    0, r_obs, 0.0, 0.0],
                    [0.0,  0.0, 0.0, r_v, 0.0],
                    [0.0, 0.0, 0.0, 0.0, r_w]])

    # PREDICT
    # Predict the state estimate at time k based on the state 
    # estimate at time k-1 and the control input applied at time k-1.
    state_estimate_k = A_k_minus_1 @ state_estimate_k_minus_1 + (
            (getB(state_estimate_k_minus_1[2],dk)) @ control_vector_k_minus_1 + 
            process_noise_v_k_minus_1)
             
           
    # Predict the state covariance estimate based on the previous
    # covariance and some noise

    jacobianA = getJacobainA(state_estimate_k_minus_1[2], dk, control_vector_k_minus_1[0])
    

    P_k = jacobianA @ P_k_minus_1 @ jacobianA.T + (Q_k)
         
    # UPDATE
    # Calculate the difference between the actual sensor measurements
    # at time k minus what the measurement model predicted 
    # the sensor measurements would be for the current timestep k.
    measurement_residual_y_k = z_k_observation_vector - (
            (H_k @ state_estimate_k) + (
            sensor_noise_w_k))

    state_estimate_k[2] = state_estimate_k[2] % (2*np.pi)
    
    # Calculate the measurement residual covariance  
    if camera_is_obstructed:

        S_k = H_k @ P_k @ H_k.T + R_k_obstr
    else:
        S_k = H_k @ P_k @ H_k.T + R_k

 
    # Calculate the near-optimal Kalman gain
    # We use pseudoinverse since some of the matrices might be
    # non-square or singular.
         
    K_k = P_k @ H_k.T @ np.linalg.pinv(S_k)
        
    # Calculate an updated state estimate for time k
    state_estimate_k = state_estimate_k + (K_k @ measurement_residual_y_k)

     
    # Update the state covariance estimate for time k
    P_k = P_k - (K_k @ H_k @ P_k)
     
 
    # Return the updated state and covariance estimates
    return state_estimate_k, P_k