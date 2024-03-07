import cv2
import numpy as np
import math
import keyboard
import matplotlib.pyplot as plt

########## WAIT_FOR_SPACE ##########

def wait_for_space():
    """
    Waits until space is type on the keyboard.

    """
    print("Press the space key to continue...")
    keyboard.wait("space")
    print("Space key pressed! Continuing...")


########## PIXELS_TO_MM ##########

def pixels_to_mm(img, map_size):
    """
    Converts pixels into mm.

    INPUTS
        img (image) : The image of the map after reshaping and resizing
        map_size (tuple of 2 values) : The real size of the map (width, length) in mm

    OUPUT
        pixel_size : the size of a pixel in mm

    """
    width_pixels,_,_ = img.shape
    width_mm,_ = map_size
    pixel_size = width_mm/width_pixels
    return pixel_size # size, in mm, of a pixel


########## NEW_SIZE_IMG ##########

def new_size_img(width_img, length_img, width_mm, length_mm):
    """
    This function takes as input the dimensions of an image of the map.
    These dimensions are not proportional to the real dimensions of the
    map. 
    The function returns the new dimensions the input image should have
    to be proportional to the real map dimensions.
    
    INPUTS
        width_img, length_img : dimensions of the image
        width_mm, length_mm : dimensions of the real map

    OUPUT
        new_size (tuple) : dimensions the image should have
        to be proportional to the real map dimensions 

    """
    desired_length_pxl = math.floor(width_img*length_mm/width_mm)
    desired_width_pxl = math.floor(length_img*width_mm/length_mm)
    crop_length = abs(length_img - desired_length_pxl)
    crop_width = abs(width_img-desired_width_pxl)
    if crop_length<crop_width:
        new_size = (width_img,desired_length_pxl)
    else:
        new_size = (desired_width_pxl,length_img)
    return new_size  


########## CAMERA_OBSTRUCTED ##########


def camera_obstructed(frame, threshold=30):
    """
    Returns 1 if the camera is obstructed, 0 otherwise.

    """
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate the average intensity of the pixels
    average_intensity = np.mean(gray_frame)

    # Check if the average intensity is below the threshold
    if average_intensity < threshold:
        # Frame is black or close to black
        return 1

    # Frame is not black
    return 0   


########## CAPTURE_FRAME ##########

def capture_frame(camera_index=0):
    """
    Returns the first frame of the camera that contains
    the robot markers and the the goal marker at the same
    time. 

    """
    # Open the camera
    cap = cv2.VideoCapture(camera_index)

    # Load the ArUco dictionary
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)

    # Initialize the ArUco parameters
    parameters = cv2.aruco.DetectorParameters_create()

    while True:
        # Capture the current frame
        ret, frame = cap.read()

        if ret:  # Check if the frame was successfully captured
            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect ArUco markers in the frame
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

            # Draw the detected markers on the frame
            frame_markers = frame.copy()
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame_markers, corners, ids)

                # Display the frame with markers
                cv2.imshow('Frame with Markers', frame_markers)

                # Check if both markers ID4 and ID5 are detected
                if 4 in ids and 5 in ids:
                    
                    # Release the camera and close all windows
                    # cap.release()
                    cv2.destroyAllWindows()

                    return frame, cap

            # Display the frame with markers
            cv2.imshow('Frame with Markers', frame_markers)

        # Exit the loop when needed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # If the loop exits without returning, release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

    return None


########## DETECT_MAP_CORNERS ##########

def detect_map_corners(camera_index=0):
    """
    Detects the map corners and returns their position. 
    It also returns the frame at the moment the corners were 
    all detected.

    """
    # Open the camera
    cap = cv2.VideoCapture(camera_index)

    # Load the ArUco dictionary
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)

    # Initialize the ArUco parameters
    parameters = cv2.aruco.DetectorParameters_create()

    last_frame = None  # Variable to store the last captured frame

    while True:
        # Capture the current frame
        ret, frame = cap.read()

        if ret:  # Check if the frame was successfully captured
            last_frame = frame.copy()  # Save the current frame

            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect ArUco markers in the frame
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

            # Draw the detected markers on the frame
            frame_markers = frame.copy()
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame_markers, corners, ids)

                # Display the frame with markers
                cv2.imshow('Frame with Markers', frame_markers)

                # Check if all four markers are detected
                if len(ids) == 4 and set(ids.flatten()) == {0, 1, 2, 3}:
                    
                    # Extract the positions of the four corners
                    map_corners = [None] * 4
                    for i in range(4):
                        marker_index = ids.tolist().index([i])
                        marker_corners = corners[marker_index][0]
                        map_corners[i] = tuple(marker_corners)

                    # Release the camera and close all windows
                    cap.release()
                    cv2.destroyAllWindows()

                    # Extracting the desired corners
                    corner_position = [
                        tuple(map(int, map_corners[0][0])),  # Top left corner of marker 0
                        tuple(map(int, map_corners[1][3])),  # Bottom left corner of marker 1
                        tuple(map(int, map_corners[2][2])),  # Bottom right corner of marker 2
                        tuple(map(int, map_corners[3][1]))   # Top right corner of marker 3
                    ]
                    
                    return corner_position, last_frame

            # Display the frame with markers
            cv2.imshow('Frame with Markers', frame_markers)

        # Exit the loop when needed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # If the loop exits without returning, release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

    return None, None  # Return None if no corners were detected


########## DETECT_ARUCO_MARKERS ##########

def detect_aruco_marker(image, target_marker_id):
    """
    Returns the position and orientation of the desired
    aruco marker if the marker is detected.

    INPUTS:
        image: the current frame of the camera
        target_marker_id : the id of the desired marker

    OUPUT :
        (center_x, center_y, marker_orientation) : position
        and orientation of the desired marker if detected,
        (0,0,0) otherwise. 

    """
    # Load the ArUco dictionary and parameters
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters_create()

    # Detect markers in the image
    corners, ids, _ = cv2.aruco.detectMarkers(image, aruco_dict, parameters=aruco_params)

    # Check if the target marker ID is detected
    if ids is not None and target_marker_id in ids:
        # Get the index of the marker with the target ID
        marker_index = np.where(ids == target_marker_id)[0][0]

        # Get the corners of the detected marker
        marker_corners = corners[marker_index][0]

        # Calculate the center of the marker
        center_x = np.mean(marker_corners[:, 0])
        center_y = np.mean(marker_corners[:, 1])

        # Calculate the orientation of the marker with respect to the x-axis
        delta_x = marker_corners[1, 0] - marker_corners[0, 0]
        delta_y = marker_corners[1, 1] - marker_corners[0, 1]
        
        marker_orientation = np.arctan2(delta_y, delta_x) % (2 * np.pi)

        # Return the detected pose
        return (center_x, center_y, marker_orientation)
    else:
        # If the target marker ID is not detected, return the previous robot pose
        return (0,0,0)


########## RESHAPE_IMG ##########

def reshape_img(img, map_size, map_corners):

    """
    Takes as input an image from the camera, and returns
    an image of the map, rescaled and resized so that it is 
    proportional to the real dimensions of the map.

    INPUTS:
        img (image) : A frame from the camera
        map_size (tuple of 2 values) : the dimensions of the real map in mm
        map_corners : the position of the corners of the map in the camera frame

    OUPUT:
        reshaped_img: the reshaped and resized image.

    """

    top_left_corner,bottom_left_corner, bottom_right_corner, top_right_corner = map_corners 
    
    # get width and length of the map in mm
    width_mm, length_mm = map_size

    # Get the size of the initial image
    (width_img, length_img, _) = img.shape
    
    # Get new size of the image
    width_pixels, length_pixels = new_size_img(width_img, length_img, width_mm, length_mm)
    new_shape_img = (length_pixels, width_pixels)

    # Specify input and output coordinates that is used when reshaping
    # to calculate the transformation matrix
    input_pts = np.float32([[top_left_corner[0],top_left_corner[1]],[bottom_left_corner[0],
                            bottom_left_corner[1]],[bottom_right_corner[0],bottom_right_corner[1]],[top_right_corner[0],top_right_corner[1]]]) 
    output_pts = np.float32([[0,0],[0,width_img],[length_img,width_img],[length_img,0]]) 
    
    # Compute the perspective transform M
    M = cv2.getPerspectiveTransform(input_pts,output_pts)
    
    # Apply the perspective transformation to the image
    out = cv2.warpPerspective(img,M,(img.shape[1], img.shape[0]),flags=cv2.INTER_LINEAR)

    # Resize the image to make it coherent with the real size of the map
    reshaped_img = cv2.resize(out, new_shape_img)

    return reshaped_img


########## IMG_TO_GRID ##########


def img_to_grid(img, map_size, map_corners):

    """
    Takes as input an image from the camera, and returns
    the occupancy grid as a matrix of zeros and ones.

    INPUTS:
        img (image) : A frame from the camera
        map_size (tuple of 2 values) : the dimensions of the real map in mm
        map_corners : the position of the corners of the map in the camera frame

    OUPUT:
        occupancy_grid : the occupancy grid. The ones represent the obstacles,
        the zeros represent the free spaces.

    """

    plt.figure()
    plt.title("Original image")
    plt.imshow(img)
    plt.show()
    
    reshaped_img = reshape_img(img, map_size, map_corners)

    plt.figure()
    plt.title("Reshaped image") 
    plt.imshow(reshaped_img)
    plt.show()
    

    # Return occupancy_grid from and image
    bilateral = cv2.bilateralFilter(reshaped_img,9,75,75)
    bw_bilateral = cv2.cvtColor(bilateral, cv2.COLOR_BGR2GRAY)

    ret,binary_image = cv2.threshold(bw_bilateral,120, 200,cv2.THRESH_BINARY)

    plt.figure()
    plt.title("Binary image")
    plt.imshow(binary_image)
    plt.show()
    

    # Grow the obstacles
    kernel = np.ones((5, 5), np.uint8)
    dilated_image = cv2.erode(binary_image, kernel, iterations=20) #increase nb of iterations to increase dilation

    plt.figure()
    plt.title("Dilated image")
    plt.imshow(dilated_image)
    plt.show()
    

    cell_size = 20  # Adjust the cell size as needed
    height, width = dilated_image.shape

    new_height = (height // cell_size) * cell_size
    new_width = (width // cell_size) * cell_size
    resized_result_image = cv2.resize(dilated_image, (new_width, new_height))

    occupancy_grid = np.zeros((new_height // cell_size, new_width // cell_size), dtype=np.uint8)

    for i in range(0, new_height, cell_size):
            for j in range(0, new_width, cell_size):
                cell = resized_result_image[i:i+cell_size, j:j+cell_size]
                occupancy = np.sum(cell == 0) / (cell_size * cell_size)
                occupancy_grid[i // cell_size, j // cell_size] = occupancy > 0.5

    return occupancy_grid  


########## GET_ROBOT_POSE ##########

def get_robot_pose(frame, map_size, map_corners, prev_robot_pose):
    """ 
    Returns the current robot pose (position and orientation)

    INPUTS:
        frame (image) : the current frame read from the camera
        map_size (tuple of 2 values) : the dimensions of the real map in mm
        map_corners : the position of the corners of the map in the camera frame
        prev_robot_pose (tuple of 3 values) : the previous robot pose

    OUTPUT:
        robot_pos (tuple of 3 values) : The current robot pose

    """

    reshaped_img = reshape_img(frame, map_size, map_corners)
    robot_pos = detect_aruco_marker(reshaped_img, 5)
    
    # Express the robot pose in mm
    factor_pixels = pixels_to_mm(frame, map_size)
    robot_pos = (robot_pos[0] * factor_pixels, robot_pos[1] * factor_pixels, robot_pos[2])

    return robot_pos


########## GET_ROBOT_POS ##########

def get_goal_pos(frame, map_size, map_corners):
    """
    Returns the current goal pose (position and orientation)

    INPUTS:
        frame (image) : the current frame read from the camera
        map_size (tuple of 2 values) : the dimensions of the real map in mm
        map_corners : the position of the corners of the map in the camera frame
       
    OUTPUT:
        marker_pos (tuple of 3 values) : The current goal pose
    
    """

    reshaped_img = reshape_img(frame, map_size, map_corners)
    marker_pos = detect_aruco_marker(reshaped_img, 4)
    
    # Express the goal pose in mm
    factor_pixels = pixels_to_mm(frame, map_size)
    marker_pos = (marker_pos[0] * factor_pixels, marker_pos[1] * factor_pixels, marker_pos[2])

    return marker_pos

