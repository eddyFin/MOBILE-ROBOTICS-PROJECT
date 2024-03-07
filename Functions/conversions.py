import math

########## IMG_TO_MAP_COORDINATES ##########

def img_to_map_coordinates(coord_tuple, image_height):
    """
    Converts a tuple in the image coordinate system into
    a tuple in the control coordinate system.

    INPUTS:
        coord_tuple : a tuple of three values in the  
        image coordinate system.
        image_height : the height of the map in the same
        unit as coord_tuple

    OUTPUT:
        (x, new_y, z) : The initial tuple in the control coordinate system.
    
    """
    x, y, z = coord_tuple
    new_y = image_height - y
    z = 2 * math.pi - z
    return (x, new_y, z)


########## GRID_TO_MAP_COORDINATES ##########

def grid_to_map_coordinates(coordinates_list, image_height):
    """
    Converts a list of tuples in the image coordinate system into
    a list of tuples in the control coordinate system.

    INPUTS:
        coordinates_list : a list of tuples in the  
        image coordinate system.
        image_height : the height of the map in the same
        unit as the tuples in coordinates_list.

    OUTPUT:
        transformed_coordinates : a list of tuples in the 
        control coordinate system.
    
    """
    transformed_coordinates = []

    for x, y in coordinates_list:
        new_x = y  # Inversion of x axis
        new_y = image_height - x
        transformed_coordinates.append((new_x, new_y))

    return transformed_coordinates


########## GRID_TO_MM ##########

def grid_to_mm(occupancy_grid, map_size):
    """
    Returns the length of a grid cell in mm.

    INPUTS:
        occupancy_grid : the occupancy grid
        map_size (tuple of 2 values) : the dimensions of the real map in mm

    OUPUT: 
         square_size : the length of a grid cell in mm.

    """

    width_grid,_ = occupancy_grid.shape
    width_mm,_ = map_size
    square_size = width_mm/width_grid

    return square_size # size, in mm, of a cell of the grid