import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


########## RECONSTRUCT_PATH ##########

def reconstruct_path(cameFrom, current):
    """
    Recurrently reconstructs the path from start node to the current node
    :param cameFrom: map (dictionary) containing for each node n the node immediately 
                     preceding it on the cheapest path from start to n 
                     currently known.
    :param current: current node (x, y)
    :return: list of nodes from start to current node
    """
    total_path = [current]
    while current in cameFrom.keys():
        # Add where the current node came from to the start of the list
        total_path.insert(0, cameFrom[current]) 
        current=cameFrom[current]
    return total_path


########## GET_MOVEMENT_4N ##########

def get_movements_4n():
    """
    Get all possible 4-connectivity movements (up, down, left right).
    :return: list of movements with cost [(dx, dy, movement_cost)]
    """
    return [(1, 0, 1.0),
            (0, 1, 1.0),
            (-1, 0, 1.0),
            (0, -1, 1.0)]


########## GET_MOVEMENT_8N ##########

def get_movements_8n():
    """
    Get all possible 8-connectivity movements. Equivalent to get_movements_in_radius(1)
    (up, down, left, right and the 4 diagonals).
    :return: list of movements with cost [(dx, dy, movement_cost)]
    """
    s2 = math.sqrt(2)
    return [(1, 0, 1.0),
            (0, 1, 1.0),
            (-1, 0, 1.0),
            (0, -1, 1.0),
            (1, 1, s2),
            (-1, 1, s2),
            (-1, -1, s2),
            (1, -1, s2)]


########## A_STAR ##########

def A_Star(start, goal, occupancy_grid, movement_type="8N"):
    """
    A* for 2D occupancy grid. Finds a path from start to goal.
    h is the heuristic function. h(n) estimates the cost to reach goal from node n.
    :param start: start node (x, y)
    :param goal_m: goal node (x, y)
    :param occupancy_grid: the grid map
    :param movement: select between 4-connectivity ('4N') and 8-connectivity ('8N')
    :return: a tuple that contains: (the resulting path in meters, the resulting path in data array indices)
    
    """
    
    # Dimensions of the occupancy grid
    x_dim = occupancy_grid.shape[0]  
    y_dim = occupancy_grid.shape[1]  

    # Get all the possible components of the occupancy_grid matrix

    # x and y are 45x35 matrices with x = [0 0 0 ...; 1 1 1 ...; 2 2 2 ...] and y = [0 1 2 ... 34; 0 1 2 ... 34; 0 1 2 ... 34]
    x,y = np.mgrid[0:x_dim:1, 0:y_dim:1] 
    pos = np.empty(x.shape + (2,)) # add a dimension
    pos[:, :, 0] = x; pos[:, :, 1] = y # superimpose x and y
    pos = np.reshape(pos, (x.shape[0]*x.shape[1], 2)) # map the components to get all the possible matrix indices
    coords = list([(int(x[0]), int(x[1])) for x in pos]) # list of tuples of indices

    # Define the heuristic, here = distance to goal ignoring obstacles
    h = np.linalg.norm(pos - goal, axis=-1)
    h = dict(zip(coords, h))

    # Check if the start and goal are within the boundaries of the map
    for point in [start, goal]:
        assert point[0]>=0 and point[0]<occupancy_grid.shape[0],"start or end goal not contained in the map"
        assert point[1]>=0 and point[1]<occupancy_grid.shape[1],"start or end goal not contained in the map"
    
    # check if start and goal nodes correspond to free spaces
    if occupancy_grid[start[0], start[1]]:
        raise Exception('Start node is not traversable')

    if occupancy_grid[goal[0], goal[1]]:
        raise Exception('Goal node is not traversable')
    
    # get the possible movements corresponding to the selected connectivity
    if movement_type == '4N':
        movements = get_movements_4n()
    elif movement_type == '8N':
        movements = get_movements_8n()
    else:
        raise ValueError('Unknown movement')
    
    # --------------------------------------------------------------------------------------------
    # A* Algorithm implementation 
    # --------------------------------------------------------------------------------------------
    
    # The set of visited nodes that need to be (re-)expanded, i.e. for which the neighbors need to be explored
    # Initially, only the start node is known.
    openSet = [start]
    
    # The set of visited nodes that no longer need to be expanded.
    closedSet = []

    # For node n, cameFrom[n] is the node immediately preceding it on the cheapest path from start to n currently known.
    cameFrom = dict()

    # For node n, gScore[n] is the cost of the cheapest path from start to n currently known.
    gScore = dict(zip(coords, [np.inf for x in range(len(coords))]))
    gScore[start] = 0

    # For node n, fScore[n] := gScore[n] + h(n). map with default value of Infinity
    fScore = dict(zip(coords, [np.inf for x in range(len(coords))]))
    fScore[start] = h[start]

    # while there are still elements to investigate
    while openSet != []:
        
        #the node in openSet having the lowest fScore[] value
        fScore_openSet = {key:val for (key,val) in fScore.items() if key in openSet}
        current = min(fScore_openSet, key=fScore_openSet.get)
        del fScore_openSet
        
        #If the goal is reached, reconstruct and return the obtained path
        if current == goal:
            return reconstruct_path(cameFrom, current)

        openSet.remove(current)
        closedSet.append(current)
        
        #for each neighbor of current:
        for dx, dy, deltacost in movements:
            
            neighbor = (current[0]+dx, current[1]+dy)
            
            # if the node is not in the map, skip
            if (neighbor[0] >= occupancy_grid.shape[0]) or (neighbor[1] >= occupancy_grid.shape[1]) or (neighbor[0] < 0) or (neighbor[1] < 0):
                continue
            
            # if the node is occupied or has already been visited, skip
            if (occupancy_grid[neighbor[0], neighbor[1]]) or (neighbor in closedSet): 
                continue
                
            # d(current,neighbor) is the weight of the edge from current to neighbor
            # tentative_gScore is the distance from start to the neighbor through current
            tentative_gScore = gScore[current] + deltacost
            
            if neighbor not in openSet:
                openSet.append(neighbor)
                
            if tentative_gScore < gScore[neighbor]:
                # This path to neighbor is better than any previous one. Record it!
                cameFrom[neighbor] = current
                gScore[neighbor] = tentative_gScore
                fScore[neighbor] = gScore[neighbor] + h[neighbor]

    # Open set is empty but goal was never reached
    print("No path found to goal")
    return []


######################################## VISUALIZATION ########################################


########## CREATE_EMPTY_PLOT ##########

def create_empty_plot(width_pixels, length_pixels, figsize=(7, 7)):
    """
    Function to create a figure of the desired dimensions & grid
    
    :param width_pixels: dimension of the map along the x-axis
    :param length_pixels: dimension of the map along the y-axis
    :param figsize: tuple, size of the figure
    :return: the fig and ax objects.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    major_ticks_x = np.arange(0, width_pixels + 1, 5)
    minor_ticks_x = np.arange(0, width_pixels + 1, 1)
    major_ticks_y = np.arange(0, length_pixels + 1, 5)
    minor_ticks_y = np.arange(0, length_pixels + 1, 1)
    
    ax.set_xticks(major_ticks_x)
    ax.set_xticks(minor_ticks_x, minor=True)
    ax.set_yticks(major_ticks_y)
    ax.set_yticks(minor_ticks_y, minor=True)
    
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)
    
    ax.set_ylim([-1, length_pixels])
    ax.set_xlim([-1, width_pixels])
    
    ax.grid(True)
    
    return fig, ax


########## DISPLAY_OCCUPANCY_GRID ##########

def display_occupancy_grid(occupancy_grid):

    """
    Displays the occupancy grid with black cells for obstacles
    and white cells for free spaces

    INPUT:
        occupancy_grid : the occupancy grid
    
    """
    
    # Be careful with the dimensions : width_pixels corresponds to the x dimension of the matrix X x Y, 
    # not the x direction of the camera frame. Same for y
    # To get a representation of the camera frame, we should rotate the displayed window 90° clockwise

    # Creating the rectangular grid
    width_pixels = occupancy_grid.shape[0]  # Dimension along the x-axis
    length_pixels = occupancy_grid.shape[1]  # Dimension along the y-axis

    # Specify the size of the displayed area
    displayed_figsize = (5, 5)

    # Create an empty plot
    fig, ax = create_empty_plot(width_pixels, length_pixels, figsize=displayed_figsize)

    # Select the colors with which to display obstacles and free cells
    cmap = colors.ListedColormap(['white', 'black']) 

    # Displaying the map
    ax.imshow(occupancy_grid.transpose(), cmap=cmap)
    plt.title("Map: free cells in white, occupied cells in black")
    plt.show()



########## DISPLAY_OPTIMAL_PATH ##########

def display_optimal_path(start,goal,occupancy_grid):

    """
    Displays the occupancy grid with black cells for obstacles
    and white cells for free spaces as well as the optimal path
    to follow.

    INPUT:
        occupancy_grid : the occupancy grid
        start : starting point
        goal : goal point
    
    """

    # Run the A* algorithm
    path = A_Star(start, goal, occupancy_grid, movement_type="8N") 
    path = np.array(path).reshape(-1, 2).transpose()

    # Select the colors with which to display obstacles and free cells
    cmap = colors.ListedColormap(['white', 'black']) 

    # Displaying the map
    fig_astar, ax_astar = create_empty_plot(occupancy_grid.shape[0], occupancy_grid.shape[1])
    ax_astar.imshow(occupancy_grid.transpose(), cmap=cmap)

    # Plot the best path found and the list of visited nodes
    ax_astar.plot(path[0], path[1], marker="o", color = 'blue');
    ax_astar.scatter(start[0], start[1], marker="o", color = 'green', s=200);
    ax_astar.scatter(goal[0], goal[1], marker="o", color = 'purple', s=200);