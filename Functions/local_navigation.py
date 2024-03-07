from Functions.vision import *
from Functions.A import *
from Functions.motor import *


###################### OBSTACLE_DETTECTION ######################
viewobsTRH=1000

async def obstacle(node, client)->bool:
    """
    Return true when proximity sensors detecte something
    """
    #delay to get proximity sensor values
    for i in range(3):
        list_prox=list(node["prox.horizontal"])
        await client.sleep(0.01)
    if((list_prox[0]> viewobsTRH) | (list_prox[1]> viewobsTRH) | (list_prox[3]> viewobsTRH) | (list_prox[4]> viewobsTRH)):
        return True
    else:
        return False
    
    
####################### OBSTACLE_AVOIDANCE #####################
memory_spr=[]
memory_spl=[]
speed0 = 100
obst_gain = [2, 3, -1, -3, -2]
nbr_prox_sensor = 5

async def obstacle_avoidance(node, client):
    """
    First it avoides the obstacle, second stores the speeds assigned to the left and right wheels and 
    invertes them after 2 seconds to achieve the necessary reorientation.
    """
    while await obstacle(node, client):
        # normal speed
        speed_left = 100
        speed_right = 100
        #delay to get proximity values
        for i in range(3):
            list_prox=list(node["prox.horizontal"])
            await client.sleep(0.01) 
        # repulsive filed  local obstacles
        for i in range(nbr_prox_sensor):
            speed_left += list_prox[i] * obst_gain[i] // 100
            speed_right += + list_prox[i] * obst_gain[4 - i] // 100
        # motor control
        motor_left_target = int(speed_left)
        motor_right_target = int(speed_right)
        
        await node.set_variables(motors(motor_left_target, motor_right_target))
        # delay between updates
        await client.sleep(0.25)
    await node.set_variables(motors(speed0,speed0))
    await client.sleep(2)
    print('wait 2 sec')
