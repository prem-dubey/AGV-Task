import cv2
import numpy as np
from ai2thor.controller import Controller
import random
import time

# Initializing AI2-THOR controller 
controller = Controller(
    scene="FloorPlan3",
    gridSize=0.25,
    agentMode="locobot",
    visibilityDistance=1.5,
    rotateStepDegrees=90
)

# To Get all of the possible reachable positions 
positions = controller.step(action="GetReachablePositions").metadata["actionReturn"]

# Setting the target position randomly ( for now it is fixed do it later )
target = {'x': 1.0, 'y': 1.1232054233551025, 'z': 0.25}

# Feature Detection Parameters (Shi-Tomasi) for goodFeaturesToTrack()
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
'''
    Parameters : 
    image: The input grayscale image for feature detection.
    maxCorners : maximum no of corner it detects .
    qualityLevel : Minimum accepted quality of a corner 
    minDistance: Minimum distance between detected corners to avoid clustering.
    mask: Optional region of interest (ROI) where corners should be detected.
    blockSize: Window size for computing gradients in corner detection.

    returns : an array of detected corner points in the form of a NumPy array'
'''

# Optical Flow Parameters (Lucas-Kanade)
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)) 
'''
parameter : 
prevImg : previous grayscale image 
nextImg: next grayscale image
prevPts: Array of points (keypoints) to track from prevImg
nextPts: Output array of tracked points in nextImg.
winSize: Size of the search window at each pyramid level .
maxLevel : No of pyramid levels for high movement tracking 
criteria: Stopping criteria ((type, max_iter, epsilon) .
Returns :
nextPts: A NumPy array of shape (N, 1, 2) containing the new (x, y) positions of the tracked points.
status: A NumPy array of shape (N, 1), where 1 means the point was successfully tracked, and 0 means tracking failed.
err: A NumPy array of shape (N, 1) containing the tracking error for each point (lower is better).
'''

# Function to capture frame and convert it to grayscale 
def get_grayscale_frame():
    event = controller.last_event
    frame = event.frame
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return gray

# Function to get current agent position 
def get_agent_position():
    event = controller.last_event
    return event.metadata["agent"]["position"]

# Function to compute distance between two points 
def compute_distance(pos1, pos2):
    return np.sqrt((pos1["x"] - pos2["x"]) ** 2 + (pos1["z"] - pos2["z"]) ** 2)

# Compute total force (Attractive + Repulsive) 
def compute_total_force(agent_pos, obstacle_pos, goal_pos, alpha=0.2, gamma=1):
    d_goal = np.linalg.norm([goal_pos["x"] - agent_pos["x"], goal_pos["z"] - agent_pos["z"]]) # this gives us the distance between bot and the target 
    F_att = alpha * d_goal 

    if not obstacle_pos:
        d_obs = float('inf') # if no obstacles d_obs = infinity 
    else:
        d_obs = min(
            np.linalg.norm([obs["position"]["x"] - agent_pos["x"], obs["position"]["z"] - agent_pos["z"]]) 
            for obs in obstacle_pos
        ) # calculating the distance between the bot and the nearest object 
    F_rep = gamma / (d_obs + 1e-5) 

    return F_att - F_rep 

# Compute Fx and Fy for deciding the final movement direction 
def compute_force_components(agent_pos, obstacle_pos, goal_pos):
    F = compute_total_force(agent_pos, obstacle_pos, goal_pos)

    theta = np.arctan2(goal_pos["z"] - agent_pos["z"], goal_pos["x"] - agent_pos["x"]) # Computing theta 

    F_x = F * np.cos(theta)
    F_y = F * np.sin(theta)

    return F_x, F_y

# Capture the first frame
old_gray = get_grayscale_frame()
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Initialize stuck detection variables
prev_position = get_agent_position()
stuck_counter = 0
stuck_threshold = 5  # Number of iterations before considering stuck

while True:
    # we can put time.sleep(0.2) to see more clearly 

    agent_pos = get_agent_position() # getting current position 
    event = controller.last_event 
    objects = event.metadata["objects"] # storing data of all of the objects
    
    total_force = compute_total_force(agent_pos, objects, target) # computing total force 
    F_x, F_y = compute_force_components(agent_pos, objects, target) # Finding F_x and F_y 

    # Move agent slightly forward
    controller.step(action="MoveAhead") # Moving ahead to get new grayscale 
    new_gray = get_grayscale_frame()

    if p0 is None or len(p0) == 0:
        p0 = cv2.goodFeaturesToTrack(new_gray, mask=None, **feature_params) # For Shi-Tomasi detector 
        old_gray = new_gray.copy()
        continue

    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params) # Calculating optical flow 

    if p1 is None or st is None or len(p1) == 0:
        p0 = cv2.goodFeaturesToTrack(new_gray, mask=None, **feature_params)
        old_gray = new_gray.copy()
        continue

    good_new = p1[st == 1] # getting all the points sucessfully tracked 
    good_old = p0[st == 1] # getting all the points sucessfully tracked previously 

    if len(good_new) == 0 or len(good_old) == 0:
        p0 = cv2.goodFeaturesToTrack(new_gray, mask=None, **feature_params)
        old_gray = new_gray.copy()
        continue

    flow_vectors = good_new - good_old # calculating the motion flow vectors 
    dx, dy = np.mean(flow_vectors, axis=0)

    # Stuck detection: check if the bot has moved
    distance_moved = compute_distance(prev_position, agent_pos)
    
    if distance_moved < 0.05:  # Small threshold, adjust as needed
        stuck_counter += 1
    else:
        stuck_counter = 0  # Reset counter if the bot moves

    prev_position = agent_pos  # Update previous position

    # Determine movement direction if the bot is stuck determine direction randomly 
    if stuck_counter >= stuck_threshold:
        direction = random.choice(["RotateLeft", "RotateRight","MoveBack"])
        controller.step(action="MoveBack") 
        controller.step(action="RotateRight")
        stuck_counter = 0  # Reset stuck counter
        print("⚠ Stuck detected! Changing direction randomly.") 
    else: # Movement logic expalined in the documentation 
        if abs(dx) + abs(dy) > abs(F_x) + abs(F_y):
            if abs(dx) > abs(dy):
                direction = "RotateRight" if dx > 0 else "RotateLeft"
            else:
                direction = "MoveAhead" if dy > 0 else "MoveBack"
        else:
            if abs(F_x) > abs(F_y):
                direction = "RotateRight" if F_x > 0 else "RotateLeft"
            else:
                direction = "MoveAhead" if F_y > 0 else "MoveBack"

    print(f"📌 Motion: dx={dx:.2f}, dy={dy:.2f} → {direction} | Motion F_x={F_x:.2f} , F_y={F_y:.2f} | Distance to Target: {compute_distance(agent_pos,target):.2f}")

    controller.step(action=direction) # updating the bot position 
    event = controller.last_event

    old_gray = new_gray.copy()
    p0 = good_new.reshape(-1, 1, 2) # updating the p0

    # When target reached break out of the loop 
    if compute_distance(agent_pos,target) <= 0.5 :
        print(f" !!!Target Reached!!!  📌 Motion: dx={dx:.2f}, dy={dy:.2f} → {direction} | Motion F_x={F_x:.2f} , F_y={F_y:.2f} | Distance to Target: {compute_distance(agent_pos,target):.2f} ")
        break

