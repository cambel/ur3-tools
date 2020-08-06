import os
from ur3e_openai.robot_envs.utils import get_conical_helix_trajectory
import numpy as np

init_pose = [ 0.01880389, -0.34560648,  0.40815178] 
center = [-0.00319718, -0.37576394,  0.45137436] 
orientation_goal =[-0.0021968,  -0.66844382,  0.74374149, -0.00516056] 

euclidean_dist = np.linalg.norm(center-init_pose)
base_trajectory = get_conical_helix_trajectory(center, euclidean_dist, euclidean_dist, 250, 6.0)

os.system('gz marker -a -t "points" -f 120')
