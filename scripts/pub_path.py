#!/usr/bin/env python
import rospy
import numpy as np
from pyquaternion import Quaternion as Quat
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Pose, Quaternion
from std_msgs.msg import Header
from ur_control.transformations import quaternion_from_euler
# from ur_control.compliant_controller import CompliantController
from ur3e_openai.robot_envs.utils import get_conical_helix_trajectory

path = Path()

inital = [-0.0012867963980835793, -0.35302556809186536, 0.4085084665890285, -0.0498093375695251, -0.6912765989420319, 0.7208648458793617, 0.0030931571809757674]
target = [-0.003197180674706364, -0.37576393689660575, 0.45137435818768146, -0.0021967960033876664, -0.6684438218043409, 0.7437414872148852, -0.005160559496475987] 
target_q = Quat(np.roll(target[3:], 1))

steps = 300

p2 = np.zeros(3)
p1 = target_q.rotate(np.array(inital[:3]) - target[:3])

traj = get_conical_helix_trajectory(p1, p2, steps, 3.0)

traj = np.apply_along_axis(target_q.rotate,1,traj)
traj = traj + target[:3]

rospy.init_node('path_node')

path = Path()
h = Header()
h.stamp = rospy.Time.now()
h.frame_id = "base_link"
path.header = h
for t in traj:
    p = PoseStamped()
    p.pose.position.x = t[0]
    p.pose.position.y = t[1]
    p.pose.position.z = t[2]
    p.pose.orientation = Quaternion()
    path.poses.append(p)

path_pub = rospy.Publisher('/path', Path, queue_size=10)
while True:
    path_pub.publish(path)
    rospy.sleep(1)

if __name__ == '__main__':
    rospy.spin()