import argparse
import rospy
from ur_control.arm import Arm

from ur_control.constants import ROBOT_GAZEBO, ROBOT_UR_MODERN_DRIVER, ROBOT_UR_RTDE_DRIVER

def main():
    """ Main function to be run. """
    parser = argparse.ArgumentParser(description='Test force control')
    parser.add_argument('--robot', action='store_true', help='for the real robot')
    parser.add_argument('--beta', action='store_true', help='for the real robot. beta driver')

    args = parser.parse_args()

    rospy.init_node('ur3_wrench')

    driver = ROBOT_GAZEBO
    if args.robot:
        driver = ROBOT_UR_MODERN_DRIVER
    elif args.beta:
        driver = ROBOT_UR_RTDE_DRIVER

    arm = Arm(ft_sensor=True, driver=driver, relative_to_ee=False)
    rospy.sleep(0.5)
    arm.set_wrench_offset(override=True)

    while not rospy.is_shutdown():
        arm.publish_wrench()

main()

