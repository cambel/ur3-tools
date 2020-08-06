import argparse
import rospy
import numpy as np
from ur_control.compliant_controller import CompliantController

from ur_control.constants import ROBOT_GAZEBO, ROBOT_UR_MODERN_DRIVER, ROBOT_UR_RTDE_DRIVER

def main():
    """ Main function to be run. """
    parser = argparse.ArgumentParser(description='Test force control')
    parser.add_argument('--robot', action='store_true', help='for the real robot')
    parser.add_argument('--beta', action='store_true', help='for the real robot. beta driver')
    parser.add_argument('--record', action='store_true', help='record ft data')

    args = parser.parse_args()

    rospy.init_node('ur3_wrench')

    driver = ROBOT_GAZEBO
    if args.robot:
        driver = ROBOT_UR_MODERN_DRIVER
    elif args.beta:
        driver = ROBOT_UR_RTDE_DRIVER

    arm = CompliantController(ft_sensor=True, driver=driver, relative_to_ee=False)
    rospy.sleep(0.5)
    arm.set_wrench_offset(override=False)

    cnt = 0
    offset_cnt = 0
    data = []
    ft_filename = "ft_data.npy"
    while not rospy.is_shutdown():
        arm.publish_wrench()

        if not args.record:
            continue

        if offset_cnt > 500:
            arm.set_wrench_offset(False)
            offset_cnt = 0
        offset_cnt += 1

        if cnt < 500:
            data.append([arm.get_ee_wrench().tolist(), arm.end_effector().tolist()])
            cnt+=1
        else:
            try:
                ldata = np.load(ft_filename, allow_pickle=True)
                ldata = ldata.tolist()
                ldata += data
                np.save(ft_filename, ldata)
                data = []
                cnt = 0
            except IOError:
                np.save(ft_filename, data)

main()