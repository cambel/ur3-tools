import argparse
import rospy
import numpy as np
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=np.inf)
import copy
from ur_control import utils, spalg, transformations
from ur_control.constants import ROBOT_GAZEBO, ROBOT_UR_MODERN_DRIVER, ROBOT_UR_RTDE_DRIVER
from ur_control.impedance_control import AdmittanceModel
from ur_control.arm import Arm

import timeit

rospy.init_node('ur3_force_control')
js_rate = utils.read_parameter('/joint_state_controller/publish_rate', 125.0)  # read publish rate if it does exist, otherwise set publish rate
arm = None


def go_to(wait=True):
    q = [0.171, -0.912,  2.21, -2.866, -1.571,  0.169]
    q = [-0.16 , -0.832,  1.863, -0.751, -0.213, -1.857]
    q = np.deg2rad([90.0, 0.0, 45.0, 0.0, -90.0, 0.0])
    q = np.deg2rad([118.35, -92.94, -94.20, 3.53, 57.27, 78.29]) #2.06563, -1.62208, -1.64403,  0.06159,  0.9995 ,  1.36647
    q = [1.96414, -1.9198 , -1.3648 ,  0.08443,  1.10098,  1.35949]
    q = [1.89951, -1.89104, -1.31663,  0.00926,  1.16537,  1.35538]
    q = [1.68605, -2.11149, -1.01201, -0.02208,  1.48276,  1.58753]

    arm.set_joint_positions(position=q, wait=wait, t=3)


def admittance_control(method="integration"):
    print("Admitance Control", method)
    # arm.set_wrench_offset(override=True)
    e = 40
    # K = np.ones(6)*100000.0
    # M = np.ones(6)
    K = 300.0 #3000
    M = 1.
    B = e*2*np.sqrt(K*M)
    dt = 0.002
    x0 = arm.end_effector()
    admittance_ctrl = AdmittanceModel(M, K, B, dt, method=method)

    target = x0[:]
    # target[2] = 0.03
    delta_x = np.array([0., -0.02, -0.0, 0., 0., 0.])
    target = transformations.pose_euler_to_quaternion(target, delta_x)
    print("Impedance model", admittance_ctrl)
    arm.set_impedance_control(target, admittance_ctrl, timeout=10., max_force=3000, indices=[1])
    # data = actuate(target, admittance_ctrl, 10, method, data=[])
    # np.save("/root/dev/tools/"+method, np.array(data))
    go_to(True)

def rotation():
    target = [-0.08855, -0.44407,  0.44978,  0.51342,  0.38084, -0.29487, 0.71022]
    pass

def main():
    """ Main function to be run. """
    parser = argparse.ArgumentParser(description='Test force control')
    parser.add_argument('-m', '--move', action='store_true',
                        help='move to position')
    parser.add_argument('-i', '--imp', action='store_true',
                        help='impedance')
    parser.add_argument('-a', '--disc_adm', action='store_true',
                        help='discrete admittance')
    parser.add_argument('-b', '--int_adm', action='store_true',
                        help='integration admittance')
    parser.add_argument('-r', '--rotation', action='store_true',
                        help='Rotation slerp')
    parser.add_argument('--robot', action='store_true', help='for the real robot')
    parser.add_argument('--relative', action='store_true', help='relative to end-effector')
    parser.add_argument('--beta', action='store_true', help='for the real robot. beta driver')

    args = parser.parse_args()

    driver = ROBOT_GAZEBO
    if args.robot:
        driver = ROBOT_UR_MODERN_DRIVER
    elif args.beta:
        driver = ROBOT_UR_RTDE_DRIVER

    global arm
    arm = Arm(ft_sensor=True, driver=driver)

    if args.move:
        go_to()
    real_start_time = timeit.default_timer()
    ros_start_time = rospy.get_time()

    if args.imp:
        admittance_control("traditional")
    if args.disc_adm:
        admittance_control("discretization")
    if args.int_adm:
        admittance_control("integration")
    if args.rotation:
        rotation()

    print("real time", round(timeit.default_timer() - real_start_time, 3))
    print("ros time", round(rospy.get_time() - ros_start_time, 3))


if __name__ == "__main__":
    main()
