import signal
import sys
import timeit
from pyquaternion import Quaternion
from ur_control.compliant_controller import CompliantController
from ur_control.hybrid_controller import ForcePositionController
from ur_control.impedance_control import AdmittanceModel
from ur_control.constants import ROBOT_GAZEBO, ROBOT_UR_MODERN_DRIVER, ROBOT_UR_RTDE_DRIVER
from ur_control import utils, spalg, transformations
import copy
import argparse
import rospy
import numpy as np
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=np.inf)
# alias ur3_stop='rostopic pub ur3/pause std_msgs/Bool True'


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

rospy.init_node('ur3_force_control')
# read publish rate if it does exist, otherwise set publish rate
js_rate = utils.read_parameter('/joint_state_controller/publish_rate', 125.0)
arm = None


def go_to(wait=True):
    q = [1.40549, -1.58042, -1.88857,  0.3481,  1.76764,  1.65495]
    q = [1.59606, -2.03468, -1.32646,  0.21371,  1.54638,  4.76003]
    q = [-1.57771, -1.53096,  1.85189, -3.53296, -1.56431,  3.20664]
    arm.set_joint_positions(position=q, wait=wait, t=10)


def admittance_control(method="integration"):
    print("Admitance Control", method)
    e = 40
    # K = np.ones(6)*100000.0
    # M = np.ones(6)
    K = 300.0  # 3000
    M = 1.
    B = e*2*np.sqrt(K*M)
    dt = 0.002
    x0 = arm.end_effector()
    admittance_ctrl = AdmittanceModel(M, K, B, dt, method=method)

    target = x0[:]
    # target[2] = 0.03
    delta_x = np.array([0., -0.0, -0.0, 0., 0., 0.])
    target = transformations.pose_from_angular_veloticy(target, delta_x)
    print("Impedance model", admittance_ctrl)
    arm.set_impedance_control(target, admittance_ctrl, timeout=10., max_force=30, indices=[1])
    go_to(True)


def hybrid():
    arm.set_wrench_offset(override=True)

    # position PD
    Kp = 200.0
    base = np.array([1.0e-7, 1.0e-7, 1.0e-7, 1.0e-5, 1.0e-5, 1.0e-5])
    Kp_pos = base * Kp
    Kd_pos = base * np.sqrt(Kp)
    x_pd = utils.PID(Kp=Kp_pos, Kd=Kd_pos)

    # Force PD
    Kp = 100.0
    base = np.array([1.0e-6, 1.0e-6, 1.0e-6, 1.0e-4, 1.0e-4, -1.0e-3])
    Kp_force = base * Kp
    Kv_force = base * np.sqrt(Kp)
    F_pd = utils.PID(Kp=Kp_force, Kd=Kv_force)

    alpha = [1, 1, 1, 0.5, 1, 1]

    model = ForcePositionController(
        position_pd=x_pd,
        force_pd=F_pd,
        alpha=np.diag(alpha),
        dt=0.002)

    target = arm.end_effector(rot_type='euler')
    # delta_x = np.array([0., -0.0, -0.0, 0., 0., 0.])
    # target = transformations.pose_from_angular_veloticy(target, delta_x)

    model.set_goals(position=target, force=np.zeros(6))

    arm.set_hybrid_control_ik(model, timeout=10.0, max_force=30)
    go_to(True)


def fuji():
    poses = [
        [ 1.65282, -1.7068 , -1.41426, -0.59385,  1.47488, -0.04232],
        [ 1.64771, -1.79547, -1.30748, -0.61168,  1.47933, -0.04537],
        [ 1.64861, -1.94881, -1.2425 , -0.27618,  1.49168, -0.02839],
        # [1.65707, -1.85939, -1.1886, -0.62075,  1.46922, -0.0436],
        # [1.66282, -1.8796, -1.14038, -0.64842,  1.46435, -0.04079],
        # [1.66424, -1.89351, -1.26391, -0.37195,  1.45706, -0.05159],
        [1.66586, -1.94291, -1.3567, -0.02795,  1.45007, -0.06896],
        [1.66575, -1.99059, -1.37793,  0.14839,  1.44853, -0.0786],
        # [1.63688, -2.02549, -1.34375,  0.22791,  1.52438, -0.01636]
    ]

    for p in poses:
        arm.set_joint_positions(p, t=2, wait=True)
        # rospy.sleep(2)


def move_back(dist=0.02):
    cpose = arm.end_effector()
    deltax = np.array([0., 0., dist, 0., 0., 0.])
    x = transformations.pose_from_angular_veloticy(cpose, deltax, dt=1.0, ee_rotation=True)
    arm.set_target_pose(pose=x, wait=True, t=3)


def grasp():
    gears = [
        [-1.98273, -1.78595,  1.87199, -3.23067, -1.18524,  1.61247],  # outer
        [-1.91274, -1.67819,  1.78408, -3.25107, -1.27268,  1.61233],  # grasping pose
    ]

    wooden = [
        [1.54081, -1.64303, -1.92721,  0.59091,  1.59805, -0.04639],
        [1.54572, -1.78777, -1.75616,  0.56463,  1.59324, -0.04758]
    ]

    peg = [
        # [-1.56864, -1.45472,  1.93848, -3.81722, -1.57414,  3.20626],
        # [-1.56876, -1.27734,  1.79153, -3.90026, -1.57405,  3.20627],
        # [-1.56778, -1.39214,  1.69506, -3.5151 , -1.57475,  3.20598] #hard
        [-3.58255, -3.18977,  1.79615, -1.74355,  0.44003,  3.20247],
        # [-3.66734, -3.20052,  1.70873, -1.64602,  0.52471,  3.20326], #align
        [-3.68122, -3.39029,  1.71532, -1.11289,  0.56763,  2.90471], # inclined
    ]

    wrs = [
        [-2.62516, -1.89414,  1.5466 ,  0.32634, -4.16983, -0.04388],
        [-2.75658, -1.81281,  1.47038,  0.32304, -4.30122, -0.04161],
    ]

    elec1 = [
        [-2.50309, -1.86732,  1.54853,  0.29613, -4.04844, -0.0514],
        [-2.79744, -1.78833,  1.47203,  0.29787, -4.34252, -0.04446]
    ]

    elec2 = [ # similar to peg
        [-1.49464, -1.47439,  1.94913, -3.80402, -1.69579,  3.44208],
        # [-1.58174, -1.43016,  1.77241, -3.56158, -1.55581,  3.13913],
        [-1.50706, -1.33052,  1.77435, -3.7733 , -1.68375,  3.44431] # wrs2
    ]

    wrs2 = [
        [-1.49465, -1.3473 ,  2.01658, -4.1342 , -1.66235,  3.145],
        [-1.49932, -1.29612,  1.82598, -3.92481, -1.65924,  3.15111]
    ]
    q = peg

    arm.set_joint_positions(position=q[0], wait=True, t=3)

    arm.set_joint_positions(position=q[1], wait=True, t=3)

    input("Press Enter to continue...")

    move_back(0.04)

    # q = [1.86656, -0.84886, -1.86904,  0.25363,  1.38468,  1.42056]
    # arm.set_joint_positions(position=q, wait=True, t=2)


def main():
    """ Main function to be run. """
    parser = argparse.ArgumentParser(description='Test force control')
    parser.add_argument('-m', '--move', action='store_true',
                        help='move to position')
    parser.add_argument('-g', '--grasp', action='store_true',
                        help='move to position')
    parser.add_argument('-b', '--move-back', action='store_true',
                        help='move to position')
    parser.add_argument('-i', '--imp', action='store_true',
                        help='impedance')
    parser.add_argument('-f', '--hybrid', action='store_true',
                        help='rotation')
    parser.add_argument('--relative', action='store_true', help='relative to end-effector')
    parser.add_argument('--fuji', action='store_true', help='fuji-sequence')

    args = parser.parse_args()
    driver = ROBOT_UR_RTDE_DRIVER

    global arm
    arm = CompliantController(ft_sensor=True, driver=driver, ee_transform=[0, 0, 0.18, 0, 0, 0, 1])

    if args.move:
        go_to()
    if args.move_back:
        move_back()
    if args.grasp:
        grasp()
    if args.hybrid:
        hybrid()
    if args.fuji:
        fuji()

    real_start_time = timeit.default_timer()

    if args.imp:
        admittance_control("discretization")

    print("real time", round(timeit.default_timer() - real_start_time, 3))


if __name__ == "__main__":
    main()
