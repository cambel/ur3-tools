import signal
import sys
import timeit
from pyquaternion import Quaternion
from ur_control.arm import Arm
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


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

rospy.init_node('ur3_force_control')
# read publish rate if it does exist, otherwise set publish rate
js_rate = utils.read_parameter('/joint_state_controller/publish_rate', 125.0)
arm = None


def go_to(wait=True):
    q = [1.67489, -1.56045, -1.72043,  0.2227,  1.39305,  1.33411]
    arm.set_joint_positions(position=q, wait=wait, t=3)


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
    target = transformations.pose_euler_to_quaternion(target, delta_x)
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

    alpha = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    model = ForcePositionController(
        position_pd=x_pd,
        force_pd=F_pd,
        alpha=np.diag(alpha),
        dt=0.002)    
        
    target =  arm.end_effector(rot_type='euler')
    # delta_x = np.array([0., -0.0, -0.0, 0., 0., 0.])
    # target = transformations.pose_euler_to_quaternion(target, delta_x)

    model.set_goals(position=target, force=np.zeros(6))

    arm.set_hybrid_control(model, timeout=10.0, max_force=30)
    go_to(True)


def move_back():
    cpose = arm.end_effector()
    deltax = np.array([0., 0., 0.04, 0., 0., 0.])
    cpose = transformations.pose_euler_to_quaternion(cpose, deltax, ee_rotation=True)
    arm.set_target_pose(pose=cpose, wait=True, t=3)


def grasp():
    q = [2.49022, -1.88061, -1.36663,  1.11555,  0.98978,  0.88794]
    arm.set_joint_positions(position=q, wait=True, t=3)

    q = [2.39823, -2.00304, -1.07592,  0.89976,  1.0406,  0.97818]
    arm.set_joint_positions(position=q, wait=True, t=2)

    input("Press Enter to continue...")

    q = [2.49022, -1.88061, -1.36663,  1.11555,  0.98978,  0.88794]
    arm.set_joint_positions(position=q, wait=True, t=2)


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

    args = parser.parse_args()
    driver = ROBOT_UR_RTDE_DRIVER

    global arm
    arm = Arm(ft_sensor=True, driver=driver)

    if args.move:
        go_to()
    if args.move_back:
        move_back()
    if args.grasp:
        grasp()
    if args.hybrid:
        hybrid()

    real_start_time = timeit.default_timer()

    if args.imp:
        admittance_control("discretization")

    print("real time", round(timeit.default_timer() - real_start_time, 3))


if __name__ == "__main__":
    main()
