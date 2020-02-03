import argparse
import rospy
import numpy as np
import copy
from ur_control import utils, spalg
from ur_control.constants import ROBOT_GAZEBO, ROBOT_UR_MODERN_DRIVER, ROBOT_UR_RTDE_DRIVER
from ur_control.force_controller import ForceController, ForcePositionController

from ur_control.arm import Arm

import matplotlib.pyplot as plt

from gps.agent.utils.gazebo_spawner import GazeboModels
from gps.agent.utils.model import Model

import timeit

rospy.init_node('ur3_force_control')

# read publish rate if it does exist, otherwise set publish rate
js_rate = utils.read_parameter('/joint_state_controller/publish_rate', 125.0)
T = 1. / js_rate
rate = rospy.Rate(js_rate)
print("dt", (T))
extra_ee = [-0.000, 0.000, -0.1481, np.pi/4, 0.000, np.pi/4, 0.000]
wrench_offset = None

color_log = utils.TextColors()

objpose = [
    [0.506381,  0.10000, 0.774062],
    [0.508031, -0.10125, 0.774062],
    [0.227448, 0.464739, 1.03],  # board
    [0.2518, 0.486, 1.060051],
    [0.287890, 0.293387, 1.03],  # peg
]

model_names = ["peg_board", "wooden_peg"]
models = [[Model(model_names[0], objpose[0]), Model(model_names[1], objpose[1])]]


def go_to():
    q = np.deg2rad([90.0, -30.0, -60.0, 0.0, 90.0, 0.0])
    q = [0.17, -0.974,  2.211, -2.808, -1.571,  0.169]

    arm.set_joint_positions(position=q, wait=True, t=3)
    rospy.sleep(0.1)


def hover():
    q = [0.973, -1.402, 1.401, -1.584, -1.575, -0.592]
    q = [0.973, -1.401, 1.427, -1.609, -1.575, -0.592]

    arm.set_joint_positions(position=q, wait=True, t=2)
    rospy.sleep(0.1)


def place_target():
    from gps.agent.utils.basic_models import SPHERE
    sphere = SPHERE % ("target", "0.025", "GreenTransparent")
    model_names = ["target"]
    objpose = [[0.0131,  0.4019,  0.3026]]
    objpose = [[-0.13101034,  0.37818616,  0.50852045]]

    models = [[Model(model_names[0], objpose[0], file_type='string', string_model=sphere, reference_frame="base_link")]]
    GazeboModels(models, 'ur3_gazebo')


def place_models():
    model_names = ["multi_peg_board"]
    model_names = ["simple_peg_board"]
    objpose = [[0.217947,  0.367654, 0.75], None]
    objpose = [[0.402628, 0.275193, 0.807442], [0, -0.3556907, 0, 0.9346037]]
    objpose = [[0.198303, 0.244189, 0.75], None]
    objpose = [[-0.381961, -0.250909, 0.816082], None] #experiments so far
    objpose = [[-0.45, -0.20, 0.86], [0, 0.1986693, 0, 0.9800666]]
    models = [[Model(model_names[0], objpose[0], orientation=objpose[1])]]
    GazeboModels(models, 'ur3_gazebo')


def grasp_cube():
    gripper_controller.open()

    # hover over white cube
    # # grasp cube
    q1 = [1.04659, -1.16732, 0.94036, -1.36649, -1.57582, -0.51876]
    q2 = [1.04617, -1.18619, 1.17739, -1.58091, -1.57577, -0.5192]

    arm.set_joint_positions(position=q1, wait=True, t=2)
    arm.set_joint_positions(position=q2, wait=True, t=2)

    GazeboModels(models, 'ur3_gazebo')
    gripper_controller.command(0.018)
    gripper_controller.wait()
    rospy.sleep(1)
    gripper_controller.grab('{0}::link'.format('wooden_peg'))

    # Hover with graspped cube
    hover()


def detach_cube():
    gripper_controller.open()
    gripper_controller.release('{0}::link'.format('wooden_peg'))


def sliding(target=[0., 0.0, 0.0, 0, 0, 0]):
    arm.set_wrench_offset(override=True)

    dt = 1. / js_rate     # time step
    duration = 20.0

    # position PD
    # proportional gain of position controller
    Kp_pos = np.array([2e-3] * 3 + [2e-8]*3)
    # derivative gain of position controller
    Kd_pos = np.array([3.2e-05] * 3 + [1e-8]*3)
    x_pd = utils.PID(Kp=Kp_pos, Kd=Kd_pos)

    # Force PD
    Fr = np.array([0., 0., 10.0, 0, 0, 0])       # Force of reference
    Kp = 10
    Kp_force = np.array([1e-8*Kp]*6)   # proportional gain of force controller
    Kv_force = np.array([5e-9*np.sqrt(Kp)]*6)     # derivative gain of force controller
    F_pd = utils.PID(Kp=Kp_force, Kd=Kv_force)

    hybrid_controller = ForcePositionController(
        x_PID=x_pd,
        F_PID=F_pd,
        xF_tradeoff=np.diag([1., 1., 0., 1, 1, 1]))

    _target = arm.end_effector(rot_type='euler')
    _target += target
    hybrid_controller.set_goals(position=_target)
    hybrid_controller.set_goals(force=Fr)
    hybrid_controller.start(dt=dt,
                            timeout=duration,
                            controller=arm)


def force_control(target=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]):
    real_start_time = timeit.default_timer()
    ros_start_time = rospy.get_time()

    dt = 1. / js_rate     # time step 0.002
    duration = 10.
    print("Force controller params", "dt", dt, "duration", duration)

    controller = hybrid_controller(alpha=[1., 1., 0., 1., 1., 1.],
                                   position_kp=100,
                                   force_kp=100)

    _target = arm.end_effector(rot_type='euler')
    print("op2", np.round(_target, 5).tolist())
    _target += target
    Fr = np.array([0., 0., 50., 0., 0., 0.])
    controller.set_goals(position=_target, force=Fr)
    controller.start(dt=0.002,
                     timeout=duration,
                     controller=arm)

    print("real time", round(timeit.default_timer() - real_start_time, 3))
    print("ros time", (rospy.get_time() - ros_start_time))


def hybrid_controller(alpha=[1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
                      position_kp=20,
                      force_kp=5):
    arm.set_wrench_offset(override=True)

    # position PD
    Kp = position_kp
    Kp_pos = np.array([Kp*1e-6] * 6)
    # derivative gain of position controller
    Kd_pos = np.array([np.sqrt(Kp)*1e-5] * 6)
    x_pd = utils.PID(Kp=Kp_pos, Kd=Kd_pos)

    # Force PD
    Kp = force_kp
    Kp_force = np.array([1e-10*Kp]*6)   # proportional gain of force controller
    Kv_force = np.array([1e-10*np.sqrt(Kp)]*6)     # derivative gain of force controller
    F_pd = utils.PID(Kp=Kp_force, Kd=Kv_force)

    return ForcePositionController(
        position_pd=x_pd,
        force_pd=F_pd,
        alpha=np.diag(alpha))


def tune_hybrid_controller():
    duration = 10.0
    t_interval = 0.05
    n_intervals = int(duration/t_interval)

    controller = hybrid_controller(alpha=[1., 1., 0., 1., 1., 1.],
                                   position_kp=100,
                                   force_kp=100)

    kp = np.linspace(100, 100, n_intervals)
    kd = np.linspace(100, 100, n_intervals)

    target_pose = arm.end_effector(rot_type='euler')
    target_force = np.array([0., 0., 10., 0., 0., 0.])
    for i in range(n_intervals):
        print("kp", kp[i], "kd", kd[i])
        controller.position_pd.reset()
        controller.force_pd.reset()
        controller.set_goals(position=target_pose, force=target_force)
        controller.start(dt=0.001,
                         timeout=t_interval,
                         controller=arm)


def simple_trajectory(controller='hybrid'):
    real_start_time = timeit.default_timer()
    ros_start_time = rospy.get_time()
    q = [0.154, -0.98,  2.05, -2.648, -1.571,  0.153]
    arm.set_joint_positions(position=q, wait=True, t=0.5)

    duration = 10.0
    angle_resolution = 1
    d_angle = np.deg2rad(angle_resolution)
    angle = 0
    radius = 0.05
    x_center = 0.25
    y_center = 0.18
    n_points = int(360/angle_resolution)
    t_step = (duration/n_points)
    print(t_step)

    if controller == 'hybrid':
        current_ee = arm.end_effector(rot_type='euler')
        h_controller = hybrid_controller(position_kp=1000)
    elif controller == 'flex':
        current_ee = arm.end_effector(ee_link='ee_link', ikfast=True)
    elif controller == 'traj_controller':
        traj = arm.joint_traj_controller
    else:
        raise Exception("invalid controller")

    q = None
    traj_points = []

    for i in range(n_points):
        ee_x = x_center + radius*np.cos(angle)
        ee_y = y_center + radius*np.sin(angle)
        ee = None
        ee = copy.copy(current_ee)
        ee[:2] = [ee_x, ee_y]
        q = arm.solve_ik(ee, q)

        if controller == 'hybrid':
            traj_points.append(ee)
        elif controller == 'flex':
            traj_points.append(q)
        elif controller == 'traj_controller':
            arm.joint_traj_controller.add_point(positions=q, time=((i+1)*t_step))

        angle += d_angle

    print("traj num points", arm.joint_traj_controller.get_num_points())
    if controller == 'traj_controller':
        arm.joint_traj_controller.start(delay=0., wait=True)
        arm.joint_traj_controller.clear_points()
        return

    rate = rospy.Rate(1/t_step)
    dt = 1. / js_rate     # time step
    for p in traj_points:
        if controller == 'hybrid':
            h_controller.position_pd.reset()
            h_controller.force_pd.reset()
            h_controller.set_goals(position=p, force=np.zeros(6))
            h_controller.start(dt=0.002,
                               timeout=t_step,
                               controller=arm)
        elif controller == 'flex':
            arm.set_joint_positions_flex(p, t_step)
            rate.sleep()

    print("real time", round(timeit.default_timer() - real_start_time, 3))
    print("ros time", round(rospy.get_time() - ros_start_time, 3))


def inverse_kinematics():
    ee = [0.266, 0.178, 0.061, 3.141, 0., 1.57]
    print("1#", np.round(arm.solve_ik(ee), 3))
    print("2#", np.round(arm.solve_ik(ee), 3))
    print("3#", np.round(arm.solve_ik(ee), 3))


def main():
    """ Main function to be run. """
    parser = argparse.ArgumentParser(description='Test force control')
    parser.add_argument('-g', '--grasp', action='store_true',
                        help='grasp cube')
    parser.add_argument('-d', '--move_down', action='store_true',
                        help='move until reach table')
    parser.add_argument('-s', '--sliding', action='store_true',
                        help='Slide in the table')
    parser.add_argument('-u', '--hover', action='store_true',
                        help='Hover over table')
    parser.add_argument('-r', '--detach', action='store_true',
                        help='Detach cube')
    parser.add_argument('-f', '--force', action='store_true',
                        help='Force control test')
    parser.add_argument('-m', '--move', action='store_true',
                        help='move to position')
    parser.add_argument('-c', '--circle', action='store_true',
                        help='move in a circle trajectory')
    parser.add_argument('--tune', action='store_true', help='tune hybrid control manually')
    parser.add_argument('--robot', action='store_true', help='for the real robot')
    parser.add_argument('--beta', action='store_true', help='for the real robot. beta driver')
    parser.add_argument('--place', action='store_true',
                        help='Place models')
    parser.add_argument('--target', action='store_true',
                        help='Place targets')
    parser.add_argument('--gripper', action='store_true',
                        help='load gripper controller')

    args = parser.parse_args()

    driver = ROBOT_GAZEBO
    if args.robot:
        driver = ROBOT_UR_MODERN_DRIVER
    elif args.beta:
        driver = ROBOT_UR_RTDE_DRIVER

    global arm
    global gripper_controller
    arm = Arm(ft_sensor=True, driver=driver)

    if args.gripper:
        from ur_control.controllers import GripperController
        gripper_controller = GripperController(attach_plugin=True)

    if args.hover:
        hover()
    if args.move:
        go_to()
    if args.detach:
        detach_cube()
    if args.grasp:
        grasp_cube()
    if args.sliding:
        sliding()
    if args.force:
        force_control()
    if args.place:
        place_models()
    if args.target:
        place_target()
    if args.circle:
        simple_trajectory()
    if args.tune:
        tune_hybrid_controller()
    # inverse_kinematics()


if __name__ == "__main__":
    main()
