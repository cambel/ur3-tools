import argparse
import rospy
import numpy as np
import copy
from ur_control import utils, spalg
from ur_control.constants import ROBOT_GAZEBO, ROBOT_UR_MODERN_DRIVER, ROBOT_UR_RTDE_DRIVER
#from ur_control.hybrid_controller import ForceController, ForcePositionController
from ur_control.hybrid_controller import ForcePositionController

from ur_control.arm import Arm

import matplotlib.pyplot as plt
#from gps.agent.ur.arm import Arm
#from gps.agent.ur.force_controller import ForceController, ForcePositionController
#import ur3_kinematics.arm as ur3_arm

from ur_gazebo.gazebo_spawner import GazeboModels
from ur_gazebo.model import Model

import timeit

rospy.init_node('ur3_force_control')

js_rate = utils.read_parameter('/joint_state_controller/publish_rate', 125.0)  # read publish rate if it does exist, otherwise set publish rate
T = 1. / js_rate
rate = rospy.Rate(js_rate)
#print "dt", (T)
extra_ee = [-0.000, 0.000, -0.1481, np.pi/4, 0.000, np.pi/4, 0.000]
wrench_offset = None

color_log = utils.TextColors()

"""
objpose = [
    [0.506381,  0.10000, 0.774062],
    [0.508031, -0.10125, 0.774062],
    [0.227448, 0.464739, 1.03],  # board
    [0.2518, 0.486, 1.060051],
    [0.287890, 0.293387, 1.03],  # peg
]
"""

objpose = [
    [0.402, -0.104, 0.845], #position of the board
    [0.402, -0.107, 0.89], # position of the peg
    [-0.24749, 0.0, 0.675], #board
    [0.2518, 0.486, 1.060051],
    [0.287890, 0.293387, 0.5], #peg
]

"""
model_names = ["peg_board", "wooden_peg"]
models = [[Model(model_names[0], objpose[0]), Model(model_names[1], objpose[1])]]
"""

model_names = ["ur_peg1950"]
models = [[Model(model_names[0], objpose[0])]]


def go_to():
    q = np.deg2rad([90.0, -51.0, 84.0, 237.0, -90.0, 0.0])
    q = np.deg2rad([90.0, -30.0, -60.0, 0.0, 90.0, 0.0])

    # test of force control
    q = np.deg2rad([90.0, -46.0, 86.0, 230.0, -90.0, 0.0])

    #peg in hole
    q = np.deg2rad([33.47, -47.82, 101.88, -142.03, -90.85, 0.80])  # goal (x,y,z) 0.250, 0.300, 0.157 Quaternion [-0.633, 0.340, -0.338, 0.607]
    q = [0.5794, -0.8153,  1.8179, -2.5778, -1.5696,  0.0097]
    q = [1.4489, -0.8151,  1.818, -2.5777, -1.5697,  1.57]
    q = [0.4916, -0.75544,  1.67592, -2.49256, -1.57016,  0.00464]
    q = [0.5339, -0.7641,  1.6964, -2.5049, -1.57,  0.5334]
    q = [1.57082, -0.16536, -0.87687,  0.00019,  1.57056,  0.00004]
    q = np.deg2rad([90.0, -30.0, -60.0, 0.0, 90.0, 0.0])
    q = [0.1697 , -1.03397,  2.20931, -2.74585, -1.57055,  0.16935]

    arm.set_joint_positions(position=q, wait=True, t=3)
    rospy.sleep(0.1)


def hover():
    q = [0.973, -1.402, 1.401, -1.584, -1.575, -0.592]
    q = [0.973, -1.401, 1.427, -1.609, -1.575, -0.592]

    arm.set_joint_positions(position=q, wait=True, t=2)
    rospy.sleep(0.1)


def place_target():
    from ur_gazebo.basic_models import SPHERE
    sphere = SPHERE % ("target", "0.025", "GreenTransparent")
    model_names = ["target"]
    objpose = [[0.0131,  0.4019,  0.3026]]
    objpose = [[-0.13101034,  0.37818616,  0.50852045]]

    models = [[Model(model_names[0], objpose[0], file_type='string', string_model=sphere, reference_frame="base_link")]]
    GazeboModels(models, 'ur3_gazebo')


def place_models():
    model_names = ["multi_peg_board"]
    objpose = [[0.198303, 0.244189, 0.75]]
    objpose = [[0.217947,  0.367654, 0.75]]
    objpose = [[0.215225, 0.225081, 0.75]]
    models = [[Model(model_names[0], objpose[0])]]
    GazeboModels(models, 'ur3_gazebo')


def grasp_cube():
    gripper_controller.open()

    # hover over white cube
    # # grasp cube
    #q1 = [1.04659, -1.16732, 0.94036, -1.36649, -1.57582, -0.51876]
    #q2 = [1.04617, -1.18619, 1.17739, -1.58091, -1.57577, -0.5192]

    q1 = [1.04659, -1.16732, 0.94036, -1.36649, -1.57582, -0.51876]
    #q2 = [1.04, -1.15, 1.4, -1.85, -1.57577, -0.5192]
    # q2 = [ 1.0466 , -1.05006,  1.57133, -2.12025, -1.57701, -0.51257] #
    q2 = [ 1.04001, -1.06238,  1.55698, -2.07659, -1.57   , -0.51   ]

    arm.set_joint_positions(position=q1, wait=True, t=2)
    arm.set_joint_positions(position=q2, wait=True, t=2)

    GazeboModels(models, 'ur3_gazebo')
    gripper_controller.command(0.018)
    gripper_controller.wait()
    rospy.sleep(1)
    #gripper_controller.grab('{0}::link'.format('wooden_peg'))
    gripper_controller.grab('{0}::link'.format('ur_peg1950'))

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
    Fr = np.array([0., 0., 5.0, 0, 0, 0])       # Force of reference
    Kp = 25
    Kp_force = np.array([1e-8*Kp]*6)   # proportional gain of force controller
    Kv_force = np.array([1e-8*np.sqrt(Kp)]*6)     # derivative gain of force controller
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


def force_control(target=[0.0, 0.0, 0.0, 0.0, 0.0, np.pi/8]):
    arm.set_wrench_offset(override=True)

    dt = 1. / js_rate     # time step
    duration = 60.

    # position PD
    Kp = 200
    Kp_pos = np.array([Kp*1e-5] * 3 + [Kp*1e-6, Kp*1e-6, Kp*1e-6])
    # derivative gain of position controller
    Kd_pos = np.array([np.sqrt(Kp)*1e-5] * 3 + [np.sqrt(Kp)*1e-6]*3)
    x_pd = utils.PID(Kp=Kp_pos, Kd=Kd_pos)

    # Force PD
    Fr = np.array([0., 0., 0., 0, 0, 0])       # Force of reference
    Kp = 5
    Kp_force = np.array([1e-8*Kp]*5 + [1e-8*Kp])   # proportional gain of force controller
    Kv_force = np.array([1e-8*np.sqrt(Kp)]*5 + [1e-8*np.sqrt(Kp)])     # derivative gain of force controller
    F_pd = utils.PID(Kp=Kp_force, Kd=Kv_force)

    hybrid_controller = ForcePositionController(
        x_PID=x_pd,
        F_PID=F_pd,
        xF_tradeoff=np.diag([0.8, .8, 0.5, 1., 1., 0.5]))

    # _target = arm._task_space()
    _target = arm.end_effector(rot_type='euler')
    #print "op2", np.round(_target, 5).tolist()
    _target += target
    hybrid_controller.set_goals(position=_target)
    hybrid_controller.set_goals(force=Fr)
    hybrid_controller.start(dt=dt,
                            timeout=duration,
                            controller=arm
                            )
    # print "op2", np.round(arm._task_space(), 5).tolist()
    #print "op2", np.round(arm.end_effector(rot_type='euler')-_target, 5).tolist()


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
    arm = Arm(ft_sensor=True, driver=driver, ee_transform=[0, 0, 0.05, 0, 0, 0, 1])

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


if __name__ == "__main__":
    main()
