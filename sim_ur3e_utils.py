import argparse
import rospy
import numpy as np
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=np.inf)
import copy
from ur_control import utils, spalg, transformations
from ur_control.constants import ROBOT_GAZEBO, ROBOT_UR_MODERN_DRIVER, ROBOT_UR_RTDE_DRIVER
from ur_control.impedance_control import AdmittanceModel
from ur_control.compliant_controller import CompliantController
from pyquaternion import Quaternion
import timeit


rospy.init_node('ur3_force_control')
js_rate = utils.read_parameter('/joint_state_controller/publish_rate', 125.0)  # read publish rate if it does exist, otherwise set publish rate
arm = None


def go_to(wait=True):
    q = [0.171, -0.912,  2.21, -2.866, -1.571,  0.169]
    q = [-0.16 , -0.832,  1.863, -0.751, -0.213, -1.857]
    q = np.deg2rad([118.35, -92.94, -94.20, 3.53, 57.27, 78.29]) #2.06563, -1.62208, -1.64403,  0.06159,  0.9995 ,  1.36647
    q = [1.96414, -1.9198 , -1.3648 ,  0.08443,  1.10098,  1.35949]
    q = np.deg2rad([90.0, 0.0, 45.0, 0.0, -90.0, 0.0])
    q = [1.89951, -1.89104, -1.31663,  0.00926,  1.16537,  1.35538]
    q = [1.90647, -1.02627, -1.30762, -0.46714,  1.31118,  0.00343] # door
    q = [1.93048, -2.13818, -1.21322,  0.36888,  1.05412,  1.09136]
    q = [-2.27058, -1.72553,  1.89436, -3.41743,  5.39808,  0.01899]
    q = [-2.70307, -1.83606,  1.46569,  0.34989,  2.01514, -0.04245]
    q = [-2.25055, -1.7502 ,  1.8647 , -3.25888,  5.39604,  0.01208]
    q = [-3.82148, -1.74987,  1.86476, -3.25883,  5.39585,  0.01206]

    arm.set_joint_positions(position=q, wait=wait, t=0.5)

def trajectory():
    traj = [
        [-2.25055, -1.7502 ,  1.8647 , -3.25888,  5.39604,  0.01208],
        [-3.82148, -1.74987,  1.86476, -3.25883,  5.39585,  0.01206],
        [-3.12333, -1.74988,  1.86476, -3.25883,  5.39585,  0.01205],
        [-1.55255, -1.74981,  1.86477, -3.25882,  5.39581,  0.01205],
        [-0.67992, -1.74976,  1.86478, -3.25882,  5.39578,  0.01207],
        [-2.36069, -1.54673,  1.30599, -4.53528,  4.66136,  0.74619],
        [-2.76354, -1.45668,  1.36001, -4.24608,  4.47966,  1.1765 ],
        [-1.38552, -1.16804,  0.83775, -4.41607,  5.04895, -0.2031 ]
    ]
    for t in traj:
        arm.set_joint_positions(position=t, wait=True, t=1.0)


def admittance_control(method="integration"):
    print("Admitance Control", method)
    # arm.set_wrench_offset(override=True)
    e = 0.0001
    K = np.ones(6) * 300.0
    M = np.ones(6) * 0.01
    # B = np.ones(6) * 10000
    B = e*2*np.sqrt(K*M)
    # K = 300.0 #3000
    # M = 1.
    dt = 0.002
    x0 = arm.end_effector()
    model = AdmittanceModel(M, K, B, dt, method=method)

    k = 1200
    position_kp = np.array([2.0e-3, 2.0e-3, 2.0e-3, 1.0e-2, 1.0e-2, 1.0e-2]) * k
    position_kd = np.array([2.0e-3, 2.0e-3, 2.0e-3, 1.0e-2, 1.0e-2, 1.0e-2]) * np.sqrt(k)
    model.p_controller = utils.PID(Kp=position_kp, Kd=position_kd)
    
    target = x0[:]
    target[1] -= 0.02
    target[2] += 0.02
    
    print("Impedance model", model)
    timeout = 10.
    initime = rospy.get_time()
    while not rospy.is_shutdown() \
                and (rospy.get_time() - initime) < timeout: 
        arm.set_impedance_control(target, model, timeout=0.05, max_force=500)
    
    print("ok?")
    go_to(True)

def rotation():
    go_to(True)
    target = [-0.09108, -0.51868,  0.47657, -0.49215, -0.48348,  0.51335, -0.5104]
    target_quaternion = Quaternion(np.roll(target[3:], 1))
    current_quaternion = arm.end_effector()[3:]

    from_quaternion = Quaternion(np.roll(current_quaternion, 1))

    rotate = Quaternion(axis=[1,0,0], degrees=00.0)
    to_quaternion = from_quaternion * rotate

    # rotate = Quaternion(axis=[0,1,0], degrees=-50.0)
    # to_quaternion *= rotate

    rotate = Quaternion(axis=[0,0,1], degrees=50.0)
    to_quaternion *= rotate

    old_q = from_quaternion
    for q in Quaternion.intermediates(q0=from_quaternion, q1=target_quaternion, n=10):
        w = transformations.angular_velocity_from_quaternions(old_q,q,1.0/10.0)
        old_q = q
        delta = np.concatenate((np.zeros(3),w.vector))
        
        to_pose = transformations.pose_euler_to_quaternion(arm.end_effector(), delta, dt=1.0/10.0)
        arm.set_target_pose_flex(to_pose, t=1.0/10.0)
        rospy.sleep(1.0/10.0)

    dist = Quaternion.distance(target_quaternion, Quaternion(np.roll(arm.end_effector()[3:], 1)))
    print(dist)

    # slerp = Quaternion.slerp(q0=from_quaternion,q1=target_quaternion, amount=0.5)

    # q1 = from_quaternion
    # q2 = slerp

    # w = transformations.angular_velocity_from_quaternions(q1,q2,1.0)

    # delta = np.concatenate((np.zeros(3),w.vector))

    # print("zeros", delta)

    # to_pose = transformations.pose_euler_to_quaternion(arm.end_effector(), delta, dt=1.0)

    # pose = np.concatenate((arm.end_effector()[:3], np.roll(slerp.elements, -1)))
    # arm.set_target_pose(to_pose, wait=True, t=1)

def rotation_pd():
    go_to(True)
    kp = [5e-1,5e-1,5e-1,0.5e+1,0.5e+1,0.5e+1]
    kd = [1e-2,1e-2,1e-2,0.1,0.1,0.1]
    pd = utils.PID(Kp=kp, Kd=kd)

    target = [-0.08337, -0.47366,  0.44836,  0.6636 ,  0.25114, -0.23011, 0.66604]
    target = [-0.09108, -0.51868,  0.47657, -0.49215, -0.48348,  0.51335, -0.5104]
    Qd = Quaternion(np.roll(target[3:], 1))

    Qc = Quaternion(np.roll(arm.end_effector()[3:],1))
    # rot_error = orientation_error(Qd, Qc)
    # print(rot_error, )

    timeout = 30.0
    dt = 0.002
    initime = rospy.get_time()
    print("initial diff",target-arm.end_effector())
    while not rospy.is_shutdown() \
                and (rospy.get_time() - initime) < timeout:
        Qc = Quaternion(np.roll(arm.end_effector()[3:],1))

        rot_error = orientation_error(Qc, Qd)
        trans_error = target[:3] - arm.end_effector()[:3]

        error = np.concatenate((trans_error, rot_error))
        print(np.round(error, 5))
        step = pd.update(error, dt=dt) # angular velocity

        pose = transformations.pose_from_angular_veloticy(arm.end_effector(), step, dt=dt)

        arm.set_target_pose_flex(pose, t=dt)
        rospy.sleep(dt)
    print("final pose",target-arm.end_effector())

def angular_velocity(Q,Q_dot):
    return jacobian(Q)*Q_dot.elements

def jacobian(Q):
    return 2 * E(Q).T
    
def E(Q):
    assert isinstance(Q, Quaternion)
    n = Q.scalar
    e = Q.vector
    E = np.concatenate(([np.array(-e).T], n*np.identity(3) - spalg.skew(e)), axis=0)
    return E

def orientation_error(Qd, Qc):
    # ne = Qc.scalar*Qd.scalar + np.dot(np.array(Qc.vector).T,Qd.vector)
    ee = Qc.scalar*np.array(Qd.vector) - Qd.scalar*np.array(Qc.vector) + np.dot(spalg.skew(Qc.vector),Qd.vector)
    return ee

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
    parser.add_argument('--relative', action='store_true', help='relative to end-effector')
    parser.add_argument('--rotation_pd', action='store_true', help='relative to end-effector')

    args = parser.parse_args()

    driver = ROBOT_GAZEBO

    global arm
    arm = CompliantController(ft_sensor=True, driver=driver, ee_transform=[-0.,   -0.,   0.05,  0,    0.,    0.,    1.  ])

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
    if args.rotation_pd:
        rotation_pd()
    trajectory()

    print("real time", round(timeit.default_timer() - real_start_time, 3))
    print("ros time", round(rospy.get_time() - ros_start_time, 3))


if __name__ == "__main__":
    main()
