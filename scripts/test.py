import numpy as np
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=np.inf)
cost_l1 = 1.0
cost_l2 = 10.0
cost_alpha = 0.00001

# def l1l2(dist, weights=[0.35, 0.35, 0.35, 0.1, 0.1, 0.1]):
def l1l2(dist, weights=[1.0, 1.0, 1.0, 0.1, 0.1, 0.1]):
    l1 = cost_l1 * np.array(weights)
    l2 = cost_l2 * np.array(weights)
    dist = dist
    norm = (0.5 * (dist ** 2) * l2 +
            np.log(cost_alpha + (dist ** 2)) * l1)
    return norm.sum()

print( l1l2(np.zeros(6)))
print( l1l2(np.ones(6)))

# a = '/home/cambel/dev/ft_data.npy'
# # b = '/media/cambel/Extra/research/IROS20_revised/SAC/fuji-pc/no_penalization2/03-no-step/state_20200511T085019impedance13-2.npy'
# # c = '/media/cambel/Extra/research/IROS20_revised/SAC/fuji-pc/no_penalization2/03-no-step/state_20200511T085019impedance13-3.npy'
# an = np.load(a,allow_pickle=True)
# # for d in an:

# # cn = np.load(c,allow_pickle=True)
# ft = np.array([a for a in an[:,0]])
# print(an.shape)
# print(ft.shape)
# print(ft[0])
# print(np.array(an[0]).shape)
# print(np.array(an[0])[0])
# print(np.array(an[3])[0])
# print(np.array(an[0]).shape)
# print(np.array(an[0])[0])
# c = np.delete(an, 0,axis=0)
# print(c.shape)

# bn = np.load(b, allow_pickle=True)
# cn = np.concatenate((an, bn))
# np.save(c, cn)
# np.save(a, c)

# from pyquaternion import Quaternion
# from ur_control.transformations import integrateUnitQuaternionDMM, integrateUnitQuaternionEuler

# q0 = Quaternion([.5,.5,.5,.5])
# q0.normalised
# print("q0", q0)
# w = np.zeros(3)
# w[0] = 1
# dt = 0.05

# print(integrateUnitQuaternionDMM(q0,w,dt))
# print(integrateUnitQuaternionEuler(q0,w,dt))

# from ur_control import transformations
# true_target = [-0.00312934, -0.37939538,  0.4540351 , -0.00235073, -0.6813838 ,  0.73190453, -0.00513233]
# error = np.random.normal(scale=[0.005,0.005,0.005,0.005,0.005,0.005], size=6)
# new_target = transformations.pose_euler_to_quaternion(true_target, error)

# print(new_target)


## remove outliers
# filename = '/media/cambel/Extra/research/MDPI/simulation/domain-rand/individual/p_24/20200716T105106.195386_SAC_randerror_p24/detailed_log.npy'
# a = np.load(filename, allow_pickle=True)
# b = a[1:]
# print(a.shape, b.shape)
# np.save(filename,b)