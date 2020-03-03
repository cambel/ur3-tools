from gps.sample.sample_list import SampleList
from gps.sample.sample import Sample
from gps.utility.data_logger import DataLogger
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as tick

from plotter_utils import smooth, reformat_large_tick_values

def force_reward(force):
    net_force = np.sum(np.abs(force))
    return net_force if net_force <= 50.0 else 50.0*1.5

def l1l2(dist):
    cost_l2 = 0.01
    cost_l1 = 1.0
    cost_alpha = 0.0001
    return (0.5 * (dist ** 2) * cost_l2 +
            np.log(cost_alpha + (dist ** 2)) * cost_l1)

def reward(force, pose, action):
    total = np.zeros(force.shape[0])
    cforce = np.zeros(force.shape[0])
    cdist = np.zeros(force.shape[0])
    cstep = np.zeros(force.shape[0])
    caction = np.zeros(force.shape[0])
    goal = np.zeros(force.shape[0])

    dist_w = 0.1
    force_w = 0.2
    action_w = 0.1

    for i in range(force.shape[0]):
        dist = np.linalg.norm(ee_pose[i], axis=-1)
        dist_cost = l1l2(dist) * dist_w
        cdist[i] = dist_cost
        ft_cost = force_reward(force[i]) * force_w
        cforce[i] = ft_cost
        reward = -1*(ft_cost + dist_cost)
        reward -= 2.0  # penalty per step
        cstep[i] = 2.0
        caction[i] = np.sum(0.5*action[i]**2 * action_w)
        reward -= caction[i]
        if dist < 3.0:
            reward += 100.
            goal[i] = 100.
            total[i] = reward
            break
        total[i] = reward
    
    return cdist-goal, caction, cforce, cstep#, total#, goal

data_logger = DataLogger()

x = []
f = []
cost = []
weight = 0.75

for i in range(15):
    # filename = '/root/dev/abc/wood/traj_sample_itr_%02d.pkl' % i
    filename = '/root/dev/abc/wood18/traj_sample_itr_%02d.pkl' % i
    # print(filename)
    traj = data_logger.unpickle(filename)
    for t in traj:
        for s in t.get_samples():
            ee_pose = s.get_X()[:,:6]
            x.append(ee_pose)
            force = s.get_X()[:,12:18]
            f.append(force)
            action = s.get_U()
            cost.append(list(reward(force, ee_pose, action)))

print(np.array(cost).shape)
cost = np.sum(cost, axis=2)
print(np.array(cost).shape)

ax = plt.subplot(111)

l = ['Distance','Action','Force','Step','total',]
x_axis = np.linspace(0, cost.shape[0]*200, cost.shape[0]) 

# for i in range(cost.shape[1]-1):
#     y_axis = smooth(cost[:,i], weight)
#     plt.plot(x_axis, y_axis, label=l[i])

plt.stackplot(x_axis, cost.T, labels=l)

# y = np.array(smooth(cost[:,4], weight))
# np.save('/root/dev/gps18.npy',[x_axis,y],allow_pickle=True)
# plt.plot(x_axis, smooth(cost[:,4], weight), label="GPS Impedance [18]")

# plt.yticks(np.arange(-1000, -500, 100.0))
# plt.yticks(np.arange(-1000, -100, 100.0))
# plt.xticks(np.arange(0, 10000, 1000.0))
plt.xlabel("Steps", size='x-large')
plt.ylabel("Cumulative reward", size='x-large')
# plt.legend(title='Cost')
plt.legend()
ax.xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
plt.show()




# force_x_episode = np.apply_along_axis(force_reward, axis=2, arr=f)
# print(force_x_episode.shape)

# print("x",np.array(x).shape)
# pose_x_episode = np.mean(np.apply_along_axis(l1l2, axis=2, arr=x), axis=2)
# print(pose_x_episode.shape)

# # x_axis = np.linspace(1, 200, 200)
# # cnt = 1
# # for e in force_x_episode:
# #     plt.plot(x_axis, e, label=cnt)
# #     cnt+=1

# mean_force = np.mean(force_x_episode, axis=1)
# mean_force = np.delete(mean_force, 2)

# x_axis = np.arange(1, 30, 1)
# plt.plot(x_axis, mean_force, label='impedance force')


# x_axis = np.arange(1, 31, 1)
# mean_dist = np.mean(np.abs(pose_x_episode), axis=1)
# plt.plot(x_axis, mean_dist, label='impedance force')