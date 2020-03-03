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
        dist = np.linalg.norm(pose[i], axis=-1)
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
            reward += 300.0
            goal[i] = 300.0
            total[i] = reward
            break
        total[i] = reward
    
    return cdist, caction, cforce, cstep, total

data_logger = DataLogger()

weight = 0.9

def load_data(folder, ftype):
    cost = []
    f = []
    x = []
    for i in range(15):
        filename = folder + ftype + ('%02d.pkl' % i)
        # filename = folder + ('traj_sample_itr_%02d.pkl' % i)
        traj = data_logger.unpickle(filename)
        for t in traj:
            for s in t.get_samples():
                ee_pose = s.get(3)
                x.append(ee_pose)
                force = s.get(22)
                f.append(force)
                action = s.get_U()
                cost.append(list(reward(force, ee_pose, action)))

    print(np.array(cost).shape)
    cost = np.sum(cost, axis=2)
    print(np.array(cost).shape)
    return cost, f, x, action

# cost, f, x = load_data('/root/dev/abc/wood18/', 'pol_sample_itr_')
cost, f, x, pact = load_data('/root/dev/abc/wood24/', 'traj_sample_itr_')
pcost, pf, px, pact = load_data('/root/dev/abc/sim18/', 'pol_sample_itr_')
# cost, f, x, act = load_data('/root/dev/abc/peg18/', 'pol_sample_itr_')
# cost, f, x, act = load_data('/root/dev/abc/peg24/', 'pol_sample_itr_')


ax = plt.subplot(111)
l = ['Distance','Action','Force','Step','total',]
x_axis = np.linspace(0, cost.shape[0]*200, cost.shape[0]) 

# for i in range(cost.shape[1]-1):
#     y_axis = smooth(cost[:,i], weight)
#     plt.plot(x_axis, y_axis, label=l[i])

# y = np.array(smooth(cost[:,4], weight))
# np.save('/root/dev/gps18.npy',[x_axis,y],allow_pickle=True)
# plt.plot(x_axis, smooth(cost[:,4], weight), label="GPS Impedance [18]")

print(np.array(f).shape)
x = np.linspace(1,200, 200)

# for i in range(len(pf)):
# # for i in range(5):
#     y = pf[i][:,1]
#     plt.plot(x, y, label="policy %s" % i , linewidth=1)

y = np.average(f[:3], axis=0)[:,1]
plt.plot(x, y, color='C1', label="Initial policy", ls='--',linewidth=1)

y = np.average(f[-3:], axis=0)[:,1]
plt.plot(x, y, color='C2', label="Final policy", linewidth=1)

# abc = np.sum(pact, axis=1)

# plt.plot(x, abc, color='C3', label="action", linewidth=1)

# plt.yticks(np.arange(0, 30, 5.0))
# plt.yticks(np.arange(0, 700, 100.0))
# plt.xticks(np.arange(0, 10000, 1000.0))
plt.xlabel("Steps", size='x-large')
# plt.ylabel("Cumulative reward", size='x-large')
plt.ylabel("Force [y]", size='x-large')
# plt.legend(title='Cost')
plt.grid(linestyle='--', linewidth=0.5)
plt.legend()
ax.xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
plt.show()
