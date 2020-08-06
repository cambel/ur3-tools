# Draw a conical helix
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from pyquaternion import Quaternion
cm = plt.get_cmap("CMRmap")

def diff_steps(a,b,steps):
    return np.linspace(a-b, 0, steps)

def concat_vec(x,y,z,steps):
    x = x.reshape(-1, steps)
    y = y.reshape(-1, steps)
    z = z.reshape(-1, steps)
    return np.concatenate((x, y, z), axis=0).T

def linear_trajectory(p1, p2, steps):
    x = np.array(p2) + (np.array(p2)-p1)
    x = diff_steps(p1[0], p2[0], steps)
    y = diff_steps(p1[1], p2[1], steps)
    z = diff_steps(p1[2], p2[2], steps)
    return concat_vec(x, y, z, steps)

def circle(radius, theta_offset, revolutions, steps):
    theta = np.linspace(0, 2*np.pi*revolutions, steps) + theta_offset
    x = radius*np.cos(theta)
    y = radius*np.sin(theta)
    return x, y

def circunference(p1, p2, steps):
    euclidean_dist = np.linalg.norm(np.array(p2[:2])-np.array(p1[:2]))
    x,y = circle(euclidean_dist, 0.0, 2.0, steps)
    x += p2[0]
    y += p2[1]
    z = np.zeros(steps)+p1[2]
    return concat_vec(x, y, z, steps)


def get_conical_helix_trajectory(p1, p2, steps):
    """ Compute Cartesian conical helix between 2 points"""
    euclidean_dist = np.linalg.norm(np.array(p2[:2])-np.array(p1[:2]))
    radius = np.linspace(euclidean_dist,0,steps)
    theta_offset = np.arctan2((p1[1]- p2[1]),(p1[0]-p2[0]))

    revolutions = 5.0
    x,y = circle(radius, theta_offset, revolutions, steps)
    x += p2[0]
    y += p2[1]
    z = np.linspace(p1[2], p2[2], steps)
    return concat_vec(x, y, z, steps)

def plot(ax, vec, marker='o', label=None):
    ax.plot([vec[0]], [vec[1]], [vec[2]], marker=marker, label=label)


target = [-0.00319718, -0.37576394,  0.45137436, -0.4986701138808597, 0.49256585342539383, 0.5200487598517108, -0.4881150324852347]
target = [-0.00319718, -0.37576394,  0.45137436, -0.0021967960033876664, -0.6684438218043409, 0.7437414872148852, -0.0051605594964759876]
target = [-0.01455178, -0.37641212,  0.44570216, -0.00443673,  0.67128779, -0.73901023, -0.05671771] 

init = [-0.02702883, -0.35275209,  0.46842613,  -0.0021968 , -0.66844382,  0.74374149, -0.00516056] 
init = [0.01249222, -0.35495369,  0.40881974, -0.00129695, -0.69332819,  0.71635534, -0.07829032]                                                                                                                        
steps = 300

init_pose = init[:3]
final_pose = target[:3]

target_q = Quaternion(np.roll(target[3:], 1))

p2 = np.zeros(3)
p1 = target_q.rotate(np.array(init_pose) - final_pose)

traj = get_conical_helix_trajectory(p1, p2, steps)
traj2 = get_conical_helix_trajectory(init_pose, final_pose, steps)

traj = np.apply_along_axis(target_q.rotate,1,traj)
traj = traj + final_pose

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
col = np.arange(steps)
# ax.scatter(traj2[:,0], traj2[:,1], traj2[:,2], s=15,c=col, marker='.')
ax.scatter(traj[:,0], traj[:,1], traj[:,2], s=15,c=col, marker='.')
# plot(ax, p1, label='p1')
# plot(ax, p2, label='p2')
plot(ax, init_pose, label='initial')
plot(ax, final_pose, label='final')

# make the panes transparent
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ax.set_xticks([-0.0032, -0.025])
# ax.set_yticks([-0.35, -0.38])
# ax.set_zticks([0.45, 0.46, 0.47])
# make the grid lines transparent
# ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
# ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
# ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.legend()
plt.show()