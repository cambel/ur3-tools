import numpy as np

def extract_obs(obs):
    # 6 + 6 + 24 + 6*n + 1
    speed = obs[0][6:9].astype(np.float) # Distance [0:6]
    return speed


def process_data(episode_data):
    rdist = []
    for ep in episode_data:
        dist = extract_obs(ep)
        rdist.append(dist)
    return np.array(rdist)

def process(filename, index=-1):
    data = np.load(filename, allow_pickle=True)

    dist = process_data(np.array(data[index]))
    x = np.average(np.abs(dist[:,0]))
    y = np.average(np.abs(dist[:,1]))
    z = np.average(np.abs(dist[:,2]))
    print('x', x)
    print('y', y)
    print('z', z)
    return x,y,z

filename = '/home/cambel/dev/results/350k_SAC_randgoal_p24/state_20200727T113458.npy'
rx,ry,rz = process(filename, 2)

filename = '/home/cambel/dev/learned/simtest_state_20200727T101705.npy'
sx,sy,sz = process(filename, 0)

print("ratio", sx/rx,sy/ry,sz/rz)
    