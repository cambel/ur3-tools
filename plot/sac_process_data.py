import numpy as np
from simulate_cost import distance_force_action_step_goal
import os

def get_files(folder, keys):
    files = os.listdir(folder)
    res = []
    for k in keys:
        res.append([folder + '/' + fl + '/detailed_log.npy' for fl in files if k in fl and os.path.isdir(folder + '/' + fl)])
    return res

def process_files(files):
    for filename in files:
        tmp = filename.replace('.npy', '')
        name = tmp.split('/')[-1]
        folder = filename.replace(name+'.npy', '')
        save_to = folder + 'post-' + name
        data = data = np.load(filename, allow_pickle=True)
        process_data(data, save_to)

def process_data(data, save_to):
    res = []
    for i in range(len(data)):
        reward, collision, n_steps = compute_cost(np.array(data[i]))
        # episode #, # step of episode, # steps session, reward, collision
        res.append([(i+1), n_steps, n_steps*(i+1), reward, collision])
    np.save(save_to, np.array(res))

def compute_cost(episode_data):
    max_distance = episode_data[0][0][:6].astype(np.float)
    reward = 0
    collision = False
    for ep in episode_data:
        dist, force, actions = extract_obs(ep)
        r, collision = distance_force_action_step_goal(dist, force, actions, max_distance)
        reward += r
    return reward, collision, episode_data.shape[0]

def extract_obs(obs):
    dist    = obs[0][:6].astype(np.float)
    force   = obs[0][12:18].astype(np.float)
    actions = obs[0][18:-1].astype(np.float)
    return dist, force, actions


sac_folder = '/media/cambel/Extra/research/IROS20_revised/SAC/50k v3 both/'
file_list = os.listdir(sac_folder)
sac_keys = ['hybrid_iros20_9', 'hybrid_iros20_14', 'hybrid_iros20_19', 'hybrid_iros20_24']
sac_files = get_files(sac_folder, sac_keys)

process_files(sac_files)