import numpy as np
from simulate_cost import distance_force_action_step_goal
import os

def get_files(folder, keys):
    files = os.listdir(folder)
    res = []
    for k in keys:
        res.append([folder + fl for fl in files if k in fl])
    return res

def process_files(files):
    files = np.array(files).flatten()
    for filename in files:
        print(filename)
        tmp = filename.replace('.npy', '')
        name = tmp.split('/')[-1]
        folder = filename.replace(name+'.npy', '')
        if not os.path.exists(folder + 'post'):
            os.makedirs(folder + 'post')
        save_to = folder + 'post/' + name
        data = data = np.load(filename, allow_pickle=True)
        process_data(data, save_to)

def process_data(data, save_to):
    res = []
    for i in range(len(data)):
        reward, collision, n_steps = compute_cost(np.array(data[i]))
        # episode #, # step of episode, # steps session, reward, collision
        res.append([(i+1), n_steps, n_steps*(i+1), reward, collision])
        if n_steps*(i+1) > 50000:
            break
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


sac_folder = '/media/cambel/Extra/research/IROS20_revised/SAC/no_penalization_both/'
sac_folder = '/media/cambel/Extra/research/IROS20_revised/SAC/50k v3/'
sac_folder = '/media/cambel/Extra/research/IROS20_revised/Real/ring/good/'
sac_folder = '/media/cambel/Extra/research/IROS20_revised/Real/wrs/01/'
sac_folder = '/media/cambel/Extra/research/IROS20_revised/SAC/fuji-pc/50k v4 no step/'
file_list = os.listdir(sac_folder)
sac_keys = ['impedance13', 'hybrid14']
sac_keys = ['impedance8','impedance13n','impedance13pd','impedance18','hybrid9', 'hybrid14', 'hybrid19', 'hybrid24']
sac_files = get_files(sac_folder, sac_keys)

process_files(sac_files)