import numpy as np


def pre_actions_24_to_25():
    pass


def pre_actions_14_to_25(actions):
    assert len(actions) == 14, len(actions)
    x = actions[:6]
    pds = actions[6:8]
    pds = np.repeat(pds, 6)
    alpha = actions[8:]
    return np.concatenate([x, pds, alpha, [0]])


filename = '/home/cambel/dev/pretrain/randgoal_p14.npy'
filename = '/home/cambel/dev/pretrain/randerror_p14.npy'
savefile = '/home/cambel/dev/pretrain/processed_randerror_p14_3.npy'

data = np.load(filename, allow_pickle=True)
print(data.shape)
savedata = []

actions_dim = 25

# os.finish()

for d in data[:1000]:
    episode = np.array(d)
    for i in range(len(episode)-1):
        step = episode[i].flatten()[:-1]
        ft = step[-6:]
        ft = np.tile(ft, 12)
        act = pre_actions_14_to_25(step[12:-6])
        y = episode[i+1][0][12:step.shape[0]-6]
        y = pre_actions_14_to_25(y)
        obs = np.concatenate([step[:12].ravel(), act.ravel(), ft.ravel()])
        savedata.append([obs, y])

print(np.array(savedata).shape)
np.save(savefile, savedata, allow_pickle=True)
