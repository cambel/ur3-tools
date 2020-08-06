import numpy as np
import os

def get_files(folder, keys):
    files = os.listdir(folder)
    res = []
    for k in keys:
        res.append([folder + fl for fl in files if k in fl])
    return res

sac_folder = '/media/cambel/Extra/research/IROS20_revised/SAC/no_penalization_both/post/'
sac_folder = '/media/cambel/Extra/research/IROS20_revised/Real/ring/good/post/'
sac_folder = '/media/cambel/Extra/research/IROS20_revised/Real/wrs/01/post/'
sac_folder = '/media/cambel/Extra/research/IROS20_revised/SAC/no_penalization2 ur3e/01/post/'
sac_folder = '/media/cambel/Extra/research/IROS20_revised/SAC/fuji-pc/50k v4 no step/post/'

file_list = os.listdir(sac_folder)
sac_keys = ['impedance13', 'hybrid14']
sac_keys = ['impedance8','impedance13n','impedance13pd','impedance18','hybrid9', 'hybrid14', 'hybrid19', 'hybrid24']

sac_files = get_files(sac_folder, sac_keys)

files = np.array(sac_files).flatten()
for f in files:
    data = np.load(f)
    print(f, np.sum(data[:,4]))
