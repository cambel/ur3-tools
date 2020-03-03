import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as tick

from plotter_utils import smooth, reformat_large_tick_values

def load_file(filename, remove_list):
    data = np.load(filename)
    for r in remove_list:
        data = np.delete(data, r, 0)
    return data

filename = "/media/Extra/research/real/peg/02/20200221T105331hybrid_position_force14.npy"
filename = "/media/Extra/research/real/wood_toy/02/20200221T094035hybrid_position_force14.npy"

hyb14 = load_file(filename, [0])
print(hyb14.shape)
filename = "/media/Extra/research/real/wood_toy/02/20200221T092258impedance13.npy"
imp13pd = load_file(filename, [0])

# filename = '20200220T203933hybrid_position_force9.npy'
# da = np.load(filename, allow_pickle=True)
# print(da)

weight = 0.9

ws = [0.1,0.1,0.1,1,1]
labels = ['distance', 'action', 'force', 'step', 'goal']
# goal = data[:,4]
# data = data[:,:4]
# data[:,0] += goal

# for i in range(hyb14.shape[1]-1):
#     plt.plot(x, hyb14[:,i]*ws[i], label=labels[i])

x = np.linspace(1, hyb14.shape[0], hyb14.shape[0])
plt.plot(x, smooth(hyb14[:,2],weight), label='SAC Hybrid [14]')

x = np.linspace(1, imp13pd.shape[0], imp13pd.shape[0])
plt.plot(x, smooth(imp13pd[:,2],weight), label='SAC Impedance [13pd]')

plt.ylabel('Force [N]')
plt.xlabel('Step')
plt.legend()
ax = plt.subplot(111)
ax.xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
plt.show()
