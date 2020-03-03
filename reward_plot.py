import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as tick

from plotter_utils import smooth, reformat_large_tick_values


data = np.load('/media/Extra/research/real/wood_toy/sac/impedance/01/20200213T123104impedance18.npy')

print(data.shape)
weight = 0.5

ax = plt.subplot(111)
box = ax.get_position()
# ax.set_position([box.x0+0.02, box.y0 + box.height * 0.3, box.width, box.height * 0.8])

x = np.linspace(0, data.shape[0]*200, data.shape[0])

colors = ['green', 'grey', 'blue','brown', 'purple']
labels = ['Distance', 'Action', 'Force', 'Step', 'Goal']

# for i in range(4):
#     y = smooth(-1*data[:,i], weight)
#     ax.plot(x, y, '-', label=labels[i])

y = smooth(np.sum(data,axis=1), weight)
# np.save('/home/cambel/dev/sac18.npy',[x,y])
ax.plot(x, y, '-', label='SAC Impedance [18]')

plt.yticks(np.arange(-1000, -100, 100.0))

plt.xlabel("Steps", size='x-large')
plt.ylabel("Cumulative reward", size='x-large')
plt.legend()
# plt.legend(title='Cost')
ax.xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
plt.show()