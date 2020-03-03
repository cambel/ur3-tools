import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as tick

from plotter_utils import smooth, reformat_large_tick_values


# data_sac = np.load('/home/cambel/dev/sac18.npy')
# data_gps = np.load('/media/Extra/research/real/wood_toy/gps/gpsimpedance18.npy',allow_pickle=True)

# print(data_sac[1].shape)
# print(data_gps[1].shape)

ax = plt.subplot(111)
box = ax.get_position()

# ax.plot(data_sac[0], data_sac[1], '-', label='SAC Impedance [18]')
# ax.plot(data_gps[0], data_gps[1], '-', label='GPS Impedance [18]')

# plt.yticks(np.arange(-1000, -100, 100.0))

# plt.xlabel("Steps", size='x-large')
# plt.ylabel("Cumulative reward", size='x-large')
# plt.legend()
# ax.xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))

def norm(dist):
    return (0.5 * (dist ** 2) * 0.01 +
                np.log(0.00001 + (dist ** 2)) * 0.1)

x = np.linspace(-50,50.,1000)
y = norm(x)
plt.plot(x,y)

plt.show()


