import numpy as np
import matplotlib.pyplot as plt

def plot_file(filename, hz):
    data = np.load(filename)
    error = data[0]
    update = data[1]
    print(error.shape)
    x = np.linspace(0, len(error), len(error))
    print(x.shape)
    plt.plot(x, error[:,0,1], label='Error pos'+str(hz)+'hz')
    # plt.plot(x, error[:,1,1], label='Error force'+str(hz)+'hz')
    plt.plot(x, update[:,0,1], label='Update'+str(hz)+'hz')


# filename = "/media/cambel/Extra/research/MDPI/reality_gap/sim_80hz.npy"
# plot_file(filename, 80)
# filename = "/media/cambel/Extra/research/MDPI/reality_gap/sim_100hz.npy"
# plot_file(filename, 100)
# filename = "/media/cambel/Extra/research/MDPI/reality_gap/sim_200hz.npy"
# plot_file(filename, 200)
# filename = "/media/cambel/Extra/research/MDPI/reality_gap/sim_400hz.npy"
# plot_file(filename, 400)
# filename = '/media/cambel/Extra/research/MDPI/reality_gap/sim_500hz.npy'
# plot_file(filename, 500)

# filename = '/media/cambel/Extra/research/MDPI/reality_gap/real_80hz.npy'
# plot_file(filename, 80)
# filename = '/media/cambel/Extra/research/MDPI/reality_gap/real_100hz.npy'
# plot_file(filename, 100)
# filename = '/media/cambel/Extra/research/MDPI/reality_gap/real_200hz.npy'
# plot_file(filename, 200)
# filename = '/media/cambel/Extra/research/MDPI/reality_gap/real_400hz.npy'
# plot_file(filename, 400)
# filename = '/media/cambel/Extra/research/MDPI/reality_gap/real_500hz.npy'
# plot_file(filename, 500)
# filename = '/media/cambel/Extra/research/MDPI/reality_gap/real_500hz_x6.npy'
filename = '/media/cambel/Extra/research/MDPI/reality_gap/v2/real_500hz_pos_x6.npy'
filename = '/media/cambel/Extra/research/MDPI/reality_gap/v2/real_500hz.npy'
plot_file(filename, 500)

plt.legend(loc='best')
plt.show()