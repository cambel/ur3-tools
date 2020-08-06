import csv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

colors = iter(['#550000', '#D46A6A', '#004400', '#55AA55', '#061539', '#4F628E'])

amd_a = "/root/dev/tools/discretization.npy"
amd_b = "/root/dev/tools/integration.npy"
amd_i = "/root/dev/tools/traditional.npy"

def plot_ft():
    adm_a = np.load(amd_a, allow_pickle=True)
    adm_b = np.load(amd_b, allow_pickle=True)
    adm_i = np.load(amd_i, allow_pickle=True)
    
    print("a", adm_a.shape, "b", adm_b.shape, "c", adm_i.shape)

    t = 10 #sec

    xa = np.linspace(0, t, num=adm_a.shape[0])
    xb = np.linspace(0, t, num=adm_b.shape[0])
    xi = np.linspace(0, t, num=adm_i.shape[0])
    ax = plt.subplot(111)

    v = 1
    ax.plot(xa, adm_a[:,v], '--', color='Blue', label='admittance discrete')
    ax.plot(xb, adm_b[:,v], 'k', color='Orange', label='admittance integration')
    ax.plot(xi, adm_i[:,v], '--', color='Green', label='impedance')
    ax.legend()
    plt.xlim((0,t))
    # plt.yticks(np.arange(-5, 50, 5.0))
    plt.xticks(np.arange(0, t, 1.0))
    plt.grid(linestyle='--', linewidth=0.5)
    plt.xlabel("Time (s)", size='x-large')
    plt.ylabel("Force (N)", size='x-large')
    # plt.ylabel("Distance (m)", size='x-large')

    plt.show()

plot_ft()