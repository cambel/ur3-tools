from plotter_utils import smooth, reformat_large_tick_values, csv_to_list, npy_to_list, extrapolate
import numpy as np
import os
import matplotlib.cm as cm
import matplotlib.ticker as tick
import matplotlib.pyplot as plt
import csv
import matplotlib
matplotlib.use('TkAgg')
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=np.inf)


colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def get_files(folder, keys):
    files = os.listdir(folder)
    res = []
    for k in keys:
        res.append([folder + '/' + fl + '/detailed_log.npy' for fl in files if k in fl and os.path.isdir(folder + '/' + fl)])
    return res


hyb_folder = '/media/cambel/Extra/research/IROS20_revised/SAC/no_penalization2'
hyb_folder = '/media/cambel/Extra/research/IROS20_revised/SAC/no_penalization2 ur3e'
hyb_folder = '/media/cambel/Extra/research/IROS20_revised/SAC/no_penalization_both'
hyb_folder = '/media/cambel/Extra/research/IROS20_revised/SAC/50k v3 alien/results'
hyb_folder = '/media/cambel/Extra/research/IROS20_revised/SAC/50k v3/results'
hyb_folder = '/media/cambel/Extra/research/IROS20_revised/SAC/50k v3 both/'
file_list = os.listdir(hyb_folder)
hyb_keys = ['hybrid_iros20_9', 'hybrid_iros20_14', 'hybrid_iros20_19', 'hybrid_iros20_24']
sac_hyb_files = get_files(hyb_folder, hyb_keys)

sac_folder = '/media/cambel/Extra/research/IROS20_revised/SAC/no_penalization2'
sac_folder = '/media/cambel/Extra/research/IROS20_revised/SAC/no_penalization2 ur3e'
sac_folder = '/media/cambel/Extra/research/IROS20_revised/SAC/no_penalization_both'
sac_folder = '/media/cambel/Extra/research/IROS20_revised/SAC/50k v3/results'
sac_folder = '/media/cambel/Extra/research/IROS20_revised/SAC/50k v3 alien/results'
sac_folder = '/media/cambel/Extra/research/IROS20_revised/SAC/50k v3 both/'
file_list = os.listdir(sac_folder)
sac_keys = ['SAC_imp-iros20_8', 'SAC_imp-iros20_13n', 'SAC_imp-iros20_13pd', 'SAC_imp-iros20_18']
sac_imp_files = get_files(sac_folder, sac_keys)

ring_folder = '/media/cambel/Extra/research/IROS20_revised/Real/ring/good'
file_list = os.listdir(ring_folder)
ring_keys = ['SAC_ring_imp13pd', 'SAC_ring_hyb14']
sac_ring_files = get_files(ring_folder, ring_keys)
# print(sac_files)

wrs_folder = '/media/cambel/Extra/research/IROS20_revised/Real/wrs/01'
file_list = os.listdir(wrs_folder)
wrs_keys = ['SAC_wrs_imp13pd', 'SAC_wrs_hyb14']
sac_wrs_files = get_files(wrs_folder, wrs_keys)

mode = 0

def pad(arr, length):
    new_arr = []
    for a in arr:
        res = np.ones(length)*a[-1]
        res[:len(a)] = a
        new_arr.append(res)
    return new_arr

def cumulative_value(data):
    for i in range(len(data)):
        if i > 0:
            data[i] = data[i-1] + data[i]
    return data

def prepare_data(files, weight, points=12000):
    # mode 0: reward x episode
    # mode 1: steps x episode
    # mode 2: reward x step
    # mode 3: cumulative reward x episode
    # mode 4: cumulative reward x step
    d = []
    max_len = 0
    for f in files:
        data = np.load(f, allow_pickle=True)
        tmp = None
        if mode >= 3:
            index = 3
            tmp = cumulative_value(np.copy(data[:, index]))
        else:
            index = 3 if mode == 0 or mode == 2 else 2
            tmp = smooth(data[:, index], weight)
        if mode == 2 or mode == 4:
            _, tmp = extrapolate(data[:,1], tmp, points)
        print(data[:,0].shape)
        d.append(tmp)
        max_len = len(tmp) if len(tmp) > max_len else max_len
    x = np.linspace(0, max_len, max_len)
    d = pad(d, max_len)
    print(np.array(d).shape)
    y = np.average(d, axis=0)
    y_std = np.std(d, axis=0)
    return x, y, y_std


def simple_plot(ax, files, weight, alpha, color, label, line_style='-', linewidth=1.0):
    x, y, y_std = prepare_data(files, weight, points=20000)
    # np.savetxt(label+'.csv', np.array([x,y,y_std]).T, delimiter=',', fmt='%f')
    c = color
    ax.plot(x, y, line_style, color=c, label=label, linewidth=linewidth)
    ax.fill_between(x, y-y_std, y+y_std, antialiased=True, linewidth=0.5, alpha=alpha, edgecolor=c, facecolor=c)
    if mode >= 3:
         plt.yscale("log")
    if mode == 0 or mode == 1 or mode == 3:
        ax.axvline(x=len(x),c=color)


def plot_ft():
    weight = 0.8
    alpha = 0.1

    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0+0.0, box.y0 + box.height * 0.005, box.width * 1.1, box.height * 1.13])  # label in

    # Ring
    # simple_plot(ax, sac_ring_files[0], weight, alpha, colors[0], 'SAC I-13pd', '-', linewidth=1)
    # simple_plot(ax, sac_ring_files[1], weight, alpha, colors[1], 'SAC H-14', '-', linewidth=1)

    # WRS
    simple_plot(ax, sac_wrs_files[0], weight, alpha, colors[0], 'SAC I-13pd', '-', linewidth=1)
    simple_plot(ax, sac_wrs_files[1], weight, alpha, colors[1], 'SAC H-14', '-', linewidth=1)


    # ### SAC ###
    # # Hybrid #
    # simple_plot(ax, sac_hyb_files[0], weight, alpha, colors[0], 'SAC H-9', '-' , linewidth=1)
    # simple_plot(ax, sac_hyb_files[1], weight, alpha, colors[1], 'SAC H-14', '-', linewidth=1)
    # simple_plot(ax, sac_hyb_files[2], weight, alpha, colors[2], 'SAC H-19', '-', linewidth=1)
    # simple_plot(ax, sac_hyb_files[3], weight, alpha, colors[3], 'SAC H-24', '-', linewidth=1)

    # # Impedance #
    # simple_plot(ax, sac_imp_files[0], weight, alpha, colors[0], 'SAC I-8', '-', linewidth=1)
    # simple_plot(ax, sac_imp_files[1], weight, alpha, colors[1], 'SAC I-13', '-', linewidth=1)
    # simple_plot(ax, sac_imp_files[2], weight, alpha, colors[2], 'SAC I-13pd', '-', linewidth=1)
    # simple_plot(ax, sac_imp_files[3], weight, alpha, colors[3], 'SAC I-18', '-', linewidth=1)

    ax.legend(loc='upper left', ncol=2, prop={'size': 10})
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13),
    #       fancybox=True, shadow=True, ncol=2)
    # if mode == 0 or mode == 1 or mode == 3:
    #     plt.xlim((0,6000))
    # plt.ylim([-20, 220])
    # plt.yticks(np.arange(0, 400, 100.0), fontsize=10)
    plt.xticks(fontsize=12)
    plt.grid(linestyle='--', linewidth=0.5)
    if mode == 2 or mode == 4:
        plt.xlabel("Steps", size='xx-large')
    else:
        plt.xlabel("Episode", size='xx-large')
    plt.ylabel("Reward", size='xx-large')
    # plt.ylabel("Distance (m)", size='x-large')
    # ax.set_rasterized(True)
    ax.xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
    plt.show()


plot_ft()
