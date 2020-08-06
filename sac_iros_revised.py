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



ring_folder = '/media/cambel/Extra/research/IROS20_revised/Real/ring/good'
file_list = os.listdir(ring_folder)
ring_keys = ['SAC_ring_imp13pd', 'SAC_ring_hyb14']
sac_ring_files = get_files(ring_folder, ring_keys)
# print(sac_files)

wrs_folder = '/media/cambel/Extra/research/IROS20_revised/Real/wrs/01'
file_list = os.listdir(wrs_folder)
wrs_keys = ['SAC_wrs_imp13pd', 'SAC_wrs_hyb14']
sac_wrs_files = get_files(wrs_folder, wrs_keys)

mode = 2
ctrl = 0
savename =""
sac_folder ="/media/cambel/Extra/research/IROS20_revised/SAC/both/no_penalization"
if ctrl == 0:
    savename = "sim-hyb-penalization-v2.png" 
    sac_folder = '/media/cambel/Extra/research/IROS20_revised/SAC/both/penalization'
elif ctrl == 1:
    savename = "sim-imp-penalization-v3.png" 
    sac_folder = '/media/cambel/Extra/research/IROS20_revised/SAC/both/penalization'
elif ctrl == 2:
    savename = "sim-hyb-no_penalization-v2.png" 
    sac_folder = '/media/cambel/Extra/research/IROS20_revised/SAC/fuji-pc/no_penalization/all/'
elif ctrl == 3:
    savename = "sim-imp-no_penalization-v3.png" 
    sac_folder = '/media/cambel/Extra/research/IROS20_revised/SAC/fuji-pc/no_penalization/all/'
elif ctrl == 4:
    savename = "ring-real-expv2.png" 
elif ctrl == 5:
    savename = "wrs-real-expv2.png" 


file_list = os.listdir(sac_folder)
sac_keys = ['SAC_imp-iros20_8', 'SAC_imp-iros20_13n', 'SAC_imp-iros20_13pd', 'SAC_imp-iros20_18','hybrid_iros20_9', 'hybrid_iros20_14', 'hybrid_iros20_19', 'hybrid_iros20_24']
sac_files = get_files(sac_folder, sac_keys)

def pad(arr, length):
    new_arr = []
    for a in arr:
        if len(a) < length:
            res = np.ones(length)*a[-1]
            res[:len(a)] = a
            new_arr.append(res)
        else:
            new_arr.append(a[:length])
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
    avg_len = 0
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
        avg_len += len(tmp)
    avg_len = int(avg_len / len(files))
    x = np.linspace(0, max_len, max_len)
    d = pad(d, max_len)
    print(np.array(d).shape)
    y = np.average(d, axis=0)
    y_std = np.std(d, axis=0)/1.5
    return x, y, y_std


def simple_plot(ax, files, weight, alpha, color, label, line_style='-', linewidth=1.0):
    points = 50000 if ctrl < 4 else 20000
    x, y, y_std = prepare_data(files, weight, points=points)
    # np.savetxt(label+'.csv', np.array([x,y,y_std]).T, delimiter=',', fmt='%f')
    c = color
    ax.plot(x, y, line_style, color=c, label=label, linewidth=linewidth)
    ax.fill_between(x, y-y_std, y+y_std, antialiased=True, linewidth=0.5, alpha=alpha, edgecolor=c, facecolor=c)
    if mode >= 3:
         plt.yscale("log")
    if mode == 0 or mode == 1 or mode == 3:
        ax.axvline(x=len(x),c=color)


def plot_ft():
    weight = 0.97 if ctrl < 4 else 0.7
    alpha = 0.1

    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0+0.0, box.y0 + box.height * 0.005, box.width * 1.1, box.height * 1.13])  # label in



    # ### SAC ###
    if ctrl == 0 or ctrl == 2 :
        # # # Hybrid #
        # simple_plot(ax, sac_files[4], weight, alpha, colors[1], 'P-9', '-' , linewidth=1)
        simple_plot(ax, sac_files[5], weight, alpha, colors[4], 'P-14', '-', linewidth=1)
        # simple_plot(ax, sac_files[6], weight, alpha, colors[3], 'P-19', '-', linewidth=1)
        simple_plot(ax, sac_files[7], weight, alpha, colors[7], 'P-24', '-', linewidth=1)
    elif ctrl == 1 or ctrl == 3:
        # # Impedance #
        simple_plot(ax, sac_files[0], weight, alpha, colors[0], 'A-8', '-', linewidth=1)
        simple_plot(ax, sac_files[1], weight, alpha, colors[5], 'A-13', '-', linewidth=1)
        simple_plot(ax, sac_files[2], weight, alpha, colors[2], 'A-13pd', '-', linewidth=1)
        simple_plot(ax, sac_files[3], weight, alpha, colors[6], 'A-18', '-', linewidth=1)
    elif ctrl == 4:
        # Ring
        simple_plot(ax, sac_ring_files[0], weight, alpha, colors[2], 'A-13pd', '-', linewidth=1)
        simple_plot(ax, sac_ring_files[1], weight, alpha, colors[3], 'P-14', '-', linewidth=1)
    elif ctrl == 5:
        # WRS
        simple_plot(ax, sac_wrs_files[0], weight, alpha, colors[0], 'A-13pd', '-', linewidth=1)
        simple_plot(ax, sac_wrs_files[1], weight, alpha, colors[1], 'P-14', '-', linewidth=1)


    ax.legend(loc='lower right', ncol=2, prop={'size': 15})
    ax.legend(loc='upper left', ncol=2, prop={'size': 15})
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13),
    #       fancybox=True, shadow=True, ncol=2)
    if mode == 0 or mode == 1 or mode == 3:
        plt.xlim((0,1850))
    if ctrl < 4:
        plt.ylim([-20, 220])
    else:
        plt.ylim([-20, 280])

    # plt.yticks(np.arange(0, 400, 100.0), fontsize=10)
    plt.yticks(fontsize=13)
    plt.xticks(fontsize=13)
    plt.grid(linestyle='--', linewidth=0.5)
    # if mode == 2 or mode == 4:
        # plt.xlabel("Steps", size='xx-large')
    # else:
    #     plt.xlabel("Episode", size='xx-large')
    # plt.ylabel("Reward", size='xx-large')
    # plt.ylabel("Distance (m)", size='x-large')
    # ax.set_rasterized(True)
    ax.xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
    # plt.show()
    plt.savefig('/home/cambel/dev/data/'+savename)


plot_ft()
