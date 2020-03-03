import csv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import matplotlib.cm as cm
import os
import numpy as np
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=np.inf)

from plotter_utils import smooth, reformat_large_tick_values, csv_to_list, npy_to_list, prepare_data

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def get_files(folder, data, keys):
    res = []
    for k in keys:
        res.append([folder + d for d in data if k in d])
    return res

imp_folder = '/media/cambel/Extra/research/IROS2020/data/all/gps/impedance/'
data = os.listdir(imp_folder)
imp_keys = ['imp8.npy', 'imp13.npy', 'imp13pd.npy', 'imp18.npy']
imp_files = get_files(imp_folder, data, imp_keys)

hyb_folder = '/media/cambel/Extra/research/IROS2020/data/all/gps/hybrid/'
data = os.listdir(hyb_folder)
hyb_keys = ['hyb9.npy', 'hyb14.npy', 'hyb19.npy', 'hyb24.npy']
hyb_files = get_files(hyb_folder, data, hyb_keys)

sac_folder = '/media/cambel/Extra/research/IROS2020/data/all/sac/'
data = os.listdir(sac_folder)
sac_keys = ['imp8.csv', 'imp13imp.csv', 'imp13pd.csv', 'imp18.csv', 'hyb9.csv', 'hyb14.csv', 'hyb19.csv', 'hyb24.csv']
sac_files = get_files(sac_folder, data, sac_keys)

def simple_plot(ax, files, weight, alpha, color, label, line_style='-', linewidth=1.0):
    x,y,y_std = prepare_data(files, weight)
    # np.savetxt(label+'.csv', np.array([x,y,y_std]).T, delimiter=',', fmt='%f')
    c = color
    ax.plot(x, y, line_style, color=c, label=label, linewidth=linewidth)
    ax.fill_between(x, y-y_std, y+y_std, antialiased=True, linewidth=0.5, alpha=alpha, edgecolor=c, facecolor=c)

def plot_ft():
    weight = 0.6
    alpha = 0.1

    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0+0.05, box.y0 + box.height * 0.005, box.width * 1.061, box.height * 1.13]) # label in
    # ax.set_position([box.x0+0.01, box.y0 + box.height * 0.15, box.width * 1.09, box.height * 0.97]) # small graph
    # ax.set_position([box.x0+0.01, box.y0 + box.height * 0.25, box.width * 1.09, box.height * 0.89])

    # simple_plot(ax, [0], weight, alpha, h_colors[0], 'GPS H-9')

    ### GPS ###
    simple_plot(ax, imp_files[0], weight, alpha, colors[0], 'GPS I-8',  '-', linewidth=1.5)
    simple_plot(ax, imp_files[1], weight, alpha, colors[1], 'GPS I-13', '-', linewidth=1.5)
    simple_plot(ax, imp_files[2]  , weight, alpha, colors[2], 'GPS I-13pd', '-', linewidth=1.5)
    simple_plot(ax, imp_files[3]  , weight, alpha, colors[3], 'GPS I-18',   '-', linewidth=1.5)

    # # hybrid #
    simple_plot(ax, hyb_files[0], weight, alpha, colors[0], 'GPS H-9' ,  '-', linewidth=1.5)
    simple_plot(ax, hyb_files[1], weight, alpha, colors[1], 'GPS H-14',  '-', linewidth=1.5)
    simple_plot(ax, hyb_files[2], weight, alpha, colors[2], 'GPS H-19',  '-', linewidth=1.5)
    simple_plot(ax, hyb_files[3], weight, alpha, colors[3], 'GPS H-24',  '-', linewidth=1.5)

    # ### SAC ###
    # # Hybrid #
    simple_plot(ax, sac_files[4], weight, alpha, colors[4], 'SAC H-9', '-' , linewidth=1.5)
    simple_plot(ax, sac_files[5], weight, alpha, colors[5], 'SAC H-14', '-', linewidth=1.5)
    simple_plot(ax, sac_files[6], weight, alpha, colors[6], 'SAC H-19', '-', linewidth=1.5)
    simple_plot(ax, sac_files[7], weight, alpha, colors[7], 'SAC H-24', '-', linewidth=1.5)

    # # Impedance #
    simple_plot(ax, sac_files[0], weight, alpha, colors[4], 'SAC I-8', '-', linewidth=1.5)
    simple_plot(ax, sac_files[1], weight, alpha, colors[5], 'SAC I-13', '-', linewidth=1.5)
    simple_plot(ax, sac_files[2], weight, alpha, colors[6], 'SAC I-13pd', '-', linewidth=1.5)
    simple_plot(ax, sac_files[3], weight, alpha, colors[7], 'SAC I-18', '-', linewidth=1.5)
    
    ax.legend(loc='lower right', ncol=2, prop={'size': 17})
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13),
    #       fancybox=True, shadow=True, ncol=2)
    # plt.xlim((0,t))
    plt.ylim([-2100, -300])
    plt.yticks(np.arange(-2100, 0, 300.0), fontsize=14)
    plt.xticks(fontsize=15)
    plt.grid(linestyle='--', linewidth=0.5)
    plt.xlabel("Step count", size='xx-large')
    plt.ylabel("Reward", size='xx-large')
    # plt.ylabel("Distance (m)", size='x-large')
    # ax.set_rasterized(True)
    ax.xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
    plt.show()

plot_ft()
