import csv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import matplotlib.cm as cm
import numpy as np

np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=np.inf)

from plotter_utils import smooth, reformat_large_tick_values, csv_to_list, npy_to_list, prepare_data

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

h_colors = ['darkblue', 'blue', '#1f77b4', 'cyan']

i_colors = ['green', 'lawngreen', 'aquamarine', 'cyan']

sh_colors = ['darkred', 'red', 'tomato','#ff7f0e']

si_colors = ['indigo', 'blueviolet', 'magenta', 'pink']

data = [
    '/media/Extra/research/real/experimentsv2/peg-hybrid24.npy', 
    '/media/Extra/research/real/experimentsv2/peg-imp18.npy', 
    '/media/Extra/research/real/experimentsv2/wooden-hybrid24.npy', 
    '/media/Extra/research/real/experimentsv2/wooden-imp18.npy', 

    '/media/Extra/research/real/peg/sac/01-SAC_peg_hybrid14.csv',
    '/media/Extra/research/real/peg/sac/01-SAC_peg_imp13pd.csv',
    '/media/Extra/research/real/peg/sac/02-SAC_peg_imp13pd.csv',
    '/media/Extra/research/real/peg/gps/01-peg14.npy',    
    '/media/Extra/research/real/peg/gps/01-peg13pd.npy',

    '/media/Extra/research/real/wood_toy/sac/01-SAC_wood_toy_imp13pd.csv',
    '/media/Extra/research/real/wood_toy/sac/01-SAC_wood_toy_hybrid14.csv',
]

def simple_plot(ax, files, weight, alpha, color, label, line_style='-', linewidth=1.0):
    x,y,y_std = prepare_data(files, weight, 9000)
    c = color
    ax.plot(x, y, line_style, color=c, label=label, linewidth=linewidth)
    # ax.fill_between(x, y-y_std, y+y_std, antialiased=True, linewidth=0.5, alpha=alpha, edgecolor=c, facecolor=c)

def plot_ft():
    weight = 0.75
    alpha = 0.1

    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0+0.01, box.y0 + box.height * 0.0, box.width * 1.09, box.height * 1.13]) # label in
    
    # simple_plot(ax, [0], weight, alpha, colors[0], 'GPS Hybrid [24]', '-')
    # simple_plot(ax, [1], weight, alpha, colors[1], 'GPS Impedance [18]', '-')
    # simple_plot(ax, [2], weight, alpha, colors[2], 'GPS Hybrid [9]', '-')
    # simple_plot(ax, [3], weight, alpha, colors[3], 'GPS Impedance [18]', '-')

    simple_plot(ax, [data[7]], weight, alpha, colors[0], 'GPS Hybrid [14]', ':', linewidth=2)
    simple_plot(ax, [data[4]], weight, alpha, colors[0], 'SAC Hybrid [14]', '-')
    simple_plot(ax, [data[8]], weight, alpha, colors[1], 'GPS Impedance [13pd]', ':', linewidth=2)
    simple_plot(ax, data[5:7], weight, alpha, colors[1], 'SAC Impedance [13pd]', '-')

    # simple_plot(ax, [data[4]], weight, alpha, colors[2], 'SAC Hybrid [14]', '-')
    # simple_plot(ax, data[5:7], weight, alpha, colors[3], 'SAC Impedance [13pd]', '-')
    
    ax.legend(loc='lower right', ncol=2)
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13),
    #       fancybox=True, shadow=True, ncol=2)
    # plt.xlim((0,t))
    plt.yticks(np.arange(-1400, -200, 200.0))
    plt.xticks(np.arange(00, 10000, 1000))
    plt.grid(linestyle='--', linewidth=0.5)
    plt.xlabel("Step count", size='x-large')
    plt.ylabel("Reward", size='x-large')
    ax.xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
    plt.show()

plot_ft()
