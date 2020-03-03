import csv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import matplotlib.cm as cm
import numpy as np

np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=np.inf)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

h_colors = ['darkblue', 'blue', '#1f77b4', 'cyan']

i_colors = ['green', 'lawngreen', 'aquamarine', 'cyan']

sh_colors = ['darkred', 'red', 'tomato','#ff7f0e']

si_colors = ['indigo', 'blueviolet', 'magenta', 'pink']

data = [
    '/media/Extra/research/real/gps/impedance/05/01-impedance8.npy', #good (new cost function)

    '/media/Extra/research/real/gps/impedance/04/02-impedance8.npy', #no good
    
    '/media/Extra/research/real/gps/impedance/03/impedance8.npy', # 5 iterations. no good

    '/media/Extra/research/real/gps/hybrid/01/01-hybrid9.npy',

    '/media/Extra/research/real/sac/01-hybrid9.csv',
    '/media/Extra/research/real/sac/01-impedance8.csv',
    '/media/Extra/research/real/sac/02-impedance8.csv',
    '/media/Extra/research/real/sac/03-impedance8.csv',

    '/media/Extra/research/real/wood_toy/gps/gpsimpedance18.npy',

    '/media/Extra/research/real/wood_toy/gps/hybrid/02-hybrid9.npy',
    '/media/Extra/research/real/wood_toy/gps/sacimpedance18.npy',

    '/media/Extra/research/real/wood_toy/sac/hybrid/01-hybrid9.csv',
    '/media/Extra/research/real/wood_toy/sac/impedance/01/01-impedance18.csv'
]

def csv_to_list(filename):
    with open(filename, 'r') as f:
        csv_data = list(csv.reader(f, delimiter=","))
    l = np.array(csv_data[:], dtype=np.float64).T
    print(l.shape)
    return l

def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
    return smoothed

def prepare_data(indices, weight):
    d = []
    for i in indices:
        tmp = None
        if data[i].endswith('.csv'):
            tmp = csv_to_list(data[i])[1]
            tmp = tmp[:45]
        else:
            tmp = npy_to_list(data[i])
        tmp = smooth(tmp, weight)
        d.append(tmp)
    x = []
    if data[i].endswith('.csv'):
        x = np.linspace(0, 200.0*len(d[0]), len(d[0]))
    else:
        x = np.linspace(0, 200.0*len(d[0]), len(d[0]))
    y = np.average(d, axis=0)
    y_std = np.std(d, axis=0) 
    return x, y, y_std

def npy_to_list(filename):
    data = np.load(filename)
    data = np.sum(data, axis=3)
    data = data.reshape(-1)
    print(data.shape)
    return data

def simple_plot(ax, data, weight, alpha, color, label, line_style):
    x,y,y_std = prepare_data(data, weight)
    c = color
    ax.plot(x, y, line_style, color=c, label=label)
    ax.fill_between(x, y-y_std, y+y_std, antialiased=True, linewidth=1, alpha=alpha, edgecolor=c, facecolor=c)

def plot_ft():
    weight = 0.75
    alpha = 0.1

    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0+0.01, box.y0 + box.height * 0.0, box.width * 1.09, box.height * 1.13]) # label in
    # ax.set_position([box.x0+0.01, box.y0 + box.height * 0.15, box.width * 1.09, box.height * 0.97]) # small graph
    # ax.set_position([box.x0+0.01, box.y0 + box.height * 0.25, box.width * 1.09, box.height * 0.89])

    ### GPS ###
    # hybrid #
    # simple_plot(ax, [3], weight, alpha, colors[0], 'GPS Hybrid [9]', '-')
    # simple_plot(ax, [4], weight, alpha, colors[1], 'SAC Hybrid [9]', '-')
    # simple_plot(ax, [3,4,5], weight, alpha, h_colors[1], 'GPS Hybrid [14]', '-')
    # simple_plot(ax, [6,7,8], weight, alpha, h_colors[2], 'GPS Hybrid [19]', '-')
    # simple_plot(ax,  [9,10], weight, alpha, h_colors[3], 'GPS Hybrid [24]', '-')

    # # impedance #
    # simple_plot(ax, [0], weight, alpha, colors[2], 'GPS Impedance [8]', '-')

    simple_plot(ax, [11], weight, alpha, colors[0], 'SAC Hybrid [9]', '-')
    simple_plot(ax, [9], weight, alpha, colors[1], 'GPS Hybrid [9]', '-')
    data_gps = np.load('/media/Extra/research/real/wood_toy/gps/gpsimpedance18.npy',allow_pickle=True)
    ax.plot(data_gps[0], data_gps[1], '-', color=colors[2], label='GPS Impedance [18]')
    simple_plot(ax, [12], weight, alpha, colors[3], 'SAC Impedance [18]', '-')

    # simple_plot(ax, [7], weight, alpha, colors[3], 'SAC Impedance [8]', '-')

    # simple_plot(ax, [1], weight, alpha, i_colors[1], 'GPS Impedance [8] old', '-')
    # simple_plot(ax, [2], weight, alpha, i_colors[2], 'GPS Impedance [8] old old', '-')

    # simple_plot(ax, [5], weight, alpha, si_colors[1], 'SAC Impedance [8]', '-')
    # simple_plot(ax, [6], weight, alpha, si_colors[2], 'SAC Impedance [8]', '-')

    ax.legend(loc='lower right', ncol=2)
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13),
    #       fancybox=True, shadow=True, ncol=2)
    # plt.xlim((0,t))
    # plt.yticks(np.arange(-1000, 200, 200.0))
    # plt.xticks(np.arange(-10, 12100, 2000))
    plt.grid(linestyle='--', linewidth=0.5)
    plt.xlabel("Steps", size='x-large')
    plt.ylabel("Cumulative reward", size='x-large')
    # plt.ylabel("Distance (m)", size='x-large')
    # ax.set_rasterized(True)
    ax.xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
    plt.show()
def reformat_large_tick_values(tick_val, pos):
    """
    Turns large tick values (in the billions, millions and thousands) such as 4500 into 4.5K and also appropriately turns 4000 into 4K (no zero after the decimal).
    """
    if tick_val >= 1000000000:
        val = round(tick_val/1000000000, 1)
        new_tick_format = '{:}B'.format(val)
    elif tick_val >= 1000000:
        val = round(tick_val/1000000, 1)
        new_tick_format = '{:}M'.format(val)
    elif tick_val >= 1000:
        val = round(tick_val/1000, 1)
        new_tick_format = '{:}K'.format(val)
    elif tick_val < 1000:
        new_tick_format = round(tick_val, 1)
    else:
        new_tick_format = tick_val

    # make new_tick_format into a string value
    new_tick_format = str(new_tick_format)
    
    # code below will keep 4.5M as is but change values such as 4.0M to 4M since that zero after the decimal isn't needed
    index_of_decimal = new_tick_format.find(".")
    
    if index_of_decimal != -1:
        value_after_decimal = new_tick_format[index_of_decimal+1]
        if value_after_decimal == "0":
            # remove the 0 after the decimal point since it's not needed
            new_tick_format = new_tick_format[0:index_of_decimal] + new_tick_format[index_of_decimal+2:]
            
    return new_tick_format

plot_ft()

# cost = np.load("/home/cambel/dev/container_gps/experiments/ur3/mdgps/hybrid/full/data_files/default/costs.npy")
# cost = np.sum(cost, axis=3)
# cost = cost.reshape(-1)
# print(cost.shape, cost)