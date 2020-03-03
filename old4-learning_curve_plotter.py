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

    # '/media/Extra/research/data/training_force2/bad/9/data_files/default1/costs.npy',

    '/media/Extra/research/data/training_force2/gps/impedance/01/01-impedance8.npy',
    '/media/Extra/research/data/training_force2/gps/impedance/01/02-impedance8.npy',
    '/media/Extra/research/data/training_force2/gps/impedance/01/03-impedance8.npy',
    '/media/Extra/research/data/training_force2/gps/impedance/01/01-impedance13.npy',
    '/media/Extra/research/data/training_force2/gps/impedance/01/02-impedance13.npy',
    # '/media/Extra/research/data/training_force2/gps/impedance/01/03-impedance13.npy',
    '/media/Extra/research/data/training_force2/gps/impedance/01/01-impedance13pd.npy',
    # '/media/Extra/research/data/training_force2/gps/impedance/01/02-impedance13pd.npy',
    # '/media/Extra/research/data/training_force2/gps/impedance/01/03-impedance13pd.npy',
    '/media/Extra/research/data/training_force2/gps/impedance/01/01-impedance18.npy',
    '/media/Extra/research/data/training_force2/gps/impedance/01/02-impedance18.npy',
    # '/media/Extra/research/data/training_force2/gps/impedance/01/03-impedance18.npy',

    '/media/Extra/research/data/training_force/gps/cumulative_reward/01-impedance8.npy',
    '/media/Extra/research/data/training_force/gps/cumulative_reward/02-impedance8.npy',
    '/media/Extra/research/data/training_force/gps/cumulative_reward/03-impedance8.npy',
    '/media/Extra/research/data/training_force/gps/cumulative_reward/01-impedance13.npy',
    '/media/Extra/research/data/training_force/gps/cumulative_reward/02-impedance13.npy',
    '/media/Extra/research/data/training_force/gps/cumulative_reward/03-impedance13.npy',
    '/media/Extra/research/data/training_force/gps/cumulative_reward/01-impedance13pd.npy',
    '/media/Extra/research/data/training_force/gps/cumulative_reward/02-impedance13pd.npy',
    '/media/Extra/research/data/training_force/gps/cumulative_reward/03-impedance13pd.npy',
    '/media/Extra/research/data/training_force/gps/cumulative_reward/01-impedance18.npy',
    '/media/Extra/research/data/training_force/gps/cumulative_reward/02-impedance18.npy',
    '/media/Extra/research/data/training_force/gps/cumulative_reward/03-impedance18.npy',

    "/media/Extra/research/data/training_force2/cumulative_reward/01-SAC_impedance_sim8.csv",
    "/media/Extra/research/data/training_force2/cumulative_reward/02-SAC_impedance_sim8.csv",
    "/media/Extra/research/data/training_force2/cumulative_reward/03-SAC_impedance_sim8.csv",
    "/media/Extra/research/data/training_force2/cumulative_reward/01-SAC_impedance_sim13imp.csv",
    "/media/Extra/research/data/training_force2/cumulative_reward/02-SAC_impedance_sim13imp.csv",
    "/media/Extra/research/data/training_force2/cumulative_reward/03-SAC_impedance_sim13imp.csv",
    "/media/Extra/research/data/training_force2/cumulative_reward/01-SAC_impedance_sim13pd.csv",
    "/media/Extra/research/data/training_force2/cumulative_reward/02-SAC_impedance_sim13pd.csv",
    "/media/Extra/research/data/training_force2/cumulative_reward/03-SAC_impedance_sim13pd.csv",
    "/media/Extra/research/data/training_force2/cumulative_reward/01-SAC_impedance_sim18.csv",
    "/media/Extra/research/data/training_force2/cumulative_reward/02-SAC_impedance_sim18.csv",
    "/media/Extra/research/data/training_force2/cumulative_reward/03-SAC_impedance_sim18.csv",

    '/media/cambel/Extra/research/data/training_force/gps/cumulative_reward/01-hybrid9.npy',
    '/media/cambel/Extra/research/data/training_force/gps/cumulative_reward/02-hybrid9.npy',
    '/media/cambel/Extra/research/data/training_force/gps/cumulative_reward/03-hybrid9.npy',
    '/media/cambel/Extra/research/data/training_force/gps/cumulative_reward/01-hybrid14.npy',
    '/media/cambel/Extra/research/data/training_force/gps/cumulative_reward/02-hybrid14.npy',
    '/media/cambel/Extra/research/data/training_force/gps/cumulative_reward/03-hybrid14.npy',
    '/media/cambel/Extra/research/data/training_force/gps/cumulative_reward/01-hybrid19.npy',
    '/media/cambel/Extra/research/data/training_force/gps/cumulative_reward/02-hybrid19.npy',
    '/media/cambel/Extra/research/data/training_force/gps/cumulative_reward/03-hybrid19.npy',
    '/media/cambel/Extra/research/data/training_force/gps/cumulative_reward/01-hybrid24.npy',
    '/media/cambel/Extra/research/data/training_force/gps/cumulative_reward/02-hybrid24.npy',

    '/media/cambel/Extra/research/data/training_force/gps/cumulative_reward/04-hybrid9.npy',
    '/media/cambel/Extra/research/data/training_force/gps/cumulative_reward/05-hybrid9.npy',
    '/media/cambel/Extra/research/data/training_force/gps/cumulative_reward/06-hybrid9.npy',
    '/media/cambel/Extra/research/data/training_force/gps/cumulative_reward/04-hybrid14.npy',
    '/media/cambel/Extra/research/data/training_force/gps/cumulative_reward/05-hybrid14.npy',
    '/media/cambel/Extra/research/data/training_force/gps/cumulative_reward/06-hybrid14.npy',
    '/media/cambel/Extra/research/data/training_force/gps/cumulative_reward/04-hybrid19.npy',
    '/media/cambel/Extra/research/data/training_force/gps/cumulative_reward/05-hybrid19.npy',
    '/media/cambel/Extra/research/data/training_force/gps/cumulative_reward/06-hybrid19.npy',
    '/media/cambel/Extra/research/data/training_force/gps/cumulative_reward/04-hybrid24.npy',
    '/media/cambel/Extra/research/data/training_force/gps/cumulative_reward/05-hybrid24.npy',
    '/media/cambel/Extra/research/data/training_force/gps/cumulative_reward/06-hybrid24.npy',
 
    "/media/cambel/Extra/research/data/training_force/sac/cumulative_reward/01-SAC_hybrid_sim_9.csv",
    "/media/cambel/Extra/research/data/training_force/sac/cumulative_reward/02-SAC_hybrid_sim_9.csv",
    "/media/cambel/Extra/research/data/training_force/sac/cumulative_reward/03-SAC_hybrid_sim_9.csv",
    "/media/cambel/Extra/research/data/training_force/sac/cumulative_reward/01-SAC_hybrid_sim_14.csv",
    "/media/cambel/Extra/research/data/training_force/sac/cumulative_reward/02-SAC_hybrid_sim_14.csv",
    "/media/cambel/Extra/research/data/training_force/sac/cumulative_reward/03-SAC_hybrid_sim_14.csv",
    "/media/cambel/Extra/research/data/training_force/sac/cumulative_reward/01-SAC_hybrid_sim_19.csv",
    "/media/cambel/Extra/research/data/training_force/sac/cumulative_reward/02-SAC_hybrid_sim_19.csv",
    "/media/cambel/Extra/research/data/training_force/sac/cumulative_reward/03-SAC_hybrid_sim_19.csv",
    "/media/cambel/Extra/research/data/training_force/sac/cumulative_reward/01-SAC_hybrid_sim_24.csv",
    "/media/cambel/Extra/research/data/training_force/sac/cumulative_reward/02-SAC_hybrid_sim_24.csv",
    "/media/cambel/Extra/research/data/training_force/sac/cumulative_reward/03-SAC_hybrid_sim_24.csv",


    '/media/cambel/Extra/research/data/training_force/gps/cumulative_reward/old2/01-impedance8.npy',
    '/media/cambel/Extra/research/data/training_force/gps/cumulative_reward/old2/02-impedance8.npy',
    '/media/cambel/Extra/research/data/training_force/gps/cumulative_reward/old2/03-impedance8.npy',
    '/media/cambel/Extra/research/data/training_force/gps/cumulative_reward/old2/01-impedance13.npy',
    '/media/cambel/Extra/research/data/training_force/gps/cumulative_reward/old2/02-impedance13.npy',
    '/media/cambel/Extra/research/data/training_force/gps/cumulative_reward/old2/03-impedance13.npy',
    '/media/cambel/Extra/research/data/training_force/gps/cumulative_reward/old2/01-impedance13pd.npy',
    '/media/cambel/Extra/research/data/training_force/gps/cumulative_reward/old2/02-impedance13pd.npy',
    '/media/cambel/Extra/research/data/training_force/gps/cumulative_reward/old2/03-impedance13pd.npy',
    '/media/cambel/Extra/research/data/training_force/gps/cumulative_reward/old2/01-impedance18.npy',
    '/media/cambel/Extra/research/data/training_force/gps/cumulative_reward/old2/02-impedance18.npy',
    '/media/cambel/Extra/research/data/training_force/gps/cumulative_reward/old2/03-impedance18.npy',
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
            tmp = tmp[:]
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
    # print(data.shape)
    return data

def simple_plot(ax, data, weight, alpha, color, label, line_style='-', linewidth=1.0):
    x,y,y_std = prepare_data(data, weight)
    c = color
    ax.plot(x, y, line_style, color=c, label=label, linewidth=linewidth)
    # ax.fill_between(x, y-y_std, y+y_std, antialiased=True, linewidth=0.5, alpha=alpha, edgecolor=c, facecolor=c)

def plot_ft():
    weight = 0.75
    alpha = 0.01

    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0+0.01, box.y0 + box.height * 0.0, box.width * 1.09, box.height * 1.13]) # label in
    # ax.set_position([box.x0+0.01, box.y0 + box.height * 0.15, box.width * 1.09, box.height * 0.97]) # small graph
    # ax.set_position([box.x0+0.01, box.y0 + box.height * 0.25, box.width * 1.09, box.height * 0.89])

    # simple_plot(ax, [0], weight, alpha, h_colors[0], 'GPS Hybrid [9]')

    ### GPS ###
    # hybrid #
    # simple_plot(ax, [0,1,2], weight, alpha, h_colors[0], 'GPS Hybrid [9]')
    # simple_plot(ax, [3,4,5], weight, alpha, h_colors[1], 'GPS Hybrid [14]')
    # simple_plot(ax, [6,7,8], weight, alpha, h_colors[2], 'GPS Hybrid [19]')
    # simple_plot(ax,  [9,10], weight, alpha, h_colors[3], 'GPS Hybrid [24]')

    # # impedance #
    simple_plot(ax, [0,1,2], weight, alpha, colors[0], 'GPS Impedance [8]',   linewidth=1)
    simple_plot(ax, [3,4]  , weight, alpha, colors[1], 'GPS Impedance [13]',  linewidth=1)
    simple_plot(ax, [5]    , weight, alpha, colors[2], 'GPS Impedance [13pd]',linewidth=1)
    simple_plot(ax, [6]  , weight, alpha, colors[3], 'GPS Impedance [18]',  linewidth=1)

    # simple_plot(ax, [8,9,10]  , weight, alpha, colors[0], 'GPS Impedance [8] euler',    '--', linewidth=1.2)
    # simple_plot(ax, [11,12,13], weight, alpha, colors[1], 'GPS Impedance [13] euler',   '--', linewidth=1.2)
    # simple_plot(ax, [14,15,16], weight, alpha, colors[2], 'GPS Impedance [13pd] euler', '--', linewidth=1.2)
    # simple_plot(ax, [17,18,19], weight, alpha, colors[3], 'GPS Impedance [18] euler',   '--', linewidth=1.2)

    # # Impedance #
    # simple_plot(ax, [20,21,22], weight, alpha, colors[0], 'SAC Impedance [8]', ':')
    # simple_plot(ax, [23,24,25], weight, alpha, colors[1], 'SAC Impedance [13]', ':')
    # simple_plot(ax, [26,27,28], weight, alpha, colors[2], 'SAC Impedance [13pd]', ':')
    # simple_plot(ax, [29,30,31], weight, alpha, colors[3], 'SAC Impedance [18]', ':')


    # GPS 2nd set#
    # simple_plot(ax, [23,24,25], weight, alpha, h_colors[0], 'GPS Hybrid [9]')
    # simple_plot(ax, [26,27,28], weight, alpha, h_colors[1], 'GPS Hybrid [14]')
    # simple_plot(ax, [29,30,31], weight, alpha, h_colors[2], 'GPS Hybrid [19]')
    # simple_plot(ax, [32,33,34], weight, alpha, h_colors[3], 'GPS Hybrid [24]')

    # ### SAC ###
    # # Hybrid #
    # simple_plot(ax, [35,36,37], weight, alpha, sh_colors[0], 'SAC Hybrid [9]', '--')
    # simple_plot(ax, [38,39,40], weight, alpha, sh_colors[1], 'SAC Hybrid [14]', '--')
    # simple_plot(ax, [41,42,43], weight, alpha, sh_colors[2], 'SAC Hybrid [19]', '--')
    # simple_plot(ax, [44,45,46], weight, alpha, sh_colors[3], 'SAC Hybrid [24]', '--')

    
    ## GPS vs SAC
    # simple_plot(ax, [23+i for i in range(11)], weight, alpha, colors[2], 'GPS Hybrid ALL')
    # simple_plot(ax, [11+i for i in range(11)], weight, alpha, colors[0], 'GPS Impedance ALL')
    
    # simple_plot(ax, [34+i for i in range(11)], weight, alpha, colors[1], 'SAC Hybrid ALL')
    # simple_plot(ax, [47+i for i in range(11)], weight, alpha, colors[3], 'SAC Impedance ALL')

    ## GPS set 1 vs 2
    # simple_plot(ax, [i for i in range(11)], weight, alpha, colors[2], 'GPS Hybrid ALL set-1')
    # simple_plot(ax, [23+i for i in range(11)], weight, alpha, colors[0], 'GPS Hybrid ALL set-2')

    # simple_plot(ax, [11+i for i in range(11)], weight, alpha, colors[1], 'GPS Impedance ALL set-1')
    # simple_plot(ax, [59+i for i in range(11)], weight, alpha, colors[3], 'GPS Impedance ALL set-2')

    ax.legend(loc='lower right', ncol=2)
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13),
    #       fancybox=True, shadow=True, ncol=2)
    # plt.xlim((0,t))
    plt.yticks(np.arange(-2500, 100, 300.0))
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