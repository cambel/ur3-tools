import csv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=np.inf)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

h_colors = ['darkblue', 'blue', '#1f77b4', 'cyan']

i_colors = ['#9467bd', 'darkred', '#e377c2', '#ff7f0e']

sh_colors = ['darkgreen', 'green', 'lawngreen', 'lightgreen']

ih_colors = ['indigo', 'blueviolet', 'magenta', 'pink']

data = [
    "/media/cambel/Extra/research/data/training_force/sac/cumulative_reward/old2/01-SAC_hybrid_sim_9.csv",
    "/media/cambel/Extra/research/data/training_force/sac/cumulative_reward/old2/01-SAC_hybrid_sim_14.csv",
    "/media/cambel/Extra/research/data/training_force/sac/cumulative_reward/old2/01-SAC_hybrid_sim_19.csv",
    "/media/cambel/Extra/research/data/training_force/sac/cumulative_reward/old2/01-SAC_hybrid_sim_24.csv",
    "/media/cambel/Extra/research/data/training_force/sac/cumulative_reward/old2/02-SAC_hybrid_sim_14.csv",
    "/media/cambel/Extra/research/data/training_force/sac/cumulative_reward/old2/01-SAC_impedance_sim8.csv",
    "/media/cambel/Extra/research/data/training_force/sac/cumulative_reward/old2/01-SAC_impedance_sim13imp.csv",
    "/media/cambel/Extra/research/data/training_force/sac/cumulative_reward/old2/01-SAC_impedance_sim13pd.csv",
    "/media/cambel/Extra/research/data/training_force/sac/cumulative_reward/old2/01-SAC_impedance_sim18.csv",

    '/media/cambel/Extra/research/data/training_force/gps/cumulative_reward/old2/01-hybrid9.npy',
    '/media/cambel/Extra/research/data/training_force/gps/cumulative_reward/old2/01-hybrid14.npy',
    '/media/cambel/Extra/research/data/training_force/gps/cumulative_reward/old2/01-hybrid19.npy',
    '/media/cambel/Extra/research/data/training_force/gps/cumulative_reward/old2/01-hybrid24.npy',

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

    '/media/cambel/Extra/research/data/training_force/gps/cumulative_reward/old2/02-hybrid9.npy',
    '/media/cambel/Extra/research/data/training_force/gps/cumulative_reward/old2/02-hybrid14.npy',
    '/media/cambel/Extra/research/data/training_force/gps/cumulative_reward/old2/02-hybrid19.npy',
    '/media/cambel/Extra/research/data/training_force/gps/cumulative_reward/old2/02-hybrid24.npy',

    '/media/cambel/Extra/research/data/training_force/gps/cumulative_reward/old2/04-impedance8.npy',
    '/media/cambel/Extra/research/data/training_force/gps/cumulative_reward/old2/04-impedance13.npy',
    '/media/cambel/Extra/research/data/training_force/gps/cumulative_reward/old2/04-impedance13pd.npy',
    '/media/cambel/Extra/research/data/training_force/gps/cumulative_reward/old2/04-impedance18.npy',
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
        else:
            tmp = npy_to_list(data[i])
        tmp = smooth(tmp, weight)
        d.append(tmp)
    
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

def plot_ft():
    weight = 0.75
    alpha = 0.1

    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0+0.02, box.y0 + box.height * 0.3, box.width, box.height * 0.8])

    # x,y,y_std = prepare_data([0], weight)
    # c = sh_colors[0]
    # ax.plot(x, y, '--', color=c, label='SAC Hybrid [9]')
    # ax.fill_between(x, y-y_std, y+y_std, antialiased=True, linewidth=1, alpha=alpha, edgecolor=c, facecolor=c)

    # x,y,y_std = prepare_data([1,4], weight)
    # c = sh_colors[1]
    # ax.plot(x, y, '--', color=c, label='SAC Hybrid [14]')
    # ax.fill_between(x, y-y_std, y+y_std, antialiased=True, linewidth=1, alpha=alpha, edgecolor=c, facecolor=c)

    # x,y,y_std = prepare_data([2], weight)
    # c = sh_colors[2]
    # ax.plot(x, y, '--', color=c, label='SAC Hybrid [19]')
    # ax.fill_between(x, y-y_std, y+y_std, antialiased=True, linewidth=1, alpha=alpha, edgecolor=c, facecolor=c)

    # x,y,y_std = prepare_data([3], weight)
    # c = sh_colors[3]
    # ax.plot(x, y, '--', color=c, label='SAC Hybrid [24]')
    # ax.fill_between(x, y-y_std, y+y_std, antialiased=True, linewidth=1, alpha=alpha, edgecolor=c, facecolor=c)

    # # Impedance
    # x,y,y_std = prepare_data([5], weight)
    # c = colors[4]
    # ax.plot(x, y, '--', color=c, label='SAC Impedance [8]')
    # ax.fill_between(x, y-y_std, y+y_std, antialiased=True, linewidth=1, alpha=alpha, edgecolor=c, facecolor=c)

    # x,y,y_std = prepare_data([6], weight)
    # c = colors[5]
    # ax.plot(x, y, '--', color=c, label='SAC Impedance [13]')
    # ax.fill_between(x, y-y_std, y+y_std, antialiased=True, linewidth=1, alpha=alpha, edgecolor=c, facecolor=c)

    # x,y,y_std = prepare_data([7], weight)
    # c = colors[8]
    # ax.plot(x, y, '--', color=c, label='SAC Impedance [13pd]')
    # ax.fill_between(x, y-y_std, y+y_std, antialiased=True, linewidth=1, alpha=alpha, edgecolor=c, facecolor=c)

    # x,y,y_std = prepare_data([8], weight)
    # c = colors[9]
    # ax.plot(x, y, '--', color=c, label='SAC Impedance [18]')
    # ax.fill_between(x, y-y_std, y+y_std, antialiased=True, linewidth=1, alpha=alpha, edgecolor=c, facecolor=c)

    ### GPS ###
    # x,y,y_std = prepare_data([9,25], weight)
    # c = h_colors[0]
    # ax.plot(x, y, '-', color=c, label='GPS Hybrid [9]')
    # ax.fill_between(x, y-y_std, y+y_std, antialiased=True, linewidth=1, alpha=alpha, edgecolor=c, facecolor=c)

    # x,y,y_std = prepare_data([10,26], weight)
    # c = h_colors[1]
    # ax.plot(x, y, '-', color=c, label='GPS Hybrid [14]')
    # ax.fill_between(x, y-y_std, y+y_std, antialiased=True, linewidth=1, alpha=alpha, edgecolor=c, facecolor=c)

    # x,y,y_std = prepare_data([11,27], weight)
    # c = h_colors[2]
    # ax.plot(x, y, '-', color=c, label='GPS Hybrid [19]')
    # ax.fill_between(x, y-y_std, y+y_std, antialiased=True, linewidth=1, alpha=alpha, edgecolor=c, facecolor=c)

    # x,y,y_std = prepare_data([12,28], weight)
    # c = h_colors[3]
    # ax.plot(x, y, '-', color=c, label='GPS Hybrid [24]')
    # ax.fill_between(x, y-y_std, y+y_std, antialiased=True, linewidth=1, alpha=alpha, edgecolor=c, facecolor=c)

    x,y,y_std = prepare_data([13,14,15], weight)
    c = colors[4]
    ax.plot(x, y, '-', color=c, label='GPS Impedance [8]')
    ax.fill_between(x, y-y_std, y+y_std, antialiased=True, linewidth=1, alpha=alpha, edgecolor=c, facecolor=c)

    x,y,y_std = prepare_data([16,17,18], weight)
    c = colors[5]
    ax.plot(x, y, '-', color=c, label='GPS Impedance [13]')
    ax.fill_between(x, y-y_std, y+y_std, antialiased=True, linewidth=1, alpha=alpha, edgecolor=c, facecolor=c)

    x,y,y_std = prepare_data([19,20,21], weight)
    c = colors[8]
    ax.plot(x, y, '-', color=c, label='GPS Impedance [13pd]')
    ax.fill_between(x, y-y_std, y+y_std, antialiased=True, linewidth=1, alpha=alpha, edgecolor=c, facecolor=c)

    x,y,y_std = prepare_data([22,23,24], weight)
    c = colors[9]
    ax.plot(x, y, '-', color=c, label='GPS Impedance [18]')
    ax.fill_between(x, y-y_std, y+y_std, antialiased=True, linewidth=1, alpha=alpha, edgecolor=c, facecolor=c)


    # x,y,y_std = prepare_data([29], weight)
    # c = colors[4]
    # ax.plot(x, y, '--', color=c, label='GPS Impedance [8]')
    # ax.fill_between(x, y-y_std, y+y_std, antialiased=True, linewidth=1, alpha=alpha, edgecolor=c, facecolor=c)

    # x,y,y_std = prepare_data([30], weight)
    # c = colors[5]
    # ax.plot(x, y, '--', color=c, label='GPS Impedance [13]')
    # ax.fill_between(x, y-y_std, y+y_std, antialiased=True, linewidth=1, alpha=alpha, edgecolor=c, facecolor=c)

    # x,y,y_std = prepare_data([31], weight)
    # c = colors[6]
    # ax.plot(x, y, '--', color=c, label='GPS Impedance [13pd]')
    # ax.fill_between(x, y-y_std, y+y_std, antialiased=True, linewidth=1, alpha=alpha, edgecolor=c, facecolor=c)

    # x,y,y_std = prepare_data([32], weight)
    # c = colors[7]
    # ax.plot(x, y, '--', color=c, label='GPS Impedance [18]')
    # ax.fill_between(x, y-y_std, y+y_std, antialiased=True, linewidth=1, alpha=alpha, edgecolor=c, facecolor=c)

    # ax.plot(d3[0], d3[1], 'k', color=c, label='Impedance')
    # ax.plot(d4[0], d4[1], 'k', color=c, label='Hybrid')
    # ax.plot(xi, adm_i[v], '--', color=c, label='impedance')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
          fancybox=True, shadow=True, ncol=2)
    # plt.xlim((0,t))
    plt.yticks(np.arange(-2000, 100, 250.0))
    # plt.xticks(np.arange(0, t, 1.0))
    plt.grid(linestyle='--', linewidth=0.5)
    plt.xlabel("Steps", size='x-large')
    plt.ylabel("Cumulative reward", size='x-large')
    # plt.ylabel("Distance (m)", size='x-large')

    plt.show()

plot_ft()

# cost = np.load("/home/cambel/dev/container_gps/experiments/ur3/mdgps/hybrid/full/data_files/default/costs.npy")
# cost = np.sum(cost, axis=3)
# cost = cost.reshape(-1)
# print(cost.shape, cost)