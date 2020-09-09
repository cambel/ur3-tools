from plotter_utils import smooth, reformat_large_tick_values, csv_to_list, npy_to_list, extrapolate, running_mean
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

mode = 2
weight = 0.97
# filename ="/media/cambel/Extra/research/MDPI/simulation/conical/retrain_randerror_SAC_conical_p25"
# filename1 ="/media/cambel/Extra/research/MDPI/simulation/tcn-vs-default/randinit100k-force5N_SAC_wave-parallel-14"
# filename2 ="/media/cambel/Extra/research/MDPI/simulation/tcn-vs-default/randinit75k-force5N_SAC_default-parallel-14"

filename1 ="/media/cambel/Extra/research/MDPI/real/sim-vs-scratch/retrain_sim_SAC_real_square_peg"
filename2 ="/media/cambel/Extra/research/MDPI/real/sim-vs-scratch/scratch_SAC_real_square_peg"
filename3 ="/media/cambel/Extra/research/MDPI/real/square_peg/retrain_sim_x6_SAC_real_square_peg"

data = [
"/media/cambel/Extra/research/MDPI/simulation/batch_size_tests/rand_seed/20200628T130622.536582_SAC_randinit_512_p14",
"/media/cambel/Extra/research/MDPI/simulation/batch_size_tests/seed_10/20200628T201704.635845_SAC_randinit_512_p14",
"/media/cambel/Extra/research/MDPI/simulation/batch_size_tests/rand_seed/20200628T153412.691297_SAC_randinit_1024_p14",
"/media/cambel/Extra/research/MDPI/simulation/batch_size_tests/seed_10/20200628T224003.028556_SAC_randinit_1024_p14",
"/media/cambel/Extra/research/MDPI/simulation/batch_size_tests/rand_seed/20200628T175417.109277_SAC_randinit_2048_p14",
"/media/cambel/Extra/research/MDPI/simulation/batch_size_tests/seed_10/20200629T005902.167328_SAC_randinit_2048_p14",
"/media/cambel/Extra/research/MDPI/simulation/batch_size_tests/seed_10/20200629T031256.858042_SAC_randinit_4096_p14",
"/media/cambel/Extra/research/MDPI/simulation/tcn/std_01/randinit100k_SAC_wave-parallel-14",
]

data = [
'/media/cambel/Extra/research/MDPI/simulation/domain-rand/fuji/20200710T090556.990409_SAC_randinit_p14',
'/media/cambel/Extra/research/MDPI/simulation/domain-rand/individual/20200710T090654.918739_SAC_randinit_p14',
'/media/cambel/Extra/research/MDPI/simulation/domain-rand/fuji/20200710T110146.101638_SAC_randgoal_p14',
'/media/cambel/Extra/research/MDPI/simulation/domain-rand/individual/20200710T105808.068308_SAC_randgoal_p14',
'/media/cambel/Extra/research/MDPI/simulation/domain-rand/fuji/20200710T131205.269642_SAC_randstiff_p14',
'/media/cambel/Extra/research/MDPI/simulation/domain-rand/individual/20200713T221724.965564_SAC_randstiff_p14',
'/media/cambel/Extra/research/MDPI/simulation/domain-rand/fuji/20200710T153342.717334_SAC_randerror_p14',
'/media/cambel/Extra/research/MDPI/simulation/domain-rand/individual/20200710T151154.699193_SAC_randerror_p14',
'/media/cambel/Extra/research/MDPI/simulation/domain-rand/fuji/20200710T174513.021665_SAC_rand_all_p14',
'/media/cambel/Extra/research/MDPI/simulation/domain-rand/individual/20200714T001244.249073_SAC_rand_all_p14',
]
# data = [
# '/home/cambel/dev/results/SAC_randgoal_p14',
# '/home/cambel/dev/results/SAC_randstiff_p14',
# '/home/cambel/dev/results/SAC_randerror_p14',
# '/home/cambel/dev/results/SAC_randerror_x_step_p14',
# ]
# data = [
# '/media/cambel/Extra/research/MDPI/simulation/domain-rand/individual/20200710T090654.918739_SAC_randinit_p14',
# '/media/cambel/Extra/research/MDPI/simulation/domain-rand/individual/p_24/20200715T182836.015094_SAC_randinit_p24',
# '/media/cambel/Extra/research/MDPI/simulation/domain-rand/individual/20200710T105808.068308_SAC_randgoal_p14',
# '/media/cambel/Extra/research/MDPI/simulation/domain-rand/individual/p_24/20200715T232352.240061_SAC_randgoal_p24',
# ]
# filename1 = '/media/cambel/Extra/research/MDPI/simulation/domain-rand/individual/20200711T211003.734830_SAC_rand_error_5mm_p14'

data = [
'/media/cambel/Extra/research/MDPI/simulation/domain_rand/20200805T215202.243682_SAC_randinit_p24',
'/media/cambel/Extra/research/MDPI/simulation/domain_rand/20200806T122312.370407_SAC_randgoal_p24',
'/media/cambel/Extra/research/MDPI/simulation/domain_rand/20200806T154222.166939_SAC_randstiff_p24',
'/media/cambel/Extra/research/MDPI/simulation/domain_rand/20200806T191441.179942_SAC_rand_all_p24',
'/media/cambel/Extra/research/MDPI/simulation/domain_rand/20200806T222948.962759_SAC_randerror_p24',
]

data = [
'/media/cambel/Extra/research/MDPI/simulation/tcn-vs-default/version2/20200804T190253.159771_SAC_tcn_policy',
'/media/cambel/Extra/research/MDPI/simulation/tcn-vs-default/version2/20200804T135723.293938_SAC_tcn_policy',
'/media/cambel/Extra/research/MDPI/simulation/tcn-vs-default/version2/20200804T213248.774807_SAC_nn_policy',
'/media/cambel/Extra/research/MDPI/simulation/tcn-vs-default/version2/20200804T162355.027278_SAC_nn_policy',
]

data = [
'/media/cambel/Extra/research/MDPI/simulation/curriculum24/force_input/SAC_baseline',
# '/media/cambel/Extra/research/MDPI/simulation/tcn-vs-default/version2/20200804T190253.159771_SAC_tcn_policy',
# '/media/cambel/Extra/research/MDPI/simulation/domain_rand/20200805T215202.243682_SAC_randinit_p24',
'/media/cambel/Extra/research/MDPI/simulation/curriculum24/force_input/SAC_no_action_input',
'/media/cambel/Extra/research/MDPI/simulation/curriculum24/force_input/SAC_force_input_cost',
# '/media/cambel/Extra/research/MDPI/simulation/curriculum24/force_input/SAC_no_force_input',
'/media/cambel/Extra/research/MDPI/simulation/curriculum24/force_input/SAC_force_input_wrt_goal',
'/media/cambel/Extra/research/MDPI/simulation/curriculum24/force_input/SAC_baseline_random',
'/media/cambel/Extra/research/MDPI/simulation/curriculum24/force_input/SAC_force_input_random',
]

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
        data = np.load(f+"/detailed_log.npy", allow_pickle=True)
        data = data[15:]
        tmp = None
        # if mode == 2 or mode == 4:
        #     index = 3 if mode == 0 or mode == 2 else 2
        #     _, tmp = extrapolate(data[:,1], data[:, index], points)
        if mode >= 3:
            index = 3
            tmp = cumulative_value(np.copy(data[:, index]))
        else:
            index = 3 if mode == 0 or mode == 2 else 2
            tmp = smooth(data[:, index], weight)
            # print("exp",np.array(tmp).shape)
            # tmp = running_mean(tmp, 5000)
            # print("running_mean",tmp.shape)
        if mode == 2 or mode == 4:
            index = 3 if mode == 0 or mode == 2 else 2
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


def simple_plot(ax, files, weight, alpha, color, label, line_style='-', linewidth=1.0, points=50000):
    # points = 50000 
    x, y, y_std = prepare_data(files, weight, points=points)
    # y = y * 200. / 350.
    # np.savetxt(label+'.csv', np.array([x,y,y_std]).T, delimiter=',', fmt='%f')
    c = color
    ax.plot(x, y, line_style, color=c, label=label, linewidth=linewidth)
    ax.fill_between(x, y-y_std, y+y_std, antialiased=True, linewidth=0.5, alpha=alpha, edgecolor=c, facecolor=c)
    if mode >= 3:
         plt.yscale("log")
    if mode == 0 or mode == 1 or mode == 3:
        ax.axvline(x=len(x),c=color)


def plot_ft():
    alpha = 0.1

    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0+0.0, box.y0 + box.height * 0.005, box.width * 1.1, box.height * 1.13])  # label in

    # simple_plot(ax, [filename1], weight, alpha, colors[1], 'TCN-64d', '-', linewidth=1)
    # simple_plot(ax, data[:2], weight, alpha, colors[0], 'TCN network', '-', linewidth=1)
    # simple_plot(ax, data[2:], weight, alpha, colors[1], 'simple NN', '-', linewidth=1)

    # simple_plot(ax, [filename2], weight, alpha, colors[0], 'scratch', '-', linewidth=1)
    # simple_plot(ax, [filename1], weight, alpha, colors[1], 'retrain-from-simulation(manual)', '-', linewidth=1, points=20000)
    # simple_plot(ax, [filename3], weight, alpha, colors[2], 'retrain-from-simulation', '-', linewidth=1, points=15000)
    # ax.axhline(y=182, color='gray', linestyle='--')
    # ax.axvline(x=2000, color='gray', linestyle='--')
    
    # simple_plot(ax, [data[7]], weight, alpha, colors[4], 'Batch size 256', '-', linewidth=1)
    # simple_plot(ax, data[:2], weight, alpha, colors[0], 'Batch size 512', '-', linewidth=1)
    # simple_plot(ax, data[2:4], weight, alpha, colors[1], 'Batch size 1024', '-', linewidth=1)
    # simple_plot(ax, data[4:6], weight, alpha, colors[2], 'Batch size 2048', '-', linewidth=1)
    # simple_plot(ax, [data[6]], weight, alpha, colors[3], 'Batch size 4096', '-', linewidth=1)

    # simple_plot(ax, data[:2], weight, alpha, colors[0], 'Random initial pose', '-', linewidth=1)
    # simple_plot(ax, data[2:4], weight, alpha, colors[1], 'Random goal', '-', linewidth=1)
    # simple_plot(ax, data[4:6], weight, alpha, colors[2], 'Random stiffness', '-', linewidth=1)
    # simple_plot(ax, data[6:8], weight, alpha, colors[3], 'Random uncertainty error', '-', linewidth=1)
    # simple_plot(ax, data[8:], weight, alpha, colors[5], 'Random All', '-', linewidth=1)

    # simple_plot(ax, [data[0]], weight, alpha, colors[0], 'Base Random initial pose', '-', linewidth=1)
    # simple_plot(ax, [data[1]], weight, alpha, colors[1], '1) Random goal', '-', linewidth=1)
    # simple_plot(ax, [data[2]], weight, alpha, colors[2], '2) Random stiffness', '-', linewidth=1)
    # simple_plot(ax, [data[3]], weight, alpha, colors[3], '3) Random uncertainty error', '-', linewidth=1)
    # simple_plot(ax, [data[4]], weight, alpha, colors[4], '4) Random All', '-', linewidth=1)

    # simple_plot(ax, [data[0]], weight, alpha, colors[4], 'Parallel-14', '-', linewidth=1)
    # simple_plot(ax, [data[1]], weight, alpha, colors[7], 'Parallel-24', '-', linewidth=1)

    # simple_plot(ax, [data[0]], weight, alpha, colors[0], 'Perfect estimation', '-', linewidth=1)
    # simple_plot(ax, [filename1], weight, alpha, colors[1], 'Estimation error 5mm', '-', linewidth=1)

    # simple_plot(ax, [data[0]], weight, alpha, colors[0], 'Random initial pose', '-', linewidth=1)
    # simple_plot(ax, [data[1]], weight, alpha, colors[1], 'Random goal', '-', linewidth=1)
    # simple_plot(ax, [data[2]], weight, alpha, colors[2], 'Random stiffness', '-', linewidth=1)
    # simple_plot(ax, [data[3]], weight, alpha, colors[3], 'Random uncertainty error', '-', linewidth=1)
    # simple_plot(ax, [data[4]], weight, alpha, colors[4], 'Random All', '-', linewidth=1)

    simple_plot(ax, [data[0]], weight, alpha, colors[0], 'Policy input w/o $F_g$', '-', linewidth=1)
    # simple_plot(ax, [data[1]], weight, alpha, colors[1], 'Policy input w/o $a_{t-1}$', '-', linewidth=1)
    simple_plot(ax, [data[2]], weight, alpha, colors[1], 'Policy input with $F_g$', '-', linewidth=1)
    # simple_plot(ax, [data[3]], weight, alpha, colors[3], 'insertion force input wrt goal', '-', linewidth=1)
    # simple_plot(ax, [data[4]], weight, alpha, colors[4], 'baseline DR', '-', linewidth=1)
    # simple_plot(ax, [data[5]], weight, alpha, colors[5], 'insertion force input DR', '-', linewidth=1)

    # simple_plot(ax, [filename], weight, alpha, colors[0], 'Conical helix p-25', '-', linewidth=1)

    if mode == 2 or mode == 4:
        plt.xlabel("Steps", size='xx-large')
    else:
        plt.xlabel("Episode", size='xx-large')
    plt.ylim([-50, 230])
    plt.ylabel("Reward", size='xx-large')
    plt.legend()
    plt.grid(linestyle='--', linewidth=0.4)
    ax.xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
    plt.show()
    
plot_ft()