from plotter_utils import smooth, reformat_large_tick_values
import matplotlib.ticker as tick
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')


filename = '/home/cambel/dev/results/peg20k-easy_SAC_peg_ft_tcn_p14/state_20200623T113357.npy'
filename = '/home/cambel/dev/results/SAC_randerror_p14/state_20200701T115034.npy'
filename = '/home/cambel/dev/results/SAC_randinit_p14/state_20200701T120225.npy'
filename = '/home/cambel/dev/results/SAC_randerror_p14/hard_state_20200701T115717.npy'
filename = '/home/cambel/dev/results/SAC_randerror_x_step_p14/state_20200701T134301.npy'
filename = '/media/cambel/Extra/research/MDPI/simulation/tcn/SAC_randerror_x_step_p14/elec2-state_20200701T155437.npy'
filename = '/media/cambel/Extra/research/MDPI/real/tests/randinit25-scratch_SAC_peg_ft_tcn_p14/elec2-state_20200701T161028.npy'

filename = '/media/cambel/Extra/research/MDPI/simulation/tcn/SAC_randerror_x_step_p14/peg_hard_state_20200701T145842.npy'
filename = '/media/cambel/Extra/research/MDPI/simulation/tcn/SAC_randinit_p14/easy-state_20200701T150813.npy'
filename = '/media/cambel/Extra/research/MDPI/simulation/tcn/SAC_randinit_p14/elec1-state_20200701T153743.npy'
filename = '/media/cambel/Extra/research/MDPI/simulation/tcn/SAC_randstiff_p14/easy-state_20200701T150946.npy'
filename = '/media/cambel/Extra/research/MDPI/real/tests/randinit25-scratch_SAC_peg_ft_tcn_p14/elec1-state_20200701T162608.npy'
filename = '/media/cambel/Extra/research/MDPI/simulation/tcn/SAC_randerror_x_step_p14/elec1-state_20200701T161716.npy'

weight = 0.0
episode = 1 #45
insertion_dir = 0

max_dist = None

def extract_obs(obs):
    dist = obs[0][:3].astype(np.float) # Distance [0:6]
    print(len(obs[0]))
    if len(obs[0]) == 32:
        force = obs[0][12:15].astype(np.float) # Contact Force [6:12]
        xact = obs[0][18:21].astype(np.float) # 6
        pdact = obs[0][24:26].astype(np.float) # 2
        alpha = obs[0][26:29].astype(np.float) # 6
    else:
        xact = obs[0][12:15].astype(np.float) 
        pdact = obs[0][18:20].astype(np.float)
        alpha = obs[0][20:23].astype(np.float)
        force = obs[0][-7:-1].astype(np.float)
    return dist, force, xact, pdact, alpha


def process_data(episode_data):
    rforce = []
    rdist = []
    rxaction = []
    rpdaction = []
    ralphaion = []
    max_dist = None
    for ep in episode_data:
        dist, force, xact, pdact, alpha = extract_obs(ep)
        if max_dist is None:
            max_dist = dist
        rforce.append(force)
        rxaction.append(xact)
        rpdaction.append(pdact)
        ralphaion.append(alpha)
        dist[insertion_dir] += 8
        rdist.append(dist/max_dist)
    rforce = np.array(rforce).reshape((-1,6))
    print(rforce.shape)

    return np.array(rdist), np.array(rforce), np.array(rxaction), np.array(rpdaction), np.array(ralphaion)


def process(filename, index=-1):
    data = np.load(filename, allow_pickle=True)

    dist, force, d_act, pd_act, alpha = process_data(np.array(data[index]))
    x = np.linspace(1, len(dist), len(dist))

    fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(7.5, 7.5))

    f_lab = []
    if alpha.shape[1] == 1:
        f_lab = ['$k_d$']
    else:
        f_lab = ['$K_f^p$', '$S$']
    labels = ['$x$', '$a_x$']
    plot_cross_data(0, x, [dist[:,insertion_dir], d_act[:,insertion_dir]], ax, labels, ylim=[-1,1],ls = ['-', '-', '--', ':'])
    
    alpha *= -1
    labels = [ '$K_p^x$','$K_f^x$']
    plot_cross_data(1, x, [pd_act[:,0], pd_act[:,1]], ax, labels, colors = ['C3','C6','black'], ls = ['-', '-', '-', ':'])
    ax[0].set_xticklabels([])
    ax[1].set_xticklabels([])
    ax[2].set_xticklabels([])
    ax[0].set_xlim([0,len(x)*1.2])
    ax[1].set_xlim([0,len(x)*1.2])
    ax[2].set_xlim([0,len(x)*1.2])
    ax[3].set_xlim([0,len(x)*1.2])

    labels = [ '$S$']
    plot_cross_data(2, x, [alpha[:,insertion_dir]], ax, labels, colors = ['C4'], ls = ['-', '-', '-', ':'])

    labels = [ '$f_{ext}$']
    force *= np.array([40.,40,40,2,2,2])
    x = np.linspace(1, len(force), len(force))
    plot_cross_data(3, x, [force[:,insertion_dir]], ax, labels, colors = ['C5'], ls = ['-', '-', '-', ':'], ylim=[-5, 40])

def plot_cross_data(i, x, ys, ax, ys_labels, colors = ['C0','C1','gray','black'], ls = ['-', '-', '-', ':'], ylim=[-1,1]):
    lw = [1, 1, 1, 1]
    for y, yl, _ls, _lw, cl in zip(ys, ys_labels, ls, lw, colors):
        ax[i].plot(x, smooth(y, 0.0), _ls, label=yl, linewidth=_lw, color=cl)
    ax[i].legend(loc='lower right', ncol=1, prop={'size': 20}, frameon=False, bbox_to_anchor=(1.03,-.08))
    ax[i].tick_params(axis="y", labelsize=15)
    ax[i].tick_params(axis="x", labelsize=15)
    ax[i].set_ylim(ylim)
    box = ax[i].get_position()
    ax[i].set_position([box.x0-0.06, box.y0 + box.height * 0., box.width * 1.201, box.height * 1.05])


process(filename, episode)
plt.show()
