from plotter_utils import smooth, reformat_large_tick_values
import matplotlib.ticker as tick
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')


filename = '/home/cambel/dev/results/randinit-hard-fast3_SAC_wave-parallel-14/state_20200623T111712.npy'

weight = 0.0
episode = 0
insertion_dir = 1

max_dist = None

def extract_obs(obs):
    dist = obs[0][:3].astype(np.float) # Distance [0:6]
    if len(obs[0]) == 32:
        force = obs[0][12:15].astype(np.float) # Contact Force [6:12]
        xact = obs[0][18:21].astype(np.float) # 6
        pdact = obs[0][24:26].astype(np.float) # 2
        alpha = obs[0][26:29].astype(np.float) # 6
    else:
        xact = obs[0][12:15].astype(np.float) 
        pdact = obs[0][18:20].astype(np.float)
        alpha = obs[0][20:23].astype(np.float)
        force = obs[0][-6:]
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
        rdist.append(dist/max_dist)

    return np.array(rdist), np.array(rforce), np.array(rxaction), np.array(rpdaction), np.array(ralphaion)


def process(filename, index=-1):
    data = np.load(filename, allow_pickle=True)

    dist, force, d_act, pd_act, alpha = process_data(np.array(data[index]))
    x = np.linspace(1, len(force), len(force))

    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(7.5, 5.5))

    f_lab = []
    if alpha.shape[1] == 1:
        f_lab = ['$k_d$']
    else:
        f_lab = ['$K_f^p$', '$S$']
    labels = [['$f_{ext}$', '$f_{ext}$', '$f_{ext}$'], ['$x$', '$x$', '$x$'], ['$a_x$', '$a_x$', '$a_x$']]
    plot_cross_data(0, x, [force[:,insertion_dir], dist[:,insertion_dir], d_act[:,insertion_dir]], ax, labels, ylim=[-1,1],ls = ['-', '-', '--', ':'])
    
    alpha *= -1
    labels = [ ['$K_p^x$']]
    plot_cross_data(1, x, [pd_act[:,insertion_dir]], ax, labels, colors = ['C3','C3','black'], ls = ['-', '-', '-', ':'])
    ax[0].set_xticklabels([])
    ax[1].set_xticklabels([])

    labels = [ ['$S$']]
    plot_cross_data(2, x, [alpha[:,insertion_dir]], ax, labels, colors = ['C4','C8','C3','black'], ls = ['-', '-', '-', ':'])


def plot_cross_data(i, x, ys, ax, ys_labels, ylabel=['x axis', 'y axis', 'z axis'], colors = ['C0','C1','gray','black'], ls = ['-', '-', '-', ':'], ylim=[-1,1]):
    lw = [1, 1, 1, 1]
    for y, yl, _ls, _lw, cl in zip(ys, ys_labels, ls, lw, colors):
        ax[i].plot(x, smooth(y, 0.0), _ls, label=yl[0], linewidth=_lw, color=cl)
    ax[i].legend(loc='lower right', ncol=1, prop={'size': 20}, frameon=False, bbox_to_anchor=(1.03,-.08))
    ax[i].tick_params(axis="y", labelsize=15)
    ax[i].tick_params(axis="x", labelsize=15)
    ax[i].set_xlim([0,200])
    ax[i].set_ylim(ylim)
    box = ax[i].get_position()
    ax[i].set_position([box.x0-0.06, box.y0 + box.height * 0., box.width * 1.201, box.height * 1.05])


process(filename, episode)
plt.show()
