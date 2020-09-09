from plotter_utils import smooth, reformat_large_tick_values
import matplotlib.ticker as tick
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')


def extract_obs(obs):
    # 6 + 6 + 24 + 6*n + 1
    dist = obs[0][:6].astype(np.float) # Distance [0:6]
    xact = obs[0][12:18].astype(np.float) *-1.0
    pdxact = obs[0][18:24].astype(np.float) # PD pos
    pdfact = obs[0][24:30].astype(np.float) # PD force
    alpha = obs[0][30:36].astype(np.float) *-1.0
    force = obs[0][12+n_actions:18+n_actions].astype(np.float)
    if n_actions == 25:
        extra = obs[0][-2:-1].astype(np.float)
    else:
        extra = 0
    dist[3:] = np.rad2deg(dist[:3]/1000.0)
    return dist, force, xact, pdxact, pdfact, alpha, extra


def process_data(episode_data):
    rforce = []
    rdist = []
    rxaction = []
    rpdxaction = []
    rpdfaction = []
    ralphaion = []
    rextra = []
    max_dist = None
    ft_fix = None
    for ep in episode_data:
        dist, force, xact, pdxact, pdfact, alpha, extra = extract_obs(ep)
        if ft_fix is None:
            ft_fix = force
        # if max_dist is None:
        #     max_dist = np.abs(dist)
        #     max_dist[:3] = np.array([40,40,40,])
        rforce.append(force-ft_fix)
        rxaction.append(xact)
        rpdxaction.append(pdxact)
        rpdfaction.append(pdfact)
        ralphaion.append(alpha)
        dist[insertion_dir] += 0
        rdist.append(dist)
        rextra.append(extra)
    rforce = np.array(rforce).reshape((-1,6))
    print(rforce.shape)

    return np.array(rdist), np.array(rforce), np.array(rxaction), np.array(rpdxaction), np.array(rpdfaction), np.array(ralphaion), np.array(rextra)


def process(filename, index=-1):
    data = np.load(filename, allow_pickle=True)

    dist, force, d_act, pdx_act, pdf_act, alpha, extra = process_data(np.array(data[index]))
    x = np.linspace(1, len(dist), len(dist))

    figure, ax = plt.subplots(nrows=2, ncols=1, figsize=(7.5, 6))
    figure.tight_layout(h_pad=1.0)

    x_labels = ['$x$','$y$','$z$',r'$\alpha$', r'$\beta$',r'$\gamma$']
    labels = [x_labels[insertion_dir], '$a_x$']
    dist_limits = [-30,30] if insertion_dir < 3 else [-2,2]
    plot_cross_data(0, x, [dist[:,insertion_dir]], ax, labels, ylim=dist_limits,ls = ['-', '-', '--', ':'])
    ax[0].axhline(y=0.0, color='gray', linestyle='--')
    ax[1].axhline(y=0.0, color='gray', linestyle='--')

    labels = [ '$f_{ext}$']
    force *= np.array([30.,30,30,1,1,1])
    fx = np.linspace(1, len(force), len(force))
    force_limits = [-20,20] if insertion_dir < 3 else [-1,1]
    plot_cross_data(1, fx, [force[:,insertion_dir]], ax, labels, colors = ['C5'], ls = ['-', '-', '-', ':'], ylim=force_limits)

    ax[0].set_xticklabels([])
    # ax[1].set_xticklabels([])
    ax[0].set_xlim([0, 105])
    ax[1].set_xlim([0, 105])
    ax[0].set_ylim([-45,10])
    ax[1].set_ylim([-5,25])

def plot_cross_data(i, x, ys, ax, ys_labels, colors = ['C0','C1','gray','black'], ls = ['-', '-', '-', ':'], ylim=[-1,1]):
    lw = [1, 1, 1, 1]
    for y, yl, _ls, _lw, cl in zip(ys, ys_labels, ls, lw, colors):
        ax[i].plot(x, smooth(y, 0.0), _ls, label=yl, linewidth=_lw, color=cl)
    ax[i].legend(loc='lower right', ncol=1, prop={'size': 20}, frameon=False, bbox_to_anchor=(1.03,-.08))
    ax[i].tick_params(axis="y", labelsize=15)
    ax[i].tick_params(axis="x", labelsize=15)
    ax[i].set_ylim(ylim)
    box = ax[i].get_position()
    # ax[i].set_position([box.x0, box.y0 + box.height * 0., box.width, box.height * 1.075])


filename = '/media/cambel/Extra/research/MDPI/real/square_peg/retrain_sim_soft.npy'
filename = '/media/cambel/Extra/research/MDPI/real/square_peg/retrain_sim_x6_caucho.npy'
filename = '/media/cambel/Extra/research/MDPI/real/square_peg/accuracy_perf_retrain_sim_x6.npy'

filename = '/media/cambel/Extra/research/MDPI/reality_gap/real_table_500k_x6.npy'
filename = '/media/cambel/Extra/research/MDPI/reality_gap/real_table_2000k_state_20200804T114922.npy'
filename = '/media/cambel/Extra/research/MDPI/reality_gap/real_table_100k.npy'
filename = '/media/cambel/Extra/research/MDPI/reality_gap/real_table_80k.npy'

filename = '/home/cambel/dev/results/state_20200806T135611.npy'
filename = '/home/cambel/dev/results/state_20200820T183232.npy'

weight = 0.0
episode = 0
insertion_dir = 2
n_actions = 25  if 'p25' in filename else 24
max_dist = None
mode = 1

if mode == 0:
    for i in range(6):
        ft_fix = 0.1 if i == 1 else 0.0
        insertion_dir = i
        process(filename, episode)
        plt.savefig('/home/cambel/dev/data/retrain_sim_medium_stiff_ep%s_%s.png'%(episode,i))
        plt.clf()
else:
    process(filename, episode)
    plt.show()