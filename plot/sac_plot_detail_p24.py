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
    xact = obs[0][12:18].astype(np.float) 
    pdxact = obs[0][18:24].astype(np.float) # PD pos
    pdfact = obs[0][24:30].astype(np.float) # PD force
    alpha = obs[0][30:36].astype(np.float)
    force = obs[0][36:-1].astype(np.float) + ft_fix
    return dist, force, xact, pdxact, pdfact, alpha


def process_data(episode_data):
    rforce = []
    rdist = []
    rxaction = []
    rpdxaction = []
    rpdfaction = []
    ralphaion = []
    max_dist = None
    for ep in episode_data:
        dist, force, xact, pdxact, pdfact, alpha = extract_obs(ep)
        # if max_dist is None:
        #     max_dist = np.abs(dist)
        #     max_dist[:3] = np.array([40,40,40,])
        rforce.append(force)
        rxaction.append(xact)
        rpdxaction.append(pdxact)
        rpdfaction.append(pdfact)
        ralphaion.append(alpha)
        dist[insertion_dir] += 0
        rdist.append(dist)
    rforce = np.array(rforce).reshape((-1,6))
    print(rforce.shape)

    return np.array(rdist), np.array(rforce), np.array(rxaction), np.array(rpdxaction), np.array(rpdfaction), np.array(ralphaion)


def process(filename, index=-1):
    data = np.load(filename, allow_pickle=True)

    dist, force, d_act, pdx_act, pdf_act, alpha = process_data(np.array(data[index]))
    x = np.linspace(1, len(dist), len(dist))

    _, ax = plt.subplots(nrows=4, ncols=1, figsize=(7.5, 7.5))

    x_labels = ['$x$','$y$','$z$',r'$\alpha$', r'$\beta$',r'$\gamma$']
    labels = [x_labels[insertion_dir], '$a_x$']
    plot_cross_data(0, x, [dist[:,insertion_dir]], ax, labels, ylim=[-25,25],ls = ['-', '-', '--', ':'])
    ax[0].axhline(y=0.0, color='gray', linestyle='--')
    ax[1].axhline(y=0.0, color='gray', linestyle='--')

    alpha *= -1
    labels = [ '$K_p^x$','$K_f^x$']
    plot_cross_data(3, x, [pdx_act[:,insertion_dir], pdf_act[:,insertion_dir]], ax, labels, colors = ['C3','C6','black'], ls = ['-', '-', '-', ':'])

    labels = [ '$S$', '$a_x$']
    plot_cross_data(2, x, [alpha[:,insertion_dir], d_act[:,insertion_dir]], ax, labels, colors = ['C4', 'C1'], ls = ['-', '-', '-', ':'])

    labels = [ '$f_{ext}$']
    force *= np.array([30.,30,30,1,1,1])
    fx = np.linspace(1, len(force), len(force))
    force_limits = [-10,30] if insertion_dir < 3 else [-1,1]
    plot_cross_data(1, fx, [force[:,insertion_dir]], ax, labels, colors = ['C5'], ls = ['-', '-', '-', ':'], ylim=force_limits)

    ax[0].set_xticklabels([])
    ax[1].set_xticklabels([])
    ax[2].set_xticklabels([])
    ax[0].set_xlim([0,len(x)*1.2])
    ax[1].set_xlim([0,len(fx)*1.2])
    ax[2].set_xlim([0,len(x)*1.2])
    ax[3].set_xlim([0,len(x)*1.2])

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

filename = '/home/cambel/dev/results/good200k_SAC_conical_p24/state_20200722T184452.npy'
filename = '/media/cambel/Extra/research/MDPI/real/square_peg/50k_SAC_real_square_peg/09_rot_state_20200726T170030.npy'
filename = '/home/cambel/dev/results/350k_SAC_randgoal_p24/state_20200727T113458.npy'
filename = '/home/cambel/dev/results/retrain_SAC_real_square_peg/state_20200727T153347.npy'

filename = '/home/cambel/dev/results/real_no_table_428_state_20200804T115215.npy'
filename = '/home/cambel/dev/results/sim_no_table_68_state_20200804T120226.npy'
filename = '/home/cambel/dev/results/real_table_2000k_state_20200804T114922.npy'
filename = '/home/cambel/dev/results/sim_table_99_state_20200804T120558.npy'
filename = '/home/cambel/dev/results/real_table_140_state_20200804T123127.npy'
filename = '/home/cambel/dev/results/real_table_x4_state_20200804T123551.npy'
filename = '/home/cambel/dev/results/retrain_sim_SAC_real_square_peg/state_20200804T140533.npy'
filename = '/home/cambel/dev/results/350k_SAC_randgoal_p24/sim2real_state_20200804T141015.npy'

weight = 0.0
episode = 0
insertion_dir = 1
ft_fix = 0.0

max_dist = None
# for i in range(6):
#     ft_fix = 0.1 if i == 1 else 0.0
#     insertion_dir = i
#     process(filename, episode)
#     plt.savefig('/home/cambel/dev/data/sim_peg2_'+str(i)+'.png')
#     plt.clf()

process(filename, episode)
plt.show()