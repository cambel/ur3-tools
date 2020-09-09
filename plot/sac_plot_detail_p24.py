from plotter_utils import smooth, reformat_large_tick_values
import matplotlib.ticker as tick
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')


def extract_obs(obs):
    # 6 + 6 + 24 + 6*n + 1
    n_extra = 0
    dist = obs[0][:6].astype(np.float) # Distance [0:6]
    xact = obs[0][n_extra+12:n_extra+18].astype(np.float) #*-1.0
    pdxact = obs[0][n_extra+18:n_extra+24].astype(np.float) # PD pos
    pdfact = obs[0][n_extra+24:n_extra+30].astype(np.float) # PD force
    alpha = obs[0][n_extra+30:n_extra+36].astype(np.float) #*-1.0
    force = obs[0][n_extra+12+n_actions:n_extra+18+n_actions].astype(np.float)
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

    figure, ax = plt.subplots(nrows=4, ncols=1, figsize=(7.5, 6))
    figure.tight_layout(h_pad=-1.0)

    x_labels = ['$x$','$y$','$z$',r'$\alpha$', r'$\beta$',r'$\gamma$']
    labels = [x_labels[insertion_dir], '$a_x$']
    dist_limits = [-30,30] if insertion_dir < 3 else [-2,2]
    plot_cross_data(0, x, [dist[:,insertion_dir]], ax, labels, ylim=dist_limits,ls = ['-', '-', '--', ':'])
    ax[0].axhline(y=0.0, color='gray', linestyle='--')
    ax[1].axhline(y=0.0, color='gray', linestyle='--')

    alpha *= -1
    labels = [ '$K_p^x$','$K_f^x$']
    plot_cross_data(3, x, [pdx_act[:,insertion_dir], pdf_act[:,insertion_dir]], ax, labels, colors = ['C3','C6','black'], ls = ['-', '-', '-', ':'])

    labels = [ '$S$', '$a_x$', 'extra']
    if n_actions == 25:
        plot_cross_data(2, x, [alpha[:,insertion_dir], d_act[:,insertion_dir], extra], ax, labels, colors = ['C4', 'C1','C2'], ls = ['-', '-', '-', ':'])
    else:
        plot_cross_data(2, x, [alpha[:,insertion_dir], d_act[:,insertion_dir]], ax, labels, colors = ['C4', 'C1'], ls = ['-', '-', '-', ':'])

    labels = [ '$f_{ext}$']
    force *= np.array([30.,30,30,1,1,1])
    fx = np.linspace(1, len(force), len(force))
    force_limits = [-20,20] if insertion_dir < 3 else [-1,1]
    plot_cross_data(1, fx, [force[:,insertion_dir]], ax, labels, colors = ['C5'], ls = ['-', '-', '-', ':'], ylim=force_limits)

    ax[0].set_xticklabels([])
    ax[1].set_xticklabels([])
    ax[2].set_xticklabels([])
    ax[0].set_xlim([0,len(x)*1.2])
    ax[1].set_xlim([0,len(fx)*1.2])
    ax[2].set_xlim([0,len(x)*1.2])
    ax[3].set_xlim([0,len(x)*1.2])
    ax[1].set_ylim([-5,40])
    ax[2].set_ylim([-1.1,1.1])
    ax[3].set_ylim([-1.1,1.1])

def plot_cross_data(i, x, ys, ax, ys_labels, colors = ['C0','C1','gray','black'], ls = ['-', '-', '-', ':'], ylim=[-1,1]):
    lw = [1, 1, 1, 1]
    for y, yl, _ls, _lw, cl in zip(ys, ys_labels, ls, lw, colors):
        ax[i].plot(x, y, _ls, label=yl, linewidth=_lw, color=cl)
    ax[i].legend(loc='lower right', ncol=1, prop={'size': 20}, frameon=False, bbox_to_anchor=(1.03,-.08))
    ax[i].tick_params(axis="y", labelsize=15)
    ax[i].tick_params(axis="x", labelsize=15)
    ax[i].set_ylim(ylim)
    box = ax[i].get_position()
    ax[i].set_position([box.x0, box.y0 + box.height * 0., box.width, box.height * 1.075])

filename = '/home/cambel/dev/results/350k_SAC_randgoal_p24/state_20200727T101705.npy'
filename = '/media/cambel/Extra/research/MDPI/real/square_peg/50k_SAC_real_square_peg/09_rot_state_20200726T170030.npy'
filename = '/media/cambel/Extra/research/MDPI/real/square_peg/scratch_SAC_real_square_peg/real_state_20200727T144205.npy' #scratch
filename = '/media/cambel/Extra/research/MDPI/simulation/conical/20200719T204941.374932_SAC_conical_p25/state_20200720T122714.npy'
filename = '/home/cambel/dev/results/180k_SAC_conical_randerr_p24/state_20200727T172553.npy'
filename = '/home/cambel/dev/results/300k_SAC_conical_p25/state_20200727T174626.npy'
filename = '/media/cambel/Extra/research/MDPI/simulation/sim2real/350k_SAC_randgoal_p24/real_new_params_state_20200727T160625.npy'

filename = '/media/cambel/Extra/research/MDPI/reality_gap/real_no_table_428_state_20200804T115215.npy'
filename = '/media/cambel/Extra/research/MDPI/reality_gap/sim_no_table_68_state_20200804T120226.npy'
filename = '/media/cambel/Extra/research/MDPI/reality_gap/real_no_table_x6_state_20200804T125435.npy'
filename = '/media/cambel/Extra/research/MDPI/reality_gap/sim_table_99_state_20200804T120558.npy'
filename = '/media/cambel/Extra/research/MDPI/reality_gap/real_table_x6_state_20200804T124832.npy'

filename = '/media/cambel/Extra/research/MDPI/real/square_peg/accuracy_perf_sim2real.npy'
filename = '/media/cambel/Extra/research/MDPI/real/square_peg/accuracy_perf_scratch.npy'

filename = '/media/cambel/Extra/research/MDPI/real/square_peg/retrain_sim_soft.npy'
filename = '/media/cambel/Extra/research/MDPI/real/square_peg/retrain_sim_x6_caucho.npy'
filename = '/media/cambel/Extra/research/MDPI/real/square_peg/accuracy_perf_retrain_sim_x6.npy'

filename = '/media/cambel/Extra/research/MDPI/reality_gap/real_table_80k.npy'
filename = '/media/cambel/Extra/research/MDPI/reality_gap/real_table_100k.npy'
filename = '/media/cambel/Extra/research/MDPI/reality_gap/real_table_500k_x6.npy'

filename = '/home/cambel/dev/results/SAC_Linsertionil_p24/state_20200820T184436.npy'

weight = 0.0
episode = 0
insertion_dir = 1
n_actions = 25  if 'p25' in filename else 24
max_dist = None
mode = 1

if mode == 0:
    for i in range(6):
        ft_fix = 0.1 if i == 1 else 0.0
        insertion_dir = i
        process(filename, episode)
        plt.savefig('/home/cambel/dev/data/retrain_force_input' + '_ep%s_%s.png'%(episode,i))
        plt.clf()
else:
    process(filename, episode)
    plt.show()