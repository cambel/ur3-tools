from plotter_utils import smooth, reformat_large_tick_values
import matplotlib.ticker as tick
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')


filename = '/media/cambel/Extra/research/IROS20_revised/Real/ring/good/state_20200513T120853hybrid14.npy'
filename = '/media/cambel/Extra/research/IROS20_revised/Real/ring/good/state_20200513T140712impedance13.npy'
filename = "/media/cambel/Extra/research/IROS20_revised/Real/wrs/01/state_20200512T120140hybrid14.npy"
filename = "/media/cambel/Extra/research/IROS20_revised/Real/wrs/01/state_20200513T163611impedance13.npy"

weight = 0.0
episodes = 0#ri191 #h-144 a-127

max_dist = None
cost_ws = [1., 1., 1.]


def extract_obs(obs):
    dist = obs[0][:3].astype(np.float)
    force = obs[0][12:15].astype(np.float)
    # actions = obs[0][18:-1].astype(np.float)
    xact = obs[0][18:21].astype(np.float)
    pdact = obs[0][24:27].astype(np.float)
    fact = obs[0][30:-1].astype(np.float)
    return dist, force, xact, pdact, fact


def process_data(episode_data):
    rforce = []
    rdist = []
    rxaction = []
    rpdaction = []
    rfaction = []
    max_dist = None
    for ep in episode_data:
        dist, force, xact, pdact, fact = extract_obs(ep)
        if max_dist is None:
            max_dist = dist
        rforce.append(force)
        rxaction.append(xact)
        rpdaction.append(pdact)
        rfaction.append(fact)
        rdist.append(dist/max_dist)

    return np.array(rdist), np.array(rforce), np.array(rxaction), np.array(rpdaction), np.array(rfaction)


def process(filename, index=-1):
    data = np.load(filename, allow_pickle=True)

    dist, force, d_act, pd_act, f_act = process_data(np.array(data[index]))
    x = np.linspace(1, len(force), len(force))

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 5))

    f_lab = []
    if f_act.shape[1] == 1:
        f_lab = ['$k_d$']
    else:
        f_lab = ['$K_f^p$', '$S$']
    labels = [['$f_{ext}$', '$f_{ext}$', '$f_{ext}$'], ['$x$', '$x$', '$x$'], ['$a_x$', '$a_x$', '$a_x$'], ['$K_p^x$', '$a_p^x$', '$a_p^x$']]
    labels = [['$f_{ext}$', '$f_{ext}$', '$f_{ext}$'], ['$x$', '$x$', '$x$'], ['$a_x$', '$a_x$', '$a_x$'], ['$K_p^x$', '$a_p^x$', '$a_p^x$']]
    plot_cross_data(x, [force, dist, d_act, pd_act], ax, labels)
    f_act *= -1
    obj = ax[1].plot(x, smooth(f_act, weight), label='imp', linewidth=1.0)
    # ax[1].legend(iter(obj), tuple(f_lab), loc='lower right', ncol=1, prop={'size': 20}, bbox_to_anchor=(1.01,-.05))  # ,'ax','ay','az'))
    box = ax[1].get_position()
    ax[1].set_position([box.x0-0.06, box.y0 + box.height * 0.005, box.width * 1.201, box.height * 1.13])

def plot_cross_data(x, ys, ax, ys_labels, ylabel=['x axis', 'y axis', 'z axis']):
    ls = ['-', '-', '--', ':']
    lw = [1.5, 1.5, 1, 1]
    colors = ['C2','C3','C4','black']
    for i in range(1):
        for y, yl, _ls, _lw, cl in zip(ys, ys_labels, ls, lw, colors):
            ax[i].plot(x, smooth(y[:, i], 0.0), _ls, label=yl[i], linewidth=_lw, color=cl)
            # obj = ax[i].plot(x, y2[:,i], label=y2_labels[i], linewidth=1.0)
        # ax[i].legend(loc='lower right', ncol=1, prop={'size': 20}, frameon=True, bbox_to_anchor=(1.01,-.05))
        ax[i].set_xticklabels([])
        ax[i].tick_params(axis="y", labelsize=14)
        ax[i].set_xlim([0,200])
        ax[i].set_ylim([-1.1,1])
        box = ax[i].get_position()
        ax[i].set_position([box.x0-0.06, box.y0 + box.height * 0.005, box.width * 1.201, box.height * 1.13])
        # plt.setp(ax[i], ylabel=ylabel[i])


process(filename, episodes)

# plt.ylabel('Reward')
plt.ylim([-1,1])
plt.xlim([0,200])
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
# plt.xlabel('Step', size='xx-large')
# plt.legend()
# box = ax.get_position()
# ax.set_position([box.x0-0.02, box.y0 + box.height * 0.0, box.width * 1.13, box.height * 1.13]) # label in
plt.show()
