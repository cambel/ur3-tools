import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as tick

from plotter_utils import smooth, reformat_large_tick_values

def load_file(filename):
    data = np.load(filename)
    return data


base_filenames = ["/media/cambel/Extra/research/IROS2020/real/peg/02/state_20200221T104534hybrid_position_force14.npy",
"/media/cambel/Extra/research/IROS2020/real/peg/02/state_20200221T104426impedance13.npy",
"/media/cambel/Extra/research/IROS2020/real/wood_toy/02/state_20200221T100131impedance13.npy",
"/media/cambel/Extra/research/IROS2020/real/wood_toy/02/state_20200221T095404hybrid_position_force14.npy",]

final_filenames = ["/media/cambel/Extra/research/IROS2020/real/peg/02/state_20200221T105331hybrid_position_force14.npy",
"/media/cambel/Extra/research/IROS2020/real/peg/02/state_20200221T110910impedance13.npy",
"/media/cambel/Extra/research/IROS2020/real/wood_toy/02/state_20200221T100131impedance13.npy",
'/media/cambel/Extra/research/IROS2020/real/wood_toy/02/state_20200221T094035hybrid_position_force14.npy']

labels = ['SAC H-14 Final Policy', 'SAC I-13pd Final Policy', 'SAC I-13pd Final Policy', 'SAC H-14 Final Policy']

extra = [None, None, [3637,3837], None]
remove = [
    [4889,-116], # peg hybrid
    [4694,-333], # peg imp
    [4656,4750], # wood imp
    [4523,-523], # wood hybrid
]
weight = 0.0

def simple_plot(ptype, indice):
    base = load_file(base_filenames[indice])
    base = base.reshape((base.shape[0], base.shape[2]))

    final = load_file(final_filenames[indice])
    final = final.reshape((final.shape[0], final.shape[2]))
    c = []
    l = ''
    if ptype == 0: #distance
        dim = 1
        plt.ylim(1,25)
        c = ['C1', 'C0']
        l = 'dist'
    elif ptype == 1: #force
        dim = 7
        plt.ylim(-1,30)
        c = ['C2', 'C3']
        l = 'force'
    else:
        print ("WDF?")
        return

    if extra[indice] is not None:
        base = base[extra[indice][0]:extra[indice][1], :]
    x1 = np.linspace(1, base.shape[0], base.shape[0])
    plt.plot(x1, smooth(base[:,dim],weight), ':',c=c[0], label='Fixed base params', linewidth=2)
    
    final = final[remove[indice][0]:remove[indice][1],:]
    x2 = np.linspace(1, final.shape[0], final.shape[0])
    plt.plot(x2, smooth(final[:,dim],weight), c=c[1],label=labels[indice], linewidth=1.5)

    np.savetxt('base-'+labels[indice]+ l+'.csv', np.array([x1, base[:,dim]]).T, delimiter=',', fmt='%f')
    np.savetxt('final-'+labels[indice]+l+'.csv', np.array([x2, final[:,dim]]).T, delimiter=',', fmt='%f')


ax = plt.subplot(111)
box = ax.get_position()
ax.set_position([box.x0-0.07, box.y0-0.05, box.width * 1.215, box.height * 1.2]) # label in  
simple_plot(1, 1)


plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plt.ylabel('Force [N]')
# # plt.ylabel('Distance error [mm]')
# plt.xlabel('Step')
plt.legend(prop={'size': 15})
ax.xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
plt.show()
