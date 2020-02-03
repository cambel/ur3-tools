import csv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=np.inf)

colors = iter(['#550000', '#D46A6A', '#004400', '#55AA55', '#061539', '#4F628E'])

data1 = "/home/cambel/dev/data/run-20200116T012641_SAC_impedance-tag-Common_training_return.csv"
data2 = "/home/cambel/dev/data/run-20200116T014501_SAC_impedance-tag-Common_training_return.csv"
data3 = "/home/cambel/dev/data/run-20200116T015921_SAC_impedance-tag-Common_training_return.csv"
data4 = "/home/cambel/dev/data/run-20200116T024611_SAC_hybrid-tag-Common_training_return.csv"
data5 = "/home/cambel/dev/data/run-20200116T031728_SAC_hybrid-tag-Common_training_return.csv"
data6 = "/home/cambel/dev/data/run-20200116T040934_SAC_hybrid-tag-Common_training_return.csv"
data7 = "/home/cambel/dev/data/run-20200116T044040_SAC_hybrid-tag-Common_training_return.csv"
data8 = "/home/cambel/dev/data/run-20200116T052535_SAC_hybrid-tag-Common_training_return.csv"
data9 = "/home/cambel/dev/data/run-20200116T054639_SAC_hybrid-tag-Common_training_return.csv"
data10 = "/home/cambel/dev/data/run-20200116T062220_SAC_impedance-tag-Common_training_return.csv"
data11 = "/home/cambel/dev/data/run-20200116T064553_SAC_impedance-tag-Common_training_return.csv"
data12 = "/home/cambel/dev/data/run-20200116T072858_SAC_impedance-tag-Common_training_return.csv"
data13 = "/home/cambel/dev/data/run-20200116T074812_SAC_impedance-tag-Common_training_return.csv"

data14 = "/home/cambel/dev/data/run-20200116T065341.837268_SAC_hybrid_all-tag-Common_training_return.csv"
data15 = "/home/cambel/dev/data/run-20200116T072933.569657_SAC_hybrid_all-tag-Common_training_return.csv"
data16 = "/home/cambel/dev/data/run-20200116T074735.078587_SAC_hybrid_all-tag-Common_training_return.csv"
data17 = "/home/cambel/dev/data/run-20200116T080445.001318_SAC_hybrid_all-tag-Common_training_return.csv"

data18 = "/home/cambel/dev/data/run-20200116T084027.809724_SAC_hybrid_sim_position-tag-Common_training_return.csv"
data19 = "/home/cambel/dev/data/run-20200116T085540.998029_SAC_hybrid_sim_position-tag-Common_training_return.csv"
data20 = "/home/cambel/dev/data/run-20200116T090620.437910_SAC_hybrid_sim_position-tag-Common_training_return.csv"

data21 = "/media/cambel/Extra/research/data/training_force/gps/hybrid_9dof/01/data_files/default/costs.npy"
data22 = "/media/cambel/Extra/research/data/training_force/gps/hybrid_9dof/02/data_files/default/costs.npy"
data23 = "/media/cambel/Extra/research/data/training_force/gps/hybrid_9dof/03/data_files/default/costs.npy"

data24 = "/media/cambel/Extra/research/data/training_force/gps/hybrid_14dof/01/data_files/default/costs.npy"
data25 = "/media/cambel/Extra/research/data/training_force/gps/hybrid_14dof/02/data_files/default/costs.npy"
data26 = "/media/cambel/Extra/research/data/training_force/gps/hybrid_14dof/03/data_files/default/costs.npy"

data27 = "/media/cambel/Extra/research/data/training_force/gps/impedance_8dof/01/data_files/default/costs.npy"
data28 = "/media/cambel/Extra/research/data/training_force/gps/impedance_8dof/02/data_files/default/costs.npy"
data29 = "/media/cambel/Extra/research/data/training_force/gps/impedance_8dof/03/data_files/default/costs.npy"

data30 = "/media/cambel/Extra/research/data/training_force/gps/impedance_13dof/01/data_files/default/costs.npy"
data31 = "/media/cambel/Extra/research/data/training_force/gps/impedance_13dof/02/data_files/default/costs.npy"
data32 = "/media/cambel/Extra/research/data/training_force/gps/impedance_13dof/03/data_files/default/costs.npy"

data33 = "/media/cambel/Extra/research/data/training_force/sac/impedance_8dof/01.csv"
data34 = "/media/cambel/Extra/research/data/training_force/sac/impedance_8dof/02.csv"
data35 = "/media/cambel/Extra/research/data/training_force/sac/impedance_8dof/03.csv"

data36 = "/media/cambel/Extra/research/data/training_force/sac/hybrid_9dof/01.csv"
data37 = "/media/cambel/Extra/research/data/training_force/sac/hybrid_9dof/02.csv"
data38 = "/media/cambel/Extra/research/data/training_force/sac/hybrid_9dof/03.csv"

data39 = "/media/cambel/Extra/research/data/training_force/sac/hybrid_14dof/01.csv"
data40 = "/media/cambel/Extra/research/data/training_force/sac/hybrid_14dof/02.csv"
data41 = "/media/cambel/Extra/research/data/training_force/sac/hybrid_14dof/03.csv"

data42 = "/media/cambel/Extra/research/data/training_force/sac/impedance_13dof/01.csv"
data43 = "/media/cambel/Extra/research/data/training_force/sac/impedance_13dof/02.csv"
data44 = "/media/cambel/Extra/research/data/training_force/sac/impedance_13dof/03.csv"

data = [data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13, data14, data15, data16, data17,
data18, data19, data20, data21, data22, data23, data24, data25, data26, data27, data28, data29, data30, data31, data32, data33,
data34, data35, data36, data37, data38, data39, data40, data41, data42, data43, data44]

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
    weight = 0.6
    alpha = 0.05

    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0+0.02, box.y0 + box.height * 0.3, box.width, box.height * 0.8])

    # x,y,y_std = prepare_data([9,10,11,12], weight)
    # ax.plot(x, y, '-', color='navy', label='Impedance prioritized experience replay')
    # ax.fill_between(x, y-y_std, y+y_std, antialiased=True, linewidth=1, alpha=alpha, edgecolor='darkblue', facecolor='blue')

    # x,y,y_std = prepare_data([0,1], weight)
    # ax.plot(x, y, '-', color='c', label='Impedance')
    # ax.fill_between(x, y-y_std, y+y_std, antialiased=True, linewidth=1, alpha=alpha, edgecolor='c', facecolor='cyan')

    # x,y,y_std = prepare_data([4,5], weight)
    # ax.plot(x, y, '-', color='g', label='Hybrid Fix Alpha (0.8)')
    # ax.fill_between(x, y-y_std, y+y_std, antialiased=True, linewidth=1, alpha=alpha, edgecolor='darkgreen', facecolor='green')

    # x,y,y_std = prepare_data([7,8], weight)
    # ax.plot(x, y, '-', color='r', label='Hybrid Fix Alpha (0.5)')
    # ax.fill_between(x, y-y_std, y+y_std, antialiased=True, linewidth=1, alpha=alpha, edgecolor='darkred', facecolor='red')

    # x,y,y_std = prepare_data([13,14,15,16], weight)
    # ax.plot(x, y, '-', color='purple', label='Hybrid variable Alpha')
    # ax.fill_between(x, y-y_std, y+y_std, antialiased=True, linewidth=1, alpha=alpha, edgecolor='purple', facecolor='purple')

    # x,y,y_std = prepare_data([17,18,19], weight)
    # ax.plot(x, y, '-', color='grey', label='Position only')
    # ax.fill_between(x, y-y_std, y+y_std, antialiased=True, linewidth=1, alpha=alpha, edgecolor='grey', facecolor='grey')

    ### GPS ###
    x,y,y_std = prepare_data([20,21,22], weight)
    ax.plot(x, y, '-', color='purple', label='GPS-Hybrid position/force (9)')
    ax.fill_between(x, y-y_std, y+y_std, antialiased=True, linewidth=1, alpha=alpha, edgecolor='purple', facecolor='purple')

    x,y,y_std = prepare_data([23,24,25], weight)
    ax.plot(x, y, '-', color='red', label='GPS-Hybrid position/force (14)')
    ax.fill_between(x, y-y_std, y+y_std, antialiased=True, linewidth=1, alpha=alpha, edgecolor='red', facecolor='red')

    # x,y,y_std = prepare_data([26,27,28], weight)
    # ax.plot(x, y, '-', color='blue', label='GPS-Impedance (8)')
    # ax.fill_between(x, y-y_std, y+y_std, antialiased=True, linewidth=1, alpha=alpha, edgecolor='blue', facecolor='blue')

    # x,y,y_std = prepare_data([29,30,31], weight)
    # ax.plot(x, y, '-', color='g', label='GPS-Impedance (13)')
    # ax.fill_between(x, y-y_std, y+y_std, antialiased=True, linewidth=1, alpha=alpha, edgecolor='darkgreen', facecolor='green')

    # x,y,y_std = prepare_data([32,33,34], weight)
    # ax.plot(x, y, '--', color='c', label='SAC-Impedance (8)')
    # ax.fill_between(x, y-y_std, y+y_std, antialiased=True, linewidth=1, alpha=alpha, edgecolor='c', facecolor='cyan')

    x,y,y_std = prepare_data([35,36,37], weight)
    ax.plot(x, y, '--', color='navy', label='SAC-Hybrid position/force (9)')
    ax.fill_between(x, y-y_std, y+y_std, antialiased=True, linewidth=1, alpha=alpha, edgecolor='darkblue', facecolor='blue')

    x,y,y_std = prepare_data([38,39,40], weight)
    ax.plot(x, y, '--', color='y', label='SAC-Hybrid position/force (14)')
    ax.fill_between(x, y-y_std, y+y_std, antialiased=True, linewidth=1, alpha=alpha, edgecolor='y', facecolor='y')

    # x,y,y_std = prepare_data([41,42,43], weight)
    # ax.plot(x, y, '--', color='black', label='SAC-impedance (13)')
    # ax.fill_between(x, y-y_std, y+y_std, antialiased=True, linewidth=1, alpha=alpha, edgecolor='black', facecolor='black')

    # ax.plot(d3[0], d3[1], 'k', color='darkblue', label='Impedance')
    # ax.plot(d4[0], d4[1], 'k', color='darkgreen', label='Hybrid')
    # ax.plot(xi, adm_i[v], '--', color='Green', label='impedance')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
          fancybox=True, shadow=True, ncol=2)
    # plt.xlim((0,t))
    # plt.yticks(np.arange(-1800, 100, 300.0))
    # plt.xticks(np.arange(0, t, 1.0))
    plt.grid(linestyle='--', linewidth=0.5)
    plt.xlabel("Steps", size='x-large')
    plt.ylabel("Reward", size='x-large')
    # plt.ylabel("Distance (m)", size='x-large')

    plt.show()

plot_ft()

# cost = np.load("/home/cambel/dev/container_gps/experiments/ur3/mdgps/hybrid/full/data_files/default/costs.npy")
# cost = np.sum(cost, axis=3)
# cost = cost.reshape(-1)
# print(cost.shape, cost)