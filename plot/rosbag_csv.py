import csv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

colors = iter(['#550000', '#D46A6A', '#004400', '#55AA55', '#061539', '#4F628E'])

adm_disc = "/home/cambel/dev/test_tools/admitance_discrete_2019-12-17-07-08-44-ur3-ee_ft.csv"
adm_integ = "/home/cambel/dev/test_tools/admitance_integration_2019-12-17-07-10-03-ur3-ee_ft.csv"
impedance = "/home/cambel/dev/test_tools/impedance_2019-12-17-07-14-20-ur3-ee_ft.csv"
impedance = "/home/cambel/dev/test_tools/a_2019-12-17-07-55-04-ur3-ee_ft.csv"

def csv_to_list(filename):
    with open(filename, 'r') as f:
        csv_data = list(csv.reader(f, delimiter=","))
    l = np.array(csv_data[:])
    print(l.shape)
    return l.astype(float)

def plot_ft():
    adm_a = csv_to_list(adm_disc)
    # adm_a = adm_a[2600:5900]
    # adm_a = adm_a[8125:11510]
    adm_b = csv_to_list(adm_integ)
    # adm_b = adm_b[3250:6700]
    imp = csv_to_list(impedance)
    imp = imp[19000:29150]
    # imp = imp[25500:29150]
    # p = p[3600:30000,:]
    
    t = 10 #sec

    xa = np.linspace(0, t, num=adm_a.shape[0])
    xb = np.linspace(0, t, num=adm_b.shape[0])
    xi = np.linspace(0, t, num=imp.shape[0])
    ax = plt.subplot(111)

    ax.plot(xa, adm_a[:], '--', color='Blue', label='admittance discrete')
    ax.plot(xb, adm_b[:], 'k', color='Orange', label='admittance integration')
    ax.plot(xi, imp[:], '--', color='Green', label='impedance')
    ax.legend()
    plt.xlim((0,t))
    plt.yticks(np.arange(-5, 50, 5.0))
    plt.xticks(np.arange(0, t, 5.0))
    plt.grid(linestyle='--', linewidth=0.5)
    plt.xlabel("Time (s)", size='x-large')
    plt.ylabel("Force (N)", size='x-large')

    plt.show()

plot_ft()