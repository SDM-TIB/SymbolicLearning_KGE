import pandas as pd
import os
from os.path import isfile, join
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
from math import pi


def get_measure_values(cls_addres, matrix_address, cls_measures, metric):
    static = 16
    n_metric = 5
    onlyfiles = [os.path.join(cls_addres + 'clusters/', f) for f in listdir(cls_addres + 'clusters/') if
                 isfile(join(cls_addres + 'clusters/', f))]
    n_cls = len(onlyfiles)
    measure = []
    index_start = static + n_cls -1
    index_end = index_start + n_metric
    for pos in range(index_start, index_end):
        a = cls_measures.iloc[pos].to_string()
        b = a.split('\\t')[1]
        measure.append(float(b))
    cosine_matrix = pd.read_csv(matrix_address + 'matrix_sim.txt', delimiter=",")
    max_cut = sum(cosine_matrix.sum(axis=0, skipna=True))
    measure[0] = 1.0 - measure[0]
    measure[2] = (measure[2] + 0.5) / 1.5
    measure[3] = 1 - (measure[3] / max_cut)
    alg = [measure[0], measure[4], measure[3], measure[2], measure[1]]
    alg = [round(i, 2) for i in alg]
    metric = [x + y for x, y in zip(metric, alg)]
    return metric


def radar_plot(metric_semep, metric_km, key):
    # Set data
    df = pd.DataFrame(columns=['group', 'InvC', 'P', 'InvTC', 'M', 'Co'])
    # df = pd.DataFrame(columns=['group', 'InvC', 'InvTC', 'Co'])
    # === Baseline
    df.loc[0] = ['SemEP'] + metric_semep
    # df.loc[1] = ['METIS'] + metric_met
    df.loc[1] = ['KMeans'] + metric_km
    # df.loc[0] = ['SemEP'] + [metric_semep[0], metric_semep[2], metric_semep[4]]
    # df.loc[1] = ['METIS'] + [metric_met[0], metric_met[2], metric_met[4]]
    # df.loc[2] = ['KMeans'] + [metric_km[0], metric_km[2], metric_km[4]]

    # ------- PART 1: Create background

    # number of variable
    categories = list(df)[1:]
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories, size=20)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([.1, .2, .3, .4, .5, .6, .7, .8, .9], ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"],
               color="grey", size=15)
    plt.ylim(0, 1)

    # ------- PART 2: Add plots

    # Plot each individual = each line of the data
    # I don't make a loop, because plotting more than 3 groups makes the chart unreadable

    # Ind1
    values = df.loc[0].drop('group').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, color='#264653', linewidth=2, linestyle='solid', label="SemEP")
    ax.fill(angles, values, alpha=0.1, color='#264653')

    # Ind2
    # values = df.loc[1].drop('group').values.flatten().tolist()
    # values += values[:1]
    # ax.plot(angles, values, color='#2A9D8F', linewidth=2, linestyle='solid', label="METIS")
    # ax.fill(angles, values, alpha=0.1, color='#2A9D8F')

    # Ind3
    values = df.loc[1].drop('group').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, color='#E9C46A', linewidth=2, linestyle='solid', label="KMeans")
    ax.fill(angles, values, alpha=0.1, color='#E9C46A')

    # Add legend
    plt.legend(bbox_to_anchor=(1.15, 1.19), ncol=3, prop={"size": 13, 'weight': 'bold'},
               framealpha=0.0)  # loc='upper right', bbox_to_anchor=(0.1, 0.1)

    # ax.set_facecolor("linen")  # honeydew
    # Show the graph
    plt.savefig('evaluation_metric/' + key + '.pdf', format='pdf', bbox_inches='tight')
    plt.close()


# def radar_plot(metric_semep, metric_met, metric_km, key):
#     # Optionally use different styles for the graph
#     # Gallery: http://tonysyu.github.io/raw_content/matplotlib-style-gallery/gallery.html
#     # import matplotlib
#     # matplotlib.style.use('dark_background')  # interesting: 'bmh' / 'ggplot' / 'dark_background'
#
#     class Radar(object):
#         def __init__(self, figure, title, labels, rect=None):
#             if rect is None:
#                 rect = [0.05, 0.05, 1.0, 1.0]
#
#             self.n = len(title)
#             self.angles = np.arange(0, 360, 360.0 / self.n)
#
#             self.axes = [figure.add_axes(rect, projection='polar', label='axes%d' % i) for i in range(self.n)]
#
#             self.ax = self.axes[0]
#             self.ax.set_thetagrids(self.angles, labels=title, fontsize=14)
#
#             for ax in self.axes[1:]:
#                 ax.patch.set_visible(False)
#                 ax.grid(False)
#                 ax.xaxis.set_visible(False)
#
#             for ax, angle, label in zip(self.axes, self.angles, labels):
#                 ax.set_rgrids(range(0, 10), angle=angle, labels=label)
#                 ax.spines['polar'].set_visible(False)
#                 ax.set_ylim(0, 10)
#
#         def plot(self, values, *args, **kw):
#             angle = np.deg2rad(np.r_[self.angles, self.angles[0]])
#             values = np.r_[values, values[0]]
#             self.ax.plot(angle, values, *args, **kw)
#             self.ax.fill(angle, values, 'r', alpha=0.1)
#
#     if __name__ == '__main__':
#         fig = plt.figure(figsize=(5, 5))
#
#         tit = ['InvC', 'P', 'InvTC', 'M', 'Co']  # 12x
#
#         lab = [
#             ['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'],
#             ['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'],
#             ['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'],
#             ['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'],
#             ['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']
#         ]
#
#         radar = Radar(fig, tit, lab)
#         radar.plot(metric_semep, linestyle='solid', linewidth=2, color='b', alpha=0.7,
#                    label='SemEP')
#         radar.plot(metric_met, linestyle='solid', linewidth=2, color='r', alpha=0.7,
#                    label='METIS')
#         radar.plot(metric_km, linestyle='solid', linewidth=2, color='g', alpha=0.7,
#                    label='KMeans')
#
#         # if key == 'TransD':
#         radar.ax.legend(loc=(0.15, 1.04), ncol=3, fontsize='large')
#         fig.savefig('evaluation_metric/' + key + '.pdf', format='pdf', bbox_inches='tight')
#         plt.close(fig)

def GenerateRadarPlot(model_list, threshold):
    cls_measure = 'clusteringMeasures/'
    for model in model_list:
        matrix_address = cls_measure + model + '/'
        for th in threshold:
            metric_semep = [0, 0, 0, 0, 0]
            # metric_met = [0, 0, 0, 0, 0]
            metric_km = [0, 0, 0, 0, 0]
            f_semep = 'SemEP_' + str(th)
            # f_metis = 'METIS_' + str(th)
            f_kmeans = 'Kmeans_' + str(th)
            cls_address = matrix_address + f_semep + '/'
            # cls_address_metis = matrix_address + f_metis + '/'
            cls_address_km = matrix_address + f_kmeans + '/'

            try:
                cls_measures_semep = pd.read_csv(cls_address + model + '.txt', delimiter=",")
            except Exception:
                continue
            # try:
            #     cls_measures_met = pd.read_csv(cls_address_metis + model + '.txt', delimiter=",")
            # except Exception:
            #     continue
            cls_measures_km = pd.read_csv(cls_address_km + model + '.txt', delimiter=",")
            metric_semep = get_measure_values(cls_address, matrix_address, cls_measures_semep, metric_semep)
            # metric_met = get_measure_values(cls_address_metis, matrix_address, cls_measures_met, metric_met)
            metric_km = get_measure_values(cls_address_km, matrix_address, cls_measures_km, metric_km)
            with open('evaluation_metric/' + f_semep + model + '.txt', "w") as f:
                f.write(str(metric_semep) + "\n")
            # with open('evaluation_metric/' + f_metis + model + '.txt', "w") as f:
            #     f.write(str(metric_met) + "\n")
            with open('evaluation_metric/' + f_kmeans + model + '.txt', "w") as f:
                f.write(str(metric_km) + "\n")
            radar_plot(metric_semep, metric_km, key=str(th) + model)
            plt.cla()



# === Baseline
# metric_semep = [.15, .64, .74, .39, .47]
# metric_met = [.13, .54, .55, .34, .09]
# metric_km = [.36, .53, .75, .41, .40]
# radar_plot(metric_semep, metric_met, metric_km, 'relationalData')
