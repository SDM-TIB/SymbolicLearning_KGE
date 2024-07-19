import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import sys
# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Define the relative path to the directory containing ComputeCommunities.py
relative_path = '../PatternDetection'
# Construct the absolute path to the directory
path_to_script = os.path.abspath(os.path.join(current_dir, relative_path))
# Add this directory to the system path
sys.path.append(path_to_script)
import ComputeCommunities as SemCD

def target_cluster(kg_name, model, target_predicate, cls_algorithm, th):
    kg = SemCD.get_kg('../KG/' + kg_name + '/LungCancer.tsv')
    """Load KGE model"""
    path_model = '../KGEmbedding/' + kg_name + '/'
    df_donor = pd.read_csv(path_model + model + '/embedding_donors.csv')
    """Load ClinicalRecord responses file"""
    target = SemCD.get_target(kg, target_predicate, df_donor)
    path = '../PatternDetection/clusteringMeasures/' + model + '/' + cls_algorithm + '_' + str(th) + '/clusters/'

    list_donor = []
    entries = os.listdir(path)
    for file in entries:
        cls = pd.read_csv(path + file, delimiter="\t", header=None)
        cls.columns = ['ClinicalRecord']
        target.loc[target.ClinicalRecord.isin(cls.ClinicalRecord), 'cluster'] = 'Cluster ' + file[:-4].split('-')[1]
        list_donor = list_donor + list(cls.ClinicalRecord)

    target = target.loc[target.ClinicalRecord.isin(list_donor)]
    replacement_mapping_dict = {'No_Progression': 'No relapse',
                                'Relapse': 'Relapse',
                                'Progression': 'Relapse'}
    target['Relapse'].replace(replacement_mapping_dict, inplace=True)
    return target


def catplot(df_reset, model):
    g = sns.catplot(df_reset, kind="bar",
        x="cluster", y="count_values", hue='Relapse',
                    height=6, aspect=1.2, palette=['#264653', '#2A9D8F', '#E9C46A'])
    legend = g._legend  # Access the legend object
    # legend.set_title("Legend Title")  # Set the legend title
    # Set the legend's fontsize and other properties
    legend.get_title().set_fontsize(16)  # Set the title font size
    legend.get_texts()[0].set_fontsize(14)  # Set the label font size for the first item
    legend.get_texts()[1].set_fontsize(14)  # Set the label font size for the second item
    # legend.get_texts()[2].set_fontsize(14)  # Set the label font size for the second item
    # Change the legend position
    legend.set_bbox_to_anchor((0.6, 0.85))  # Adjust the position as needed

    g.set_axis_labels("", "Normalized Clinical Records", fontsize=16)
    plt.title('Distribution of Relapse by Cluster', fontsize=16)
    # ax.set_ylabel("Parameter values",fontsize=16)
    plt.tick_params(labelsize=16)
    plt.ylim(0, .9)
    # plt.savefig('Plots/Kmeans_norm_v2.pdf', bbox_inches='tight', format='pdf', transparent=True)
    # plt.savefig('Plots/METIS_norm_v2.pdf', bbox_inches='tight', format='pdf', transparent=True)
    plt.savefig(model+'.pdf', bbox_inches='tight', format='pdf', transparent=True)