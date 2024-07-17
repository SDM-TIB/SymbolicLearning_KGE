import pandas as pd
import os, sys, glob
from os import listdir
from os.path import isfile, join
from shutil import copyfile
import Utility
import matplotlib.pyplot as plt


def call_semEP(threshold, cls_addres, file_addres):
    DIR_SEM_EP = "semEP-node"
    current_path = os.path.dirname(os.path.realpath(__file__))
    th = "{:.4f}".format(float(threshold))

    commd = current_path + "/" + DIR_SEM_EP + " " + file_addres + "ClinicalRecord.txt " + file_addres + "matrix_ClinicalRecord.tsv " + str(
        th)
    print('commd: ' + commd)
    os.system(commd)
    pattern = 'ClinicalRecord'

    results_folder = glob.glob(current_path + "/" + pattern + "-*")
    # print(results_folder)
    onlyfiles = [os.path.join(results_folder[0], f) for f in listdir(results_folder[0]) if
                 isfile(join(results_folder[0], f))]
    count = 0
    for filename in onlyfiles:
        # print(filename)
        key = "cluster-" + str(count) + '.txt'
        copyfile(filename, cls_addres + 'clusters/' + key)
        count += 1
    # dicc_clusters = get_dicc_clusters(onlyfiles)

    for r, d, f in os.walk(results_folder[0]):
        for files in f:
            os.remove(os.path.join(r, files))
        os.removedirs(r)
    return len(onlyfiles)


def METIS_Undirected_MAX_based_similarity_graph(cosine_matrix, cls_address_metis):
    metislines = []
    nodes = {"name": [], "id": []}
    kv = 1
    edges = 0
    for i, row in cosine_matrix.iterrows():
        val = ""
        ix = 1
        ledges = 0
        found = False
        for k in row.keys():
            if i != k and row[k] > 0:
                val += str(ix) + " " + str(int(row[k] * 100000)) + " "
                # Only one edge is counted between two nodes, i.e., (u,v) and (v, u) edges are counted as one
                # Self links are also ignored, Notive ix>kv
                # if ix > kv:
                ledges += 1
                found = True
            ix += 1
        if found:
            # This node is connected
            metislines.append(val.strip())
            edges += ledges
            nodes["name"].append(i)
            nodes['id'].append(str(kv))
        else:
            # disconnected RDF-MTs are given 10^6 value as similarity value
            metislines.append(str(kv) + " 100000")
            edges += 1
            # ---------
            nodes["name"].append(i)
            nodes['id'].append(str(kv))
            print(i)
            print(str(kv))

        kv += 1
    nodes = pd.DataFrame(nodes)
    # print(edges)
    numedges = edges // 2
    # == Save filemetis.graph to execute METIS algorithm ==
    ff = open(cls_address_metis + 'metis.graph', 'w+')
    ff.write(str(cosine_matrix.shape[0]) + " " + str(numedges) + " 001\n")
    met = [m.strip() + "\n" for m in metislines]
    ff.writelines(met)
    ff.close()
    return nodes


def call_metis(num_cls, nodes, cls_address_metis):
    # !sudo docker run -it --rm -v /media/rivas/Data1/Data-mining/KCAP-I40KG-Embeddings/I40KG-Embeddings/result/TransD/metis:/data kemele/metis:5.1.0 gpmetis metis.graph 2
    current_path = os.path.dirname(os.path.realpath(__file__))
    EXE_METIS = "sudo docker run -it --rm -v "
    DIR_METIS = ":/data kemele/metis:5.1.0 gpmetis"
    cls_addres = cls_address_metis[:-1]
    commd = EXE_METIS + current_path + '/' + cls_addres + DIR_METIS + " metis.graph " + str(num_cls)
    print(commd)
    os.system(commd)
    parts = open(cls_address_metis + 'metis.graph.part.' + str(num_cls)).readlines()
    parts = [p.strip() for p in parts]
    # == Save each partition standads into a file ==
    i = 0
    partitions = dict((str(k), []) for k in range(num_cls))
    for p in parts:
        name = nodes.iat[i, 0]
        i += 1
        partitions[str(p)].append(name)

    i = 0
    count = 0
    for p in partitions:
        if len(partitions[p]) == 0:
            continue
        count += len(partitions[p])
        f = open(cls_address_metis + 'clusters/cluster-' + str(i) + '.txt', 'w+')
        [f.write(l + '\n') for l in partitions[p]]
        f.close()
        i += 1


def cluster_statistics(df, cls_statistics, num_cls, cls_address):
    for c in range(num_cls):
        try:
            No_Progression = df.loc[df.cluster == c][['Relapse']].value_counts()['entity:No_Progression']
        except KeyError:
            No_Progression = 0
        try:
            Progression = df.loc[df.cluster == c][['Relapse']].value_counts()['entity:Progression']
        except KeyError: #AttributeError
            Progression = 0
        try:
            Relapse = df.loc[df.cluster == c][['Relapse']].value_counts()['entity:Relapse']
        except KeyError:
            Relapse = 0
        try:
            UnKnown = df.loc[df.cluster == c][['Relapse']].value_counts()['entity:UnKnown']
        except KeyError:
            UnKnown = 0
        cls_statistics.at['entity:No_Progression', 'cluster-' + str(c)] = int(No_Progression)  # / 14
        cls_statistics.at['entity:Progression', 'cluster-' + str(c)] = int(Progression)  # / 14
        cls_statistics.at['entity:Relapse', 'cluster-' + str(c)] = int(Relapse)  # / 14
        cls_statistics.at['entity:UnKnown', 'cluster-' + str(c)] = int(UnKnown)  # / 73
    cls_statistics.to_csv(cls_address + 'cls_statistics.csv')


def get_measure(cls_measure, cls_folder, f_model):
    # measure = ' ./cma '+cls_folder + '/clusters/ ' + f_model+'/donor.txt ' + f_model +'/matrix_sim.txt > ' + cls_folder+'/'+f_model+'.txt'
    measure = '/cma ' + cls_folder + '/clusters/ ' + f_model + '/ClinicalRecord.txt ' + f_model + '/matrix_sim.txt'
    # c_measure = cls_measure[:-1]
    print(cls_measure)
    print('commd: ', cls_measure + measure)
    os.system(cls_measure + measure)


def update_cluster_folder(cls_address):
    if os.path.exists(cls_address + 'clusters/'):
        current_path = os.path.dirname(os.path.realpath(__file__))
        results_folder = glob.glob(current_path + '/' + cls_address + 'cluster*')
        for r, d, f in os.walk(results_folder[0]):
            for files in f:
                os.remove(os.path.join(r, files))
    else:
        # os.makedirs(cls_address)
        os.makedirs(cls_address + 'clusters/')


# threshold = [50, 52, 55, 57, 60, 63, 65]
threshold = [85, 87]
# model_list = ['TransH', 'DistMult', 'TransE']
# threshold = [20, 22, 25, 27, 30, 32, 35, 37, 40, 42, 45]
model_list = ['TransE']

# path_model = '../models/'
path_model = '../KGEmbedding/models/'
cls_measure = '/media/rivas/Data1/Projects/ImProVIT_Project/BiosampleDayN/PatternDetection/clusteringMeasures'

"""Load ClinicalRecord responses file"""
target = pd.read_csv('../PreprocessData/data/target.csv')
for m in model_list:
    """Load KGE model"""
    df_donor = pd.read_csv(path_model + m + '/embedding_donors.csv')
    """Labeling donors in the DataFrame"""
    df_donor = pd.merge(df_donor, target, on="ClinicalRecord")
    # print('df_donor:', df_donor.ClinicalRecord.to_list())
    file_address = 'clusteringMeasures/' + m + '/'
    path_plot = '../Plots/' + m + '/'
    for th in threshold:
        cls_address = file_address + 'SemEP_' + str(th) + '/'
        cls_address_metis = file_address + 'METIS_' + str(th) + '/'

        update_cluster_folder(cls_address)

        """Create similarity matrix of Donors"""
        sim_matrix, percentile, list_sim = Utility.matrix_similarity(df_donor.drop(columns=['Relapse']),
                                                                     Utility.cosine_sim,
                                                                     th)  # cosine_sim, euclidean_distance
        # print(sim_matrix.columns.to_list())
        Utility.SemEP_structure(file_address + 'matrix_ClinicalRecord.tsv', sim_matrix, sep=' ')
        sim_matrix.to_csv(file_address + 'matrix_sim.txt', index=False, float_format='%.5f', mode='w+', header=False)
        Utility.create_entitie(sim_matrix.columns.to_list(), file_address + 'ClinicalRecord.txt')
        """Execute SemEP"""
        num_cls = call_semEP(percentile, cls_address, file_address)
        """METIS"""
        update_cluster_folder(cls_address_metis)
        if num_cls > 1:
            nodes = METIS_Undirected_MAX_based_similarity_graph(sim_matrix, cls_address_metis)
            call_metis(num_cls, nodes, cls_address_metis)
        """Labeling donors in the matrix"""
        # Merge based on the conditions you specified
        # print('sim_matrix.columns:')
        # for i in sim_matrix.columns:
        #     print(i)
        # print('sim_matrix.index:')
        # for i in sim_matrix.index:
        #     print(i)
        sim_matrix = sim_matrix.merge(target, left_index=True, right_on='ClinicalRecord', suffixes=('_df1', '_df2'))
        # print('sim_matrix.columns:')
        # for i in sim_matrix.columns:
        #     print(i)
        # Drop redundant columns
        # sim_matrix = sim_matrix.drop(['ClinicalRecord_df1', 'ClinicalRecord_df2'], axis=1)

        cls_statistics = pd.DataFrame(columns=['cluster-' + str(x) for x in range(num_cls)],
                                      index=['entity:No_Progression', 'entity:Progression', 'entity:Relapse',
       'entity:UnKnown'])
        entries = os.listdir(cls_address + 'clusters/')
        for file in entries:
            sim_matrix.loc[
                sim_matrix.ClinicalRecord.isin(Utility.load_cluster(file, cls_address + 'clusters/')), 'cluster'] = int(
                file[:-4].split('-')[1])
            df_donor.loc[
                df_donor.ClinicalRecord.isin(Utility.load_cluster(file, cls_address + 'clusters/')), 'cluster'] = int(
                file[:-4].split('-')[1])
        """Compute statistics for each cluster"""
        cluster_statistics(sim_matrix.drop(['ClinicalRecord'], axis=1), cls_statistics, num_cls, cls_address)

        if not os.path.exists(path_plot):
            os.makedirs(path_plot)
        if len(entries) < 9:
            new_df = Utility.plot_semEP(len(entries), sim_matrix.drop(['ClinicalRecord'], axis=1), path_plot, 'PCA_th_' + str(th) + 'matrix.pdf',
                                        scale=False)
            new_df[['Relapse', 'cluster']].to_csv(path_plot + 'th_' + str(th) + '_summary.csv')
            Utility.plot_semEP(len(entries), df_donor.drop(columns=['ClinicalRecord']), path_plot, 'PCA_th_' + str(th) + '.pdf',
                               scale=False)
        df_donor.drop(columns=['cluster'], inplace=True)

        """Execute Kmeans"""
        # sim_matrix['ClinicalRecord'] = sim_matrix.index
        sim_matrix.drop(columns=['cluster'], inplace=True)
        kmeans_address = file_address + 'Kmeans_' + str(th) + '/'
        if not os.path.exists(kmeans_address):
            os.makedirs(kmeans_address)
        # num_cls = Utility.elbow_KMeans(sim_matrix.iloc[:, :-2], 1, 15, kmeans_address)  # df_donor
        # if num_cls is None:
        #     num_cls = 15
        new_df, cls_report = Utility.plot_cluster(num_cls, sim_matrix, kmeans_address, scale=False)  # df_donor
        new_df.to_csv(kmeans_address + 'cluster.csv', index=None)
        update_cluster_folder(kmeans_address)
        """Save Kmeans-Clusters"""
        for cls in range(num_cls):
            new_df.loc[new_df.cluster == cls][['ClinicalRecord']].to_csv(
                kmeans_address + 'clusters/' + 'cluster-' + str(cls) + '.txt', index=None, header=None)
        """Compute statistics for each cluster"""
        cls_statistics = pd.DataFrame(columns=['cluster-' + str(x) for x in range(num_cls)],
                                      index=['entity:No_Progression', 'entity:Progression', 'entity:Relapse',
       'entity:UnKnown'])
        cluster_statistics(new_df, cls_statistics, num_cls, kmeans_address)


    """Visualize Donors"""
    Utility.plot_treatment(df_donor, path_plot)
    # """Execute Kmeans"""
    # sim_matrix['donor'] = sim_matrix.index
    # sim_matrix.drop(columns=['cluster'], inplace=True)
    # # print(sim_matrix, '===', df_donor)
    #
    # kmeans_address = file_address + 'Kmeans/'
    # if not os.path.exists(kmeans_address):
    #     os.makedirs(kmeans_address)
    # num_cls = Utility.elbow_KMeans(sim_matrix.iloc[:, :-2], 1, 15, kmeans_address) #df_donor
    # # if num_cls is not None:
    # if num_cls is None:
    #     num_cls = 15
    # new_df, cls_report = Utility.plot_cluster(num_cls, sim_matrix, kmeans_address, scale=False) #df_donor
    # new_df.to_csv(kmeans_address + 'cluster.csv', index=None)
    # update_cluster_folder(kmeans_address)
    # """Save Kmeans-Clusters"""
    # for cls in range(num_cls):
    #     new_df.loc[new_df.cluster == cls][['donor']].to_csv(
    #         kmeans_address + 'clusters/' + 'cluster-' + str(cls) + '.txt', index=None, header=None)
    # """Compute Cluster-Measures"""
    # # get_measure(cls_measure, m + '/Kmeans', m)
    # """Compute statistics for each cluster"""
    # cls_statistics = pd.DataFrame(columns=['cluster-' + str(x) for x in range(num_cls)],
    #                                 index=['cured', 'non_cured'])
    # cluster_statistics(new_df, cls_statistics, num_cls, kmeans_address)

    """Density of Donor Similarity"""
    # print(list_sim, path_plot)
    Utility.density_plot(list_sim, path_plot)
    # ax = standard_similarity["similarity"].plot.kde(bw_method=0.1)
    # fig = ax.get_figure()
    # fig.savefig(path_plot + 'SimilarityDensity.pdf', format='pdf', bbox_inches='tight')
    # plt.close()

# def main(*args):
#     """Load TriplesFactory"""
#     tf_data, donor = load_network_from_nt_file(args[1])
#     """Split dataset in training and test set"""
#     training, testing = tf_data.split(random_state=1234)
#     """Build Knowledge Graph Embedding Model"""
#     model_list = ['DistMult', 'TransE', 'TransH', 'ERMLP', 'RESCAL']
#     # model_list = ['DistMult', 'TransE', 'TransH']
#     for m in model_list:
#         model, results = create_model(training, testing, m, 200, args[0])
#         """Obtain the embeddings of entities and relations"""
#         entity_embedding_tensor, relation_embedding_tensor = get_learned_embeddings(model)
#         """Save the embeddings of donor entities"""
#         df_donor, new_df, df_g1 = dataframe_embedding_donors(entity_embedding_tensor, donor, tf_data)
#         new_df.to_csv(args[0]+m+'/embedding_donors.csv', index=None)
#
#
# if __name__ == '__main__':
#     main(*sys.argv[1:])
