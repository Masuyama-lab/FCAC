#
# Federated Clustering via Adaptive Resonance Theory (ART)-based Clustering (FCAC)
#

import time
import networkx as nx
import numpy as np
from tqdm.auto import tqdm
from utility_fl import *

from fcac import FCAC

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Evaluation metrics
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_mutual_info_score


# data_list = ["hillvalley", "ozone", "bioresponse", "phoneme", "texture", "optdigits", "pendigits", "mozilla4", "magic", "letter", "skin"]
data_name = "optdigits"



# experimental settings
n_trial = 20
niid = True  # True:non-iid, False:iid for federated learning
epsilon = 50  # privacy budget for \epsilon-differential privacy (-1: no noise)
max_iters = 1

# data split setting
if niid == True:
    balance = False  # Number of data points among clients. True:same, False:different
    partition = "dir"  # If set as "pat", then a train_dataset becomes pathological non-i.i.d.
    alpha = 0.5  # for Dirichlet distribution in separate_data()
else:
    balance = True  # Number of data points among clients. True:same, False:different
    partition = "pat"  # "dir", "pat"
    alpha = None  # for Dirichlet distribution in separate_data()

# for results
all_training_time = []
all_n_nodes = []
all_n_clusters = []
all_ari = []
all_ami = []
all_nmi = []


print(data_name)
for i_trial in tqdm(range(n_trial), total=n_trial, desc='Trial for Averaging'):  # for averaging

    # load dataset
    DATA, TARGET, n_clients, n_classes = set_dataset(data_name, niid, i_trial)

    # training data = test data
    train_DATA = DATA
    train_TARGET = TARGET
    test_data = DATA
    test_target = TARGET
    test_dataset = {"full_data": test_data, "true_label": test_target}

    # prepare for federated learning
    train_data, train_target, statistic = separate_data((train_DATA, train_TARGET), n_clients, n_classes, alpha, niid, balance, partition)

    # Add Laplacian noise to a train_dataset
    if epsilon == -1:  # no noise setting
        noised_train_data = train_data
    else:
        noised_train_data = [add_laplace_noise(data, epsilon, seed=i_trial) for data in train_data]

    # training
    fcac = FCAC(n_clients_=n_clients, iter_server_=max_iters)
    start = time.time()
    params_server_fcac, params_clients_fcac = fcac.fit(noised_train_data)
    all_training_time.append(time.time() - start)

    # test
    server_assignments = params_server_fcac.predict(test_data)

    # evaluation
    all_ari.append(adjusted_rand_score(test_dataset['true_label'], server_assignments))
    all_ami.append(adjusted_mutual_info_score(test_dataset['true_label'], server_assignments))
    all_nmi.append(normalized_mutual_info_score(test_dataset['true_label'], server_assignments))
    all_n_nodes.append(params_server_fcac.G_.number_of_nodes())
    all_n_clusters.append(params_server_fcac.n_clusters_)


# averaged results
print(data_name)
print('--------------- FCAC (mean result)')
print('Time:', '{:.5f}'.format(np.mean(all_training_time)), '[s]')
print(' # of Nodes:', '{:.1f}'.format(np.mean(all_n_nodes)))
print(' # of Clusters:', '{:.1f}'.format(np.mean(all_n_clusters)))
print(' ARI:', '{:.5f}'.format(np.mean(all_ari)))
print(' AMI:', '{:.5f}'.format(np.mean(all_ami)))
print(' NMI:', '{:.5f}'.format(np.mean(all_nmi)))

