#
# Copyright (c) 2023 Naoki Masuyama (masuyama@omu.ac.jp)
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php
#

from collections.abc import Iterable, Iterator
from itertools import chain, count, repeat, compress
import copy
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.estimator_checks import check_estimator


from joblib import Parallel, delayed
import multiprocessing
# parallel processing for clients
def train_clients(data_device_k, params):
    # self.perform_computation(device_x)
    params.initialization(data_device_k)
    # training a client
    [params.input_signal_client(signal, data_device_k) for signal in data_device_k]
    return params


class FCAC(BaseEstimator):
    """ Federated Clustering via Adaptive Resonance Theory (ART)-based Clustering (FCAC)"""

    def __init__(
            self,
            G_=nx.Graph(),
            dim_=None,
            num_signal_=0,
            V_thres_=1.0,
            sigma_=None,
            n_clusters_=0,
            active_node_idx_=None,
            flag_set_lambda_=False,
            n_init_data_=10,
            n_active_nodes_=np.inf,
            div_mat_=None,
            div_threshold_=1.0e-6,
            div_lambda_=np.inf,
            n_clients_=1,
            client_parameter_list_=None,
            server_parameter_list_=None,
            iter_server_=1,
            lifetime_d_edge_=0,
            n_deleted_edge_=0
    ):

        # for CA+
        self.G_ = G_  # network
        self.dim_ = dim_  # Number of variables in an instance
        self.num_signal_ = num_signal_  # Counter for training instances
        self.sigma_ = sigma_  # An estimated sigma for CIM
        self.V_thres_ = V_thres_  # Similarity threshold
        self.n_clusters_ = n_clusters_  # Number of clusters
        self.active_node_idx_ = active_node_idx_  # Indexes of active nodes
        self.flag_set_lambda_ = flag_set_lambda_  # Flag for setting \lambda
        self.n_init_data_ = n_init_data_  # Number of signals for initialization of sigma
        self.n_active_nodes_ = n_active_nodes_  # Number of buffer nodes for calculating \sigma
        self.div_mat_ = div_mat_  # A matrix for diversity via determinants
        self.div_threshold_ = div_threshold_  # A threshold for diversity via determinants

        # for CAE
        self.div_lambda_ = div_lambda_  # \lambda determined by diversity via determinants
        self.lifetime_d_edge_ = lifetime_d_edge_  # Average lifetime of deleted edges
        self.n_deleted_edge_ = n_deleted_edge_  # Number of deleted edges

        # for Federated Learning
        self.n_clients_ = n_clients_  # Number of clients
        self.client_parameter_list_ = client_parameter_list_  # A parameter list for each client
        self.server_parameter_list_ = server_parameter_list_  # A parameter list for a server
        self.iter_server_ = iter_server_  # Number of iterations for a server


    def fit(self, x: np.ndarray or list):
        """
        train data in batch manner
        :param x: array-like or ndarray
        :rtype array-like or ndarray:
            parameters of a server and clients.
        """
        params_server, params_clients = self.fcac_process(x)

        return params_server, params_clients

    def predict(self, x: np.ndarray):
        """
        predict cluster index for each sample.
        :param x: array-like or ndarray
        :rtype list:
            cluster index for each sample.
        """

        self.labels_ = self.__labeling_sample_for_clustering(x)

        return self.labels_

    def fit_predict(self, x: np.ndarray or list):
        """
        train data and predict cluster index for each sample.
        :param x: array-like or ndarray
        :rtype list:
            cluster index for each sample.
        """

        return self.fit(x).__labeling_sample_for_clustering(x)

    def fcac_process(self, x: np.ndarray or list):
        # preserve parameters of each client
        params_clients = [copy.deepcopy(self) for k in range(self.n_clients_)]
        params_server = copy.deepcopy(self)


        # training clients
        # parallel processing
        n_cores = multiprocessing.cpu_count() - 1
        params_clients = Parallel(n_jobs=n_cores)(delayed(train_clients)(device_k, params_clients) for device_k, params_clients in zip(x, params_clients))
        # serial processing
        # for k in range(self.n_clients_):
        #     device_k = x[k]  # extract a dataset for a client k
        #     param_k = params_clients[k]  # parameters of a client k
        #
        #     # self.perform_computation(device_x)
        #     param_k.initialization(device_k)
        #
        #     # training a client
        #     for signal in device_k:
        #         param_k.input_signal_client(signal, device_k)

        # extract nodes and their winning_counts
        nodes_in_clients = [np.array(list(nx.get_node_attributes(params_clients[k].G_, 'weight').values())) for k in range(self.n_clients_)]
        winning_counts_in_clients = [np.array(list(nx.get_node_attributes(params_clients[k].G_, 'winning_counts').values())) for k in range(self.n_clients_)]

        # Calculate the 75th percentile for each sublist in winning_counts_in_clients
        percentiles_75 = [np.percentile(winning_counts, 75) for winning_counts in winning_counts_in_clients]

        # For each sublist in winning_counts_in_clients and nodes_in_clients, select nodes
        # where the corresponding winning_count is greater than or equal to the 75th percentile
        selected_nodes_list = [nodes[np.where(winning_counts >= percentile_75)]
                               for winning_counts, percentile_75, nodes in zip(winning_counts_in_clients, percentiles_75, nodes_in_clients)]

        # For each sublist in winning_counts_in_clients and nodes_in_clients, select nodes
        # where the corresponding winning_count is less than the 75th percentile
        non_selected_nodes_list = [nodes[np.where(winning_counts < percentile_75)]
                                   for winning_counts, percentile_75, nodes in zip(winning_counts_in_clients, percentiles_75, nodes_in_clients)]

        selected_nodes_list = np.concatenate(selected_nodes_list, axis=0)
        non_selected_nodes_list = np.concatenate(non_selected_nodes_list, axis=0)

        # randomly shuffle the elements in each array
        np.random.seed(0)
        np.random.shuffle(selected_nodes_list)
        np.random.shuffle(non_selected_nodes_list)

        server_x = np.concatenate([selected_nodes_list, non_selected_nodes_list], 0)

        # training a server
        params_server.initialization(server_x)
        for itr_s in range(self.iter_server_):
            [params_server.input_signal_server(signal, server_x) for signal in server_x]

        # add cluster information defined by their connectivity
        params_server.__node_connectivity()

        # for client update process (overwrite clients by a server)
        # params_clients = [copy.deepcopy(params_server) for k in range(self.n_clients_)]

        return params_server, params_clients

    def initialization(self, x: np.ndarray):
        """
        Initialize parameters
        :param x: array-like or ndarray
        """
        # set graph
        if len(list(self.G_.nodes)) == 0:
            self.G_ = nx.Graph()

        # set dimension of x
        if self.dim_ is None:
            self.dim_ = x.shape[1]

    def input_signal_client(self, signal: np.ndarray, x: np.ndarray):
        """
        Input a new signal one by one, which means training in online manner.
        fit() calls __init__() before training, which means resetting the state. So the function does batch training.
        :param signal: A new input signal
        :param x: array-like or ndarray
            data
        """

        if self.num_signal_ == x.shape[0]:
            self.num_signal_ = 1
        else:
            self.num_signal_ += 1

        if self.num_signal_ == 1 and self.G_.number_of_nodes() == 0:
            self.__calculate_sigma_by_active_nodes(x[0:self.n_init_data_, :], None)  # set init \sigma

        if self.flag_set_lambda_ is False or self.G_.number_of_nodes() < self.n_active_nodes_:
            new_node_idx = self.__add_node(signal)
            self.__update_active_node_index(signal, new_node_idx)

            # setup initial n_active_nodes_, div_lambda_, and V_thres_
            self.__setup_init_params_ca()

        else:
            node_list, cim = self.__calculate_cim(signal)
            s1_idx, s1_cim, s2_idx, s2_cim = self.__find_nearest_node(node_list, cim)

            if self.V_thres_ < s1_cim or self.G_.number_of_nodes() < self.n_active_nodes_:
                new_node_idx = self.__add_node(signal)
                self.__update_active_node_index(signal, new_node_idx)
                self.__calculate_sigma_by_active_nodes(None, new_node_idx)
            else:
                self.__update_s1_node(s1_idx, signal)
                self.__update_active_node_index(signal, s1_idx)

                if self.V_thres_ >= s2_cim:
                    self.__update_s2_node(s2_idx, signal)

    def input_signal_server(self, signal: np.ndarray, x: np.ndarray):
        """
        Input a new signal one by one, which means training in online manner.
        fit() calls __init__() before training, which means resetting the state. So the function does batch training.
        :param signal: A new input signal
        :param x: array-like or ndarray
            data
        """

        if self.num_signal_ == x.shape[0]:
            self.num_signal_ = 1
        else:
            self.num_signal_ += 1

        if self.num_signal_ == 1 and self.G_.number_of_nodes() == 0:
            self.__calculate_sigma_by_active_nodes(x[0:self.n_init_data_, :], None)  # set init \sigma

        if self.flag_set_lambda_ is False or self.G_.number_of_nodes() < self.n_active_nodes_:
            new_node_idx = self.__add_node(signal)
            self.__update_active_node_index(signal, new_node_idx)

            # setup initial n_active_nodes_, div_lambda_, and V_thres_
            self.__setup_init_params_cae()

        else:
            node_list, cim = self.__calculate_cim(signal)
            s1_idx, s1_cim, s2_idx, s2_cim = self.__find_nearest_node(node_list, cim)

            if self.V_thres_ < s1_cim or self.G_.number_of_nodes() < self.n_active_nodes_:
                new_node_idx = self.__add_node(signal)
                self.__update_active_node_index(signal, new_node_idx)
                self.__calculate_sigma_by_active_nodes(None, new_node_idx)
            else:
                self.__update_s1_node(s1_idx, signal)
                self.__update_active_node_index(signal, s1_idx)
                self.__update_edge_without_deletion(s1_idx)

                if self.V_thres_ >= s2_cim:
                    self.__setup_edge(s1_idx, s2_idx)
                    self.__update_adjacent_node(s1_idx, signal)


                self.__delete_edges(s1_idx)

        if self.num_signal_ % self.div_lambda_ == 0 and self.G_.number_of_nodes() > 1:
            deleted_node_list = self.__delete_nodes(0)
            self.__delete_active_node_index(deleted_node_list)

    def __setup_init_params_ca(self):
        """
        Setup initial n_active_nodes_, div_lambda_, and V_thres_
        """

        if self.G_.number_of_nodes() >= 2 and self.flag_set_lambda_ is False:
            # calculate n_active_nodes_ and div_lambda_ based on diversity via determinants
            self.__setup_n_active_nodes_and_div_lambda_ca()

        if self.G_.number_of_nodes() == self.n_active_nodes_:
            self.flag_set_lambda_ = True

            # estimate \sigma by using active nodes
            self.__calculate_sigma_by_active_nodes()

            # overwrite \sigma of all nodes
            [nx.set_node_attributes(self.G_, {k: {'sigma': self.sigma_}}) for k in list(self.G_.nodes)]

            # get similarity threshold
            self.__calculate_threshold_by_active_nodes()

    def __setup_n_active_nodes_and_div_lambda_ca(self):
        """
        Setup n_active_nodes_ and div_lambda_ by Diversity of nodes.
        https://proceedings.neurips.cc/paper/2020/hash/d1dc3a8270a6f9394f88847d7f0050cf-Abstract.html

        Setup
        >>> net = FCAC()
        >>> net.G_ = nx.Graph()
        >>> net.G_.add_node(0, weight=[0.1, 0.5], winning_counts=1, sigma=1.0)  # node1
        >>> net.G_.add_node(1, weight=[0.9, 0.6], winning_counts=1, sigma=1.0)  # node2
        >>> net.n_active_nodes_ = np.inf
        >>> net.div_lambda_ = np.inf
        >>> net.div_threshold_ = 1.0e-6
        >>> net._FCAC__setup_n_active_nodes_and_div_lambda_ca()

        First, a pairwise CIM-based similarity matrix is calculated.
        >>> net.div_mat_
        array([[1.        , 0.86058076],
               [0.86058076, 1.        ]])

        Then, determinant of div_mat_ is calculated.
        >>> div_cim = np.linalg.det(np.exp(net.div_mat_))
        >>> div_cim
        1.7980373454447525

        In this case, div_cim < self.div_threshold_ is not satisfied.
        Thus, n_active_nodes_ and div_lambda_ are not updated.
        >>> net.n_active_nodes_
        inf
        >>> net.div_lambda_
        inf

        Adding a new node until div_cim < self.div_threshold_ is satisfied.
        >>> net.G_.add_node(2, weight=[1.0, 0.9], winning_counts=1, sigma=1.0)  # node3
        >>> net._FCAC__setup_n_active_nodes_and_div_lambda_ca()
        >>> net.div_mat_
        array([[1.        , 0.86058076, 0.79504658],
               [0.86058076, 1.        , 0.97550498],
               [0.79504658, 0.97550498, 1.        ]])
        >>> div_cim = np.linalg.det(np.exp(net.div_mat_))
        >>> div_cim
        0.21027569040368288

        Adding a new node until div_cim < self.div_threshold_ is satisfied.
        >>> net.G_.add_node(3, weight=[1.0, 0.9], winning_counts=1, sigma=1.0)  # node4
        >>> net._FCAC__setup_n_active_nodes_and_div_lambda_ca()
        >>> div_cim = np.linalg.det(np.exp(net.div_mat_))
        >>> div_cim
        -1.3286163759107123e-32
        """

        nodes_list = list(self.G_.nodes)
        _, correntropy = self.__calculate_correntropy(self.G_.nodes[nodes_list[-1]]['weight'])

        if self.G_.number_of_nodes() == 2:
            self.div_mat_ = np.array([[correntropy[1], correntropy[0]], [correntropy[0], correntropy[1]]])
        else:
            self.div_mat_ = np.insert(self.div_mat_, self.div_mat_.shape[1], correntropy[0:self.div_mat_.shape[1]],
                                      axis=0)
            self.div_mat_ = np.insert(self.div_mat_, self.div_mat_.shape[1], correntropy, axis=1)

        # div_cim = np.linalg.det(self.div_mat_)
        div_cim = np.linalg.det(np.exp(self.div_mat_))

        if div_cim < self.div_threshold_ and self.G_.number_of_nodes() >= self.n_init_data_:
            self.n_active_nodes_ = self.G_.number_of_nodes()
            self.div_lambda_ = self.n_active_nodes_ * 2

    def __calculate_sigma_by_active_nodes(self, weight: np.ndarray = None, new_node_idx: int = None):
        """
        Calculate \sigma for CIM basd on active nodes

        Setup
        >>> net = FCAC()
        >>> net.G_ = nx.Graph()
        >>> net.dim_ = 2
        >>> net.G_.add_node(0, weight=[0.1, 0.5], winning_counts=1, sigma=1.0)  # node1
        >>> net.G_.add_node(1, weight=[0.9, 0.6], winning_counts=1, sigma=1.0)  # node2
        >>> net.G_.add_node(2, weight=[1.0, 0.9], winning_counts=1, sigma=1.0)  # node3

        Depending on active nodes, a value of sigma will be changed.
        >>> net.active_node_idx_ = [0, 1]
        >>> net._FCAC__calculate_sigma_by_active_nodes()
        >>> net.sigma_
        0.2834822362263465
        >>> net.active_node_idx_ = [0, 1, 2]
        >>> net._FCAC__calculate_sigma_by_active_nodes()
        >>> net.sigma_
        0.2920448418024727

        A sigma for a new node can be set by using the current sigma.
        >>> net.G_.add_node(3, weight=[0.0, 0.1], winning_counts=1, sigma=1.0)  # node4
        >>> new_node_idx = 3
        >>> net._FCAC__calculate_sigma_by_active_nodes(None, new_node_idx)
        >>> nx.get_node_attributes(net.G_, 'sigma')
        {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.2920448418024727}
        """

        if weight is None:
            active_node_idx_ = list(self.active_node_idx_)
            n_selected_weights = np.minimum(len(active_node_idx_), self.n_active_nodes_)
            selected_weights = list(
                self.__get_node_attributes_from('weight', active_node_idx_[0:int(n_selected_weights)]))
            std_weights = np.std(selected_weights, axis=0, ddof=1)
        else:
            selected_weights = weight
            std_weights = np.std(weight, axis=0, ddof=1)
        np.putmask(std_weights, std_weights == 0.0, 1.0e-6)  # If value=0, add a small value for avoiding an error.

        # Silverman's Rule
        a = np.power(4 / (2 + self.dim_), 1 / (4 + self.dim_))
        b = np.power(np.array(selected_weights).shape[0], -1 / (4 + self.dim_))
        s = a * std_weights * b
        self.sigma_ = np.median(s)

        if new_node_idx is not None:
            nx.set_node_attributes(self.G_, {new_node_idx: {'sigma': self.sigma_}})

    def __calculate_cim(self, signal: np.ndarray):
        """
        Calculate CIM between a signal and nodes.
        Return indexes of nodes and cim value between a signal and nodes

        Setup
        >>> net = FCAC()
        >>> net.G_ = nx.Graph()
        >>> net.G_.add_node(0, weight=[0.1, 0.5], winning_counts=1, sigma=1.0)
        >>> signal = np.array([0, 0])

        Return an index and a value of the cim between a node and a signal.
        >>> net._FCAC__calculate_cim(signal)
        ([0], array([0.2474779]))

        If there are multiple nodes, return multiple indexes and values of the cim.
        >>> net.G_.add_node(1, weight=[0.9, 0.6], winning_counts=1, sigma=1.0)
        >>> net._FCAC__calculate_cim(np.array([0, 0]))
        ([0, 1], array([0.2474779 , 0.49887522]))
        """
        node_list = list(self.G_.nodes)
        weights = list(self.__get_node_attributes_from('weight', node_list))
        sigma = list(self.__get_node_attributes_from('sigma', node_list))
        c = np.exp(-(signal - np.array(weights)) ** 2 / (2 * np.mean(np.array(sigma)) ** 2))
        return node_list, np.sqrt(1 - np.mean(c, axis=1))

    def __calculate_correntropy(self, signal: np.ndarray):
        """
        Calculate CIM between a signal and nodes.
        Return indexes of nodes and cim value between a signal and nodes

        Setup
        >>> net = FCAC()
        >>> net.G_ = nx.Graph()
        >>> net.G_.add_node(0, weight=[0.1, 0.5], winning_counts=1, sigma=1.0)
        >>> signal = np.array([0, 0])

        Return an index and a value of the cim between a node and a signal.
        >>> net._FCAC__calculate_cim(signal)
        ([0], array([0.2474779]))

        If there are multiple nodes, return multiple indexes and values of the cim.
        >>> net.G_.add_node(1, weight=[0.9, 0.6], winning_counts=1, sigma=1.0)
        >>> net._FCAC__calculate_cim(np.array([0, 0]))
        ([0, 1], array([0.2474779 , 0.49887522]))
        """
        node_list = list(self.G_.nodes)
        weights = list(self.__get_node_attributes_from('weight', node_list))
        sigma = list(self.__get_node_attributes_from('sigma', node_list))
        c = np.exp(-(signal - np.array(weights)) ** 2 / (2 * np.mean(np.array(sigma)) ** 2))
        return node_list, np.mean(c, axis=1)

    def __add_node(self, signal: np.ndarray) -> int:
        """
        Add a new node to G with winning count, sigma, and label_counts.
        Return an index of the new node.

        Setup
        >>> net = FCAC()
        >>> net.G_ = nx.Graph()
        >>> net.sigma_ = 0.5
        >>> net.init_label_list_ = np.array([0,0])

        Add the 1st node to G with label=0
        >>> signal = np.array([1,2])
        >>> new_node_idx = net._FCAC__add_node(signal)
        >>> new_node_idx
        0
        >>> list(net.G_.nodes.data())
        [(0, {'weight': array([1, 2]), 'winning_counts': 1, 'sigma': 0.5})]

        Add the 2nd node to G with label=1
        >>> signal = np.array([3,4])
        >>> new_node_idx = net._FCAC__add_node(signal)
        >>> new_node_idx
        1
        >>> list(net.G_.nodes.data())
        [(0, {'weight': array([1, 2]), 'winning_counts': 1, 'sigma': 0.5}), (1, {'weight': array([3, 4]), 'winning_counts': 1, 'sigma': 0.5})]
        """
        if len(self.G_.nodes) == 0:  # for the first node
            new_node_idx = 0
        else:
            new_node_idx = max(self.G_.nodes) + 1

        # Generate node
        self.G_.add_node(new_node_idx, weight=signal, winning_counts=1, sigma=self.sigma_)

        return new_node_idx

    def __update_active_node_index(self, signal, winner_idx):
        if self.active_node_idx_ is None:
            self.active_node_idx_ = np.array([winner_idx])
        else:
            delete_idx = np.where(self.active_node_idx_ == winner_idx)
            self.active_node_idx_ = np.delete(self.active_node_idx_, delete_idx)
            self.active_node_idx_ = np.append(winner_idx, self.active_node_idx_)

    def __delete_active_node_index(self, deleted_node_list: list):
        delete_idx = [np.where(self.active_node_idx_ == deleted_node_list[k]) for k in range(len(deleted_node_list))]
        self.active_node_idx_ = np.delete(self.active_node_idx_, delete_idx)

    def __calculate_threshold_by_active_nodes(self) -> float:
        """
        Calculate a similarity threshold by using active nodes.
        Return a similarity threshold

        Setup
        >>> net = FCAC()
        >>> net.G_ = nx.Graph()
        >>> net.G_.add_node(0, weight=[0.1, 0.5], winning_counts=1, sigma=1.0)  # node1
        >>> net.G_.add_node(1, weight=[0.9, 0.6], winning_counts=1, sigma=1.0)  # node2
        >>> net.G_.add_node(2, weight=[1.0, 0.9], winning_counts=1, sigma=1.0)  # node3
        >>> net.active_node_idx_ = [0, 1, 2]
        >>> net.n_active_nodes_ = 10

        Return a mean of the minimum pairwise cims among nodes 1, 2, and 3.
        >>> net._FCAC__calculate_threshold_by_active_nodes()
        >>> net.V_thres_
        0.22880218578964573

        A simple explanation of this function is as follows:
        First, we calculate cim between nodes 1-2, and 1-3, and take min of cims.
        >>> net = FCAC()
        >>> net.G_ = nx.Graph()
        >>> signal = np.array([0.1, 0.5])
        >>> net.G_.add_node(1, weight=[0.9, 0.6], winning_counts=1, sigma=1.0)
        >>> net.G_.add_node(2, weight=[1.0, 0.9], winning_counts=1, sigma=1.0)
        >>> _, cims = net._FCAC__calculate_cim(signal)
        >>> np.min(cims)
        0.37338886146591654

        Second, we calculate cim between nodes 2-1, and 2-3, and take min of cims.
        >>> net = FCAC()
        >>> net.G_ = nx.Graph()
        >>> net.G_.add_node(0, weight=[0.1, 0.5], winning_counts=1, sigma=1.0)
        >>> signal = [0.9, 0.6]
        >>> net.G_.add_node(2, weight=[1.0, 0.9], winning_counts=1, sigma=1.0)
        >>> _, cims = net._FCAC__calculate_cim(signal)
        >>> np.min(cims)
        0.1565088479515103

        Third, we calculate cim between nodes 3-1, and 3-2, and take min of cims.
        >>> net = FCAC()
        >>> net.G_ = nx.Graph()
        >>> net.G_.add_node(0, weight=[0.1, 0.5], winning_counts=1, sigma=1.0)
        >>> net.G_.add_node(1, weight=[0.9, 0.6], winning_counts=1, sigma=1.0)
        >>> signal = [1.0, 0.9]
        >>> _, cims = net._FCAC__calculate_cim(signal)
        >>> np.min(cims)
        0.1565088479515103

        A mean of them is the same value as return from the function.
        >>> np.mean([0.37338886146591654, 0.1565088479515103, 0.1565088479515103])
        0.22880218578964573
        """

        active_node_idx_ = list(self.active_node_idx_)
        n_selected_weights = np.minimum(len(active_node_idx_), self.n_active_nodes_)
        selected_weights = list(self.__get_node_attributes_from('weight', active_node_idx_[0:int(n_selected_weights)]))
        cims = [self.__calculate_cim(w)[1] for w in selected_weights]  # Calculate a pairwise cim among nodes
        [np.putmask(cims[k], cims[k] == 0.0, 1.0) for k in range(len(cims))]  # Set cims[k][k] = 1.0
        self.V_thres_ = np.mean([np.min(cims[k]) for k in range(len(cims))])

    def __find_nearest_node(self, node_list: list, cim: np.ndarray):
        """
        Get 1st and 2nd nearest nodes from a signal.
        Return indexes and weights of the 1st and 2nd nearest nodes from a signal.

        Setup
        >>> net = FCAC()
        >>> net.G_ = nx.Graph()
        >>> net.G_.add_node(0, weight=[0.1, 0.5], winning_counts=1, sigma=1.0)

        If there is only one node, return an index and the cim value of the 1st nearest node.
        In this case, for the 2nd nearest node, an index is the same as the 1st nearest node and its value is inf.
        >>> node_list = [0]
        >>> cim = np.array([0.5])
        >>> net._FCAC__find_nearest_node(node_list, cim)
        (0, 0.5, 0, inf)

        If there are two nodes, return an index and the cim value of the 1st and 2nd nearest nodes.
        >>> net.G_.add_node(1, weight=[0.9, 0.6], winning_counts=1, sigma=1.0)
        >>> node_list = [0, 1]
        >>> cim = np.array([0.5, 0.9])
        >>> net._FCAC__find_nearest_node(node_list, cim)
        (0, 0.5, 1, 0.9)
        """

        if len(node_list) == 1:
            node_list = node_list + node_list
            cim = np.array(list(cim) + [np.inf])

        idx = np.argsort(cim)
        return node_list[idx[0]], cim[idx[0]], node_list[idx[1]], cim[idx[1]]

    def __update_s1_node(self, idx, signal):
        """
        Update weight for s1 node

        Setup
        >>> net = FCAC()
        >>> net.G_ = nx.Graph()
        >>> net.sigma_ = 1.0
        >>> net.G_.add_node(0, weight=[0.1, 0.5], winning_counts=1, sigma=net.sigma_)
        >>> signal = np.array([0,0])
        >>> s1_idx = 0
        >>> net.G_.nodes[s1_idx]
        {'weight': [0.1, 0.5], 'winning_counts': 1, 'sigma': 1.0}

        Update weight, winning_counts, and label_counts of s1 node.
        >>> net._FCAC__update_s1_node(s1_idx, signal)
        >>> net.G_.nodes[s1_idx]
        {'weight': array([0.05, 0.25]), 'winning_counts': 2, 'sigma': 1.0}
        """
        # update weight and winning_counts
        weight = self.G_.nodes[idx].get('weight')
        new_winning_count = self.G_.nodes[idx].get('winning_counts') + 1
        new_weight = weight + (signal - weight) / new_winning_count
        nx.set_node_attributes(self.G_, {idx: {'weight': new_weight, 'winning_counts': new_winning_count}})

    def __get_node_attributes_from(self, attr: str, node_list: Iterable[int]) -> Iterator:
        """
        Get an attribute of nodes in G

        Setup
        >>> net = FCAC()
        >>> net.G_ = nx.Graph()
        >>> net.G_.add_node(0, weight=[0.1, 0.5], winning_counts=1, sigma=1.0)  # node 0
        >>> net.G_.add_node(1, weight=[0.9, 0.6], winning_counts=2, sigma=2.0)  # node 1
        >>> net.G_.add_node(2, weight=[1.0, 0.9], winning_counts=3, sigma=3.0)  # node 2
        >>> node_list = list(net.G_.nodes)
        >>> node_list
        [0, 1, 2]

        Get weight of node.
        >>> list(net._FCAC__get_node_attributes_from('weight', node_list))
        [[0.1, 0.5], [0.9, 0.6], [1.0, 0.9]]

        Get winning_counts of node.
        >>> list(net._FCAC__get_node_attributes_from('winning_counts', node_list))
        [1, 2, 3]

        Get sigma of node.
        >>> list(net._FCAC__get_node_attributes_from('sigma', node_list))
        [1.0, 2.0, 3.0]
        """
        att_dict = nx.get_node_attributes(self.G_, attr)
        return map(att_dict.get, node_list)

    def __update_s2_node(self, idx, signal):
        """Update weight for s2 node
        Setup
        >>> net = FCAC()
        >>> net.G_ = nx.Graph()
        >>> net.sigma_ = 1.0
        >>> net.G_.add_node(0, weight=[0.1, 0.5], winning_counts=1, sigma=net.sigma_)
        >>> signal = np.array([0,0])
        >>> s2_idx = 0
        >>> net.G_.nodes[s2_idx]
        {'weight': [0.1, 0.5], 'winning_counts': 1, 'sigma': 1.0}

        Update weight of s2 node
        Because a learning coefficient is different from __update_s1_node(), a value of weight is different.
        In addition, winning_counts of s2 node is not updated.
        >>> net._FCAC__update_s2_node(s2_idx, signal)
        >>> net.G_.nodes[s2_idx]
        {'weight': array([0.099, 0.495]), 'winning_counts': 1, 'sigma': 1.0}
        """
        weight = self.G_.nodes[idx].get('weight')
        winning_counts = self.G_.nodes[idx].get('winning_counts')
        new_weight = weight + (signal - weight) / (100 * winning_counts)
        nx.set_node_attributes(self.G_, {idx: {'weight': new_weight}})



    # =================================================================================
    # for CAE
    # =================================================================================
    def __setup_n_active_nodes_and_div_lambda_cae(self):
        """
        Setup n_active_nodes_ and div_lambda_ by Diversity of nodes.
        https://proceedings.neurips.cc/paper/2020/hash/d1dc3a8270a6f9394f88847d7f0050cf-Abstract.html

        Setup
        >>> net = FCAC()
        >>> net.G_ = nx.Graph()
        >>> net.G_.add_node(0, weight=[0.1, 0.5], winning_counts=1, sigma=1.0)  # node1
        >>> net.G_.add_node(1, weight=[0.9, 0.6], winning_counts=1, sigma=1.0)  # node2
        >>> net.n_active_nodes_ = np.inf
        >>> net.div_lambda_ = np.inf
        >>> net.div_threshold_ = 1.0e-6
        >>> net._FCAC__setup_n_active_nodes_and_div_lambda_ca()

        First, a pairwise CIM-based similarity matrix is calculated.
        >>> net.div_mat_
        array([[1.        , 0.86058076],
               [0.86058076, 1.        ]])

        Then, determinant of div_mat_ is calculated.
        >>> div_cim = np.linalg.det(np.exp(net.div_mat_))
        >>> div_cim
        1.7980373454447525

        In this case, div_cim < self.div_threshold_ is not satisfied.
        Thus, n_active_nodes_ and div_lambda_ are not updated.
        >>> net.n_active_nodes_
        inf
        >>> net.div_lambda_
        inf

        Adding a new node until div_cim < self.div_threshold_ is satisfied.
        >>> net.G_.add_node(2, weight=[1.0, 0.9], winning_counts=1, sigma=1.0)  # node3
        >>> net._FCAC__setup_n_active_nodes_and_div_lambda_ca()
        >>> net.div_mat_
        array([[1.        , 0.86058076, 0.79504658],
               [0.86058076, 1.        , 0.97550498],
               [0.79504658, 0.97550498, 1.        ]])
        >>> div_cim = np.linalg.det(np.exp(net.div_mat_))
        >>> div_cim
        0.21027569040368288

        Adding a new node until div_cim < self.div_threshold_ is satisfied.
        >>> net.G_.add_node(3, weight=[1.0, 0.9], winning_counts=1, sigma=1.0)  # node4
        >>> net._FCAC__setup_n_active_nodes_and_div_lambda_ca()
        >>> div_cim = np.linalg.det(np.exp(net.div_mat_))
        >>> div_cim
        -1.3286163759107123e-32
        """

        nodes_list = list(self.G_.nodes)
        _, correntropy = self.__calculate_correntropy(self.G_.nodes[nodes_list[-1]]['weight'])

        if self.G_.number_of_nodes() == 2:
            self.div_mat_ = np.array([[correntropy[1], correntropy[0]], [correntropy[0], correntropy[1]]])
        else:
            self.div_mat_ = np.insert(self.div_mat_, self.div_mat_.shape[1], correntropy[0:self.div_mat_.shape[1]],
                                      axis=0)
            self.div_mat_ = np.insert(self.div_mat_, self.div_mat_.shape[1], correntropy, axis=1)

        # div_cim = np.linalg.det(self.div_mat_)
        div_cim = np.linalg.det(np.exp(self.div_mat_))

        if div_cim < self.div_threshold_ and self.G_.number_of_nodes() >= self.n_init_data_:
        # if div_cim < self.div_threshold_:
            self.n_active_nodes_ = self.G_.number_of_nodes()
            self.div_lambda_ = self.n_active_nodes_ * 2

    def __setup_init_params_cae(self):
        """
        Setup initial n_active_nodes_, div_lambda_, and V_thres_
        """

        if self.G_.number_of_nodes() >= 2 and self.flag_set_lambda_ is False:
            # calculate n_active_nodes_ and div_lambda_ based on diversity via determinants
            self.__setup_n_active_nodes_and_div_lambda_cae()

        if self.G_.number_of_nodes() == self.n_active_nodes_:
            self.flag_set_lambda_ = True

            # estimate \sigma by using active nodes
            self.__calculate_sigma_by_active_nodes()

            # overwrite \sigma of all nodes
            [nx.set_node_attributes(self.G_, {k: {'sigma': self.sigma_}}) for k in list(self.G_.nodes)]

            # get similarity threshold
            self.__calculate_threshold_by_active_nodes()

    def __get_node_attributes_from(self, attr: str, node_list: Iterable[int]) -> Iterator:
        """
        Get an attribute of nodes in G

        Setup
        >>> net = FCAC()
        >>> net.G_ = nx.Graph()
        >>> net.G_.add_node(0, weight=[0.1, 0.5], winning_counts=1, sigma=1.0)  # node 0
        >>> net.G_.add_node(1, weight=[0.9, 0.6], winning_counts=2, sigma=2.0)  # node 1
        >>> net.G_.add_node(2, weight=[1.0, 0.9], winning_counts=3, sigma=3.0)  # node 2
        >>> node_list = list(net.G_.nodes)
        >>> node_list
        [0, 1, 2]

        Get weight of node.
        >>> list(net._FCAC__get_node_attributes_from('weight', node_list))
        [[0.1, 0.5], [0.9, 0.6], [1.0, 0.9]]

        Get winning_counts of node.
        >>> list(net._FCAC__get_node_attributes_from('winning_counts', node_list))
        [1, 2, 3]

        Get sigma of node.
        >>> list(net._FCAC__get_node_attributes_from('sigma', node_list))
        [1.0, 2.0, 3.0]
        """
        att_dict = nx.get_node_attributes(self.G_, attr)
        return map(att_dict.get, node_list)

    def __get_edge_attributes_from(self, attr: str, s1_idx: int, edge_list: list) -> list:
        """
        Get an attribute of edge in G

        Setup
        >>> net = FCAC()
        >>> net.G_ = nx.Graph()
        >>> net.G_.add_node(0, weight=[0.1, 0.5], winning_counts=1, sigma=1.0)  # node 0
        >>> net.G_.add_node(1, weight=[0.9, 0.6], winning_counts=2, sigma=2.0)  # node 1
        >>> net.G_.add_node(2, weight=[1.0, 0.9], winning_counts=3, sigma=3.0)  # node 2
        >>> edge_list = list(net.G_.nodes)
        >>> edge_list
        [0, 1, 2]

        Get age of edge.


        """

        attribute_values = [self.G_.edges[(s1_idx, edge_list[k])][attr] for k in range(len(edge_list))]
        return attribute_values

    def __setup_edge(self, s1_idx: int, s2_idx: int):
        """
        Add an edge between s1 and s2 nodes with age.

        Setup
        >>> net = FCAC()
        >>> net.G_ = nx.Graph()
        >>> net.G_.add_node(0, weight=[0.1, 0.5], winning_counts=1, sigma=1.0)  # s1 node
        >>> net.G_.add_node(1, weight=[0.9, 0.6], winning_counts=2, sigma=2.0)  # s2 node
        >>> s1_idx = 0
        >>> s2_idx = 1

        At first, there is no edge between s1 and s2 nodes.
        >>> list(net.G_.edges().data())
        []

        Make an edge between s1 and s2 nodes.
        >>> net._FCAC__setup_edge(s1_idx, s2_idx)
        >>> list(net.G_.edges().data())
        [(0, 1, {'age': 1})]
        """
        self.G_.add_edge(s1_idx, s2_idx, age=1)  # set an edge between s1 and s2 nodes

    def __update_edge_without_deletion(self, s1_idx: int):
        """
        Update age of edge connected to s1 node and delete an edge if age > max_edge_age.

        Setup
        >>> net = FCAC()
        >>> net.G_ = nx.Graph()
        >>> net.G_.add_node(0, weight=[0.1, 0.5], winning_counts=1, sigma=1.0)  # node 0
        >>> net.G_.add_node(1, weight=[0.9, 0.6], winning_counts=1, sigma=1.0)  # node 1
        >>> net.G_.add_node(2, weight=[1.0, 0.9], winning_counts=1, sigma=1.0)  # node 2
        >>> net.max_edge_age = 3

        There is edges between nodes 0-1 and nodes 0-2
        >>> net.G_.add_edge(0, 1, age=1)
        >>> net.G_.add_edge(0, 2, age=1)

        Suppose node 0 is s1_idx.
        >>> net._FCAC__update_edge_without_deletion(0)
        >>> net.G_.edges().data()
        EdgeDataView([(0, 1, {'age': 2}), (0, 2, {'age': 2})])

        Suppose node 1 is s1_idx.
        >>> net._FCAC__update_edge_without_deletion(1)
        >>> net.G_.edges().data()
        EdgeDataView([(0, 1, {'age': 3}), (0, 2, {'age': 2})])

        """
        # get neighbors of s1
        n_s1_list = np.array(list(self.G_.neighbors(s1_idx)))

        if n_s1_list.size != 0:
            # get edge ages of s1 neighbor nodes
            age_dict = nx.get_edge_attributes(self.G_, 'age')
            ages = np.array(list(map(age_dict.get, map(lambda t: tuple(sorted(t)), zip(repeat(s1_idx), n_s1_list)))))

            # increment edge ages of s1 neighbor nodes
            ages = ages + 1

            # update age of edges
            nx.set_edge_attributes(self.G_, dict(zip(zip(repeat(s1_idx), n_s1_list), ages)), 'age')

    def __delete_edges(self, s1_idx: int):
        """
        Delete edges based on lifetime_d_edge_ and n_deleted_edge_.
        This edge deletion mechanism is heavily inspired by SOINN+.
        C. Wiwatcharakoses, and D. Berrar,
        "SOINN+, a self-organizing incremental neural network for unsupervised learning from noisy data streams,"
        Expert Systems and Application, vol. 143, no. 1, #. 113069, April 2020.
        https://doi.org/10.1016/j.eswa.2019.113069
        Code provided by authors: https://osf.io/6dqu9/

        """

        # get idx of neighbor nodes
        n_s1 = list(self.G_.neighbors(s1_idx))

        if n_s1:
            # calculate edge deletion threshold
            edge_age = np.array(list(self.__get_edge_attributes_from('age', s1_idx, n_s1)))
            edge_age_sorted = np.sort(edge_age)
            median_edge_age_sorted = np.median(edge_age_sorted)
            q1_filter = edge_age_sorted < median_edge_age_sorted
            q3_filter = edge_age_sorted > median_edge_age_sorted

            if any(q1_filter) and any(q3_filter):  # if q1 and q3 are not NaN
                q1 = np.median(list(compress(edge_age_sorted, q1_filter)))
                q3 = np.median(list(compress(edge_age_sorted, q3_filter)))
                # omega_edge = q3 + (1.0 * (q3 - q1))
                omega_edge = q3 + (0.1 * (q3 - q1))
                ratio = self.n_deleted_edge_ / (self.n_deleted_edge_ + len(edge_age))
                edge_deletion_threshold = self.lifetime_d_edge_ * ratio + omega_edge * (1.0 - ratio)

                # indexes of edges to be deleted
                flag_delete = edge_age > edge_deletion_threshold
                delete_edge_index = list(compress(n_s1, flag_delete))

                if delete_edge_index:
                    # update n_deleted_edge_ and lifetime_d_edge_
                    numerator = self.n_deleted_edge_ * self.lifetime_d_edge_ + np.sum(
                        list(compress(edge_age, flag_delete)))
                    self.n_deleted_edge_ = self.n_deleted_edge_ + len(delete_edge_index)
                    self.lifetime_d_edge_ = numerator / self.n_deleted_edge_

                    # delete edge
                    d_edge = zip(repeat(s1_idx), delete_edge_index)
                    self.G_.remove_edges_from(d_edge)

    def __update_adjacent_node(self, s1_idx: int, signal: np.ndarray):
        """Update weight of s1 neighbors

        Setup
        >>> net = FCAC()
        >>> net.G_ = nx.Graph()
        >>> net.G_.add_node(0, weight=[0.1, 0.5], winning_counts=1, sigma=1.0)  # node 0
        >>> net.G_.add_node(1, weight=[0.9, 0.6], winning_counts=1, sigma=1.0)  # node 1
        >>> net.G_.add_node(2, weight=[1.0, 0.9], winning_counts=1, sigma=1.0)  # node 2
        >>> signal = np.array([1,2])

        There is edges between nodes 0-1 and nodes 0-2.
        >>> net.G_.add_edge(0, 1, age=4)
        >>> net.G_.add_edge(0, 2, age=1)

        Suppose node 0 is s1_idx.
        >>> s1_idx = 0

        Update weight of s1 neighbor nodes.
        >>> net._FCAC__update_adjacent_node(s1_idx, signal)

        Weights of node 1 and 2 are updated.
        >>> [np.array(net.G_.nodes[k]['weight']) for k in range(3)]
        [array([0.1, 0.5]), array([0.91, 0.74]), array([1.  , 1.01])]
        """
        # get idx of neighbor nodes
        n_s1 = list(self.G_.neighbors(s1_idx))

        if n_s1:
            # get weights of neighbor nodes
            weights = np.array(list(self.__get_node_attributes_from('weight', n_s1)))

            # get winning counts of neighbor nodes
            win_counts = np.array(list(self.__get_node_attributes_from('winning_counts', n_s1)))

            # update weights
            new_weights = weights + (signal - weights) / (10 * np.array([win_counts]).T)
            nx.set_node_attributes(self.G_, dict(zip(n_s1, new_weights)), 'weight')

    def __delete_nodes(self, degree: int) -> list:
        """
        Delete nodes if self.G.degree(x) <= degree
        Return an index of node to be deleted.

        Setup
        >>> net = FCAC()
        >>> net.G_ = nx.Graph()
        >>> net.G_.add_node(0, weight=[0.1, 0.5], winning_counts=1, sigma=1.0)  # node 0
        >>> net.G_.add_node(1, weight=[0.9, 0.6], winning_counts=1, sigma=1.0)  # node 1
        >>> net.G_.add_node(2, weight=[1.0, 0.9], winning_counts=1, sigma=1.0)  # node 2

        Suppose there is an edge between nodes 0-1.
        It means that node 2 is isolated.
        >>> net.G_.add_edge(0, 1, age=1)
        >>> net.G_.nodes
        NodeView((0, 1, 2))

        An isolated node (i.e., node 2) will be deleted.
        >>> degree = 0
        >>> to_be_removed = net._FCAC__delete_nodes(degree)
        >>> to_be_removed
        [2]
        >>> net.G_.nodes
        NodeView((0, 1))
        """
        to_be_removed = [x for x in self.G_.nodes() if self.G_.degree(x) <= degree]
        self.G_.remove_nodes_from(to_be_removed)
        return to_be_removed

    def __node_connectivity(self):
        """
        Add cluster information defined by their connectivity (i.e., cluster) to nodes

        """
        # get cluster of nodes and order of nodes
        cc_list = list(nx.connected_components(self.G_))  # get connection information, e.g., [[1,3],[0],[5,6,2]]

        # add an attribute 'cluster' to each node in G based on their connectivity
        [[nx.set_node_attributes(self.G_, {m: {'cluster': k}}) for m in list(cc_list[k])] for k in range(len(cc_list))]

        # get number of clusters
        self.n_clusters_ = len(cc_list)

    def __labeling_sample_for_clustering(self, x: np.ndarray) -> list:
        """
        A label of testing sample is determined by connectivity of nodes.
        Labeled samples should be evaluated by using clustering metrics.

        """
        # get cluster of nodes and order of nodes
        cc_list = list(nx.connected_components(self.G_))  # get connection information, e.g., [[1,3],[0],[5,6,2]]
        node_iter = chain.from_iterable(cc_list)  # flatten process, e.g., [[1,3],[0],[5,6,2]] -> [1,3,0,5,6,2]
        cluster_list = np.array(list(chain.from_iterable(map(repeat, count(0), map(len, cc_list)))))  # labeling process, e.g., [[1,3],[0],[5,6,2]] -> [2,1,3] -> [[0,0],[1],[2,2,2]] -> [0,0,1,2,2,2]

        # get weights matrix sorted along with node_iter
        weights = list(self.__get_node_attributes_from('weight', node_iter))

        # get label for each data in x with the nearest node
        # compute cim between x and nodes
        sigmas = list(self.__get_node_attributes_from('sigma', list(self.G_.nodes)))
        c = [np.exp(-(x[k, :] - np.array(weights)) ** 2 / (2 * np.mean(np.array(sigmas)) ** 2)) for k in range(len(x))]
        cim = [np.sqrt(1 - np.mean(c[k], axis=1)) for k in range(len(x))]

        # get indexes of the nearest neighbor
        nearest_node_idx = np.argmin(cim, axis=1)

        return cluster_list[nearest_node_idx]

    def plotting_ca_plus(self, x: np.ndarray = None, fig_name=None):
        fig, ax = plt.subplots()
        fig.tight_layout()
        if fig_name is not None:
            plt.title(fig_name)
        if x is not None:
            plt.plot(x[:, 0], x[:, 1], 'cx', zorder=1)
        nx.draw(self.G_, pos=nx.get_node_attributes(self.G_, 'weight'), node_size=40, node_color='r', with_labels=False,
                ax=ax)
        ax.set_axis_on()
        ax.set_axisbelow(True)
        ax.set_aspect('equal')
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        plt.grid(True)
        plt.show()


class ClusterFCAC(FCAC, ClusterMixin):
    pass


if __name__ == '__main__':

    # https://docs.python.org/3.10/library/doctest.html
    import doctest

    doctest.testmod()

    # check_estimator(FCAC())
