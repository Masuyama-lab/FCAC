
import numpy as np
import scipy as sp
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder

from sklearn.cluster import KMeans
import pickle
import math




def load_openml_dataset_for_federate_learning(data_name, num_clients, alpha, niid, balance, partition):
    tmp_data, tmp_target = fetch_openml(name=data_name, return_X_y=True, as_frame=False, parser="auto")
    tmp_target = np.array(tmp_target, dtype='int')
    num_classes = np.unique(tmp_target).shape[0]

    data, target, statistic = separate_data((tmp_data, tmp_target), num_clients, num_classes, alpha, niid, balance, partition)

    return data, target, num_classes

def load_openml_dataset(data_name, num_clients, alpha, niid, balance, partition):
    data, tmp_target = fetch_openml(name=data_name, return_X_y=True, as_frame=False, parser="auto")

    # label encoding (A->1, B->2,...)
    df = pd.DataFrame(tmp_target, columns=['label'])
    le = LabelEncoder()
    df['encoded'] = le.fit_transform(df['label'])
    tmp_target = np.array(df['encoded'])
    target = np.array(tmp_target, dtype='int')

    num_classes = np.unique(target).shape[0]

    return data, target, num_classes

def set_dataset(data_name, niid=True, SEED=0):
    # Dataset Configuration --------------------------------------------------------------
    # for i.i.d. scenario
    # Number of data points and classes are uniformly distributed in each client
    # niid = False  # True:non-iid, False:iid
    # balance = True  # Number of data points among clients. True:same, False:different
    # partition = "pat"  # "dir", "pat"
    # alpha = None  # for Dirichlet distribution in separate_data()

    # for practical non-i.i.d. scenario
    # Number of data points and classes are not uniformly distributed in each client
    # niid = True  # True:non-iid, False:iid
    # balance = True  # Number of data points among clients. True:same, False:different
    # partition = "dir"  # If set as "pat", then a dataset becomes pathological non-i.i.d.
    # alpha = 1.0  # for Dirichlet distribution in separate_data()
    # ------------------------------------------------------------------------------------

    # np.random.seed(SEED)

    if niid == True:
        balance = False  # Number of data points among clients. True:same, False:different
        partition = "dir"  # If set as "pat", then a dataset becomes pathological non-i.i.d.
        alpha = 0.5  # for Dirichlet distribution in separate_data()
    else:
        balance = True  # Number of data points among clients. True:same, False:different
        partition = "pat"  # "dir", "pat"
        alpha = None  # for Dirichlet distribution in separate_data()

    selected_dataset = data_name

    if selected_dataset == "mnist":
        num_clients = 100
        num_classes = 10
        dir_path = "../Dataset/FL/mnist/"
        DATA, TARGET = generate_mnist(dir_path, num_clients, num_classes, alpha, niid, balance, partition)

    elif selected_dataset == "fmnist":
        num_clients = 100
        num_classes = 10
        dir_path = "../Dataset/FL/fmnist/"
        DATA, TARGET = generate_fmnist(dir_path, num_clients, num_classes, alpha, niid, balance, partition)

    elif selected_dataset == "har":
        num_clients = 50
        num_classes = 6
        dir_path = "../Dataset/FL/har/"
        DATA, TARGET = generate_har(dir_path, num_clients, num_classes, alpha, niid, balance, partition)

    elif selected_dataset == "plants":
        # https://openml.org/search?type=data&status=any&id=1491
        selected_dataset = 'one-hundred-plants-margin'
        num_clients = 5
        DATA, TARGET, num_classes = load_openml_dataset(selected_dataset, num_clients, alpha, niid, balance, partition)

    elif selected_dataset == "madelon":
        # https://www.openml.org/search?type=data&status=active&id=1485
        num_clients = 10
        DATA, TARGET, num_classes = load_openml_dataset(selected_dataset, num_clients, alpha, niid, balance, partition)

    elif selected_dataset == "bioresponse":
        # https://www.openml.org/search?type=data&status=active&id=4134
        selected_dataset = 'Bioresponse'
        num_clients = 10
        DATA, TARGET, num_classes = load_openml_dataset(selected_dataset, num_clients, alpha, niid, balance, partition)

    elif selected_dataset == "waveform":
        # https://openml.org/search?type=data&status=active&id=60
        selected_dataset = 'waveform-5000'
        num_clients = 10
        DATA, TARGET, num_classes = load_openml_dataset(selected_dataset, num_clients, alpha, niid, balance, partition)

    elif selected_dataset == "phoneme":
        # https://openml.org/search?type=data&status=any&id=1489
        num_clients = 10
        DATA, TARGET, num_classes = load_openml_dataset(selected_dataset, num_clients, alpha, niid, balance, partition)

    elif selected_dataset == "texture":
        # https://openml.org/search?type=data&status=active&id=40499
        num_clients = 50
        DATA, TARGET, num_classes = load_openml_dataset(selected_dataset, num_clients, alpha, niid, balance, partition)

    elif selected_dataset == "optdigits":
        # https://openml.org/search?type=data&status=active&id=28
        num_clients = 50
        DATA, TARGET, num_classes = load_openml_dataset(selected_dataset, num_clients, alpha, niid, balance, partition)

    elif selected_dataset == "pendigits":
        # https://openml.org/search?type=data&status=any&id=32
        num_clients = 50
        DATA, TARGET, num_classes = load_openml_dataset(selected_dataset, num_clients, alpha, niid, balance, partition)

    elif selected_dataset == "mozilla4":
        # https://openml.org/search?type=data&sort=runs&status=any&id=1046
        num_clients = 10
        selected_dataset = 'mozilla4'
        DATA, TARGET, num_classes = load_openml_dataset(selected_dataset, num_clients, alpha, niid, balance, partition)

    elif selected_dataset == "isolet":
        # https://openml.org/search?type=data&sort=runs&status=active&id=300
        num_clients = 100
        selected_dataset = 'isolet'
        DATA, TARGET, num_classes = load_openml_dataset(selected_dataset, num_clients, alpha, niid, balance, partition)

    elif selected_dataset == "magic":
        # https://www.openml.org/search?type=data&status=active&id=1120
        selected_dataset = "MagicTelescope"
        num_clients = 50
        DATA, TARGET, num_classes = load_openml_dataset(selected_dataset, num_clients, alpha, niid, balance, partition)

    elif selected_dataset == "letter":
        # https://openml.org/search?type=data&status=any&id=6
        num_clients = 100
        DATA, TARGET, num_classes = load_openml_dataset(selected_dataset, num_clients, alpha, niid, balance, partition)

    # elif selected_dataset == "click":
    #     # https://www.openml.org/search?type=data&status=active&id=1220
    #     selected_dataset = "Click_prediction_small"
    #     num_clients = 5
    #     DATA, TARGET, num_classes = load_openml_dataset(selected_dataset, num_clients, alpha, niid, balance, partition)

    elif selected_dataset == "hillvalley":
        # https://www.openml.org/search?type=data&status=active&id=1479
        selected_dataset = 'hill-valley'
        num_clients = 5
        DATA, TARGET, num_classes = load_openml_dataset(selected_dataset, num_clients, alpha, niid, balance, partition)

    elif selected_dataset == "ozone":
        # https://www.openml.org/search?type=data&status=active&id=1487
        selected_dataset = 'ozone-level-8hr'
        num_clients = 5
        DATA, TARGET, num_classes = load_openml_dataset(selected_dataset, num_clients, alpha, niid, balance, partition)

    elif selected_dataset == "skin":
        # https://www.openml.org/search?type=data&status=active&id=1502
        selected_dataset = "skin-segmentation"
        num_clients = 100
        DATA, TARGET, num_classes = load_openml_dataset(selected_dataset, num_clients, alpha, niid, balance, partition)

    else:
        raise Exception('Select an existing dataset.')

    # randomize data
    np.random.seed(SEED)
    m = len(DATA)
    idx = np.random.permutation(m)
    # Shuffle DATA and TARGET arrays using the shuffled index
    DATA = DATA[idx]
    TARGET = TARGET[idx]

    return DATA, TARGET, num_clients, num_classes


# data_list = ["har", "mnist", "fmnist"]
# data_list = ["plants", "waveform", "texture", "optdigits", "pendigits", "mozilla4", "isolet", "letter"]

# data_name = "plants"
# niid = True  # True:non-iid, False:iid
# SEED = 1  # random seed
#
# DATA, TARGET, num_clients, num_classes = set_dataset(data_name, niid, SEED)


def add_laplace_noise(data, epsilon, seed=None):
    # Add Laplace noise to data
    #   epsilon  : privacy budget
    #   sensitivity   : sensitivity ( abs(v_max - v_min) )
    #   scale    : scale parameter = deltaF/epsilon
    #   default location = 0
    # Return a dataset with Laplace noise

    data = np.array(data)

    sensitivity = np.apply_along_axis(lambda x: np.abs(np.max(x) - np.min(x)), axis=0, arr=data)
    rng = np.random.default_rng(seed=seed)
    scale = sensitivity / epsilon
    noise = rng.laplace(scale=scale, size=data.shape)
    return data + noise


def find_nearest_centroid(data_points, centroids):
    # Calculate the distances between data_points and centroids
    distances = sp.spatial.distance.cdist(data_points, centroids)
    # Find the index of the nearest centroid for each data point
    nearest_centroid_indices = np.argmin(distances, axis=1)
    return nearest_centroid_indices


# dataset_utils.py --------------------------------
# https://github.com/TsingZ0/PFL-Non-IID/blob/master/dataset/utils/dataset_utils.py
batch_size = 10
train_size = 0.75  # merge original training set and test set, then split it manually.
least_samples = batch_size / (1 - train_size)  # least samples for each client


def separate_data(data, num_clients, num_classes, alpha=1.0, niid=False, balance=False, partition=None, class_per_client=2):

    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]

    dataset_content, dataset_label = data

    dataidx_map = {}

    if not niid:
        partition = 'pat'
        class_per_client = num_classes

    if partition == 'pat':
        idxs = np.array(range(len(dataset_label)))
        idx_for_each_class = []
        for i in range(num_classes):
            idx_for_each_class.append(idxs[dataset_label == i])

        class_num_per_client = [class_per_client for _ in range(num_clients)]
        for i in range(num_classes):
            selected_clients = []
            for client in range(num_clients):
                if class_num_per_client[client] > 0:
                    selected_clients.append(client)
                selected_clients = selected_clients[:int(num_clients / num_classes * class_per_client)]

            num_all_samples = len(idx_for_each_class[i])
            num_selected_clients = len(selected_clients)
            num_per = num_all_samples / num_selected_clients
            if balance:
                num_samples = [int(num_per) for _ in range(num_selected_clients - 1)]
            else:
                np.random.seed(0)
                num_samples = np.random.randint(max(num_per / 10, least_samples / num_classes), num_per, num_selected_clients - 1).tolist()
            num_samples.append(num_all_samples - sum(num_samples))

            idx = 0
            for client, num_sample in zip(selected_clients, num_samples):
                if client not in dataidx_map.keys():
                    dataidx_map[client] = idx_for_each_class[i][idx:idx + num_sample]
                else:
                    dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx + num_sample],
                                                    axis=0)
                idx += num_sample
                class_num_per_client[client] -= 1

    elif partition == "dir":
        # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
        min_size = 0
        K = num_classes
        N = len(dataset_label)
        np.random.seed(0)
        while min_size < least_samples:
            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(dataset_label == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(num_clients):
            dataidx_map[j] = idx_batch[j]
    else:
        raise NotImplementedError

    # assign data
    for client in range(num_clients):
        idxs = dataidx_map[client]
        X[client] = dataset_content[idxs]
        y[client] = dataset_label[idxs]

        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client] == i))))

    del data
    # gc.collect()

    # for client in range(num_clients):
    #     print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
    #     print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
    #     print("-" * 50)

    return X, y, statistic


# HAR_utils.py --------------------------------
# https://github.com/TsingZ0/PFL-Non-IID/blob/master/dataset/utils/HAR_utils.py
train_size = 0.75


# This is for parsing the X data, you can ignore it if you do not need preprocessing
def format_data_x(datafile):
    x_data = None
    for item in datafile:
        item_data = np.loadtxt(item, dtype=np.float32)
        if x_data is None:
            x_data = np.zeros((len(item_data), 1))
        x_data = np.hstack((x_data, item_data))
    x_data = x_data[:, 1:]
    # print(x_data.shape)
    X = None
    for i in range(len(x_data)):
        row = np.asarray(x_data[i, :])
        row = row.reshape(9, 128).T
        if X is None:
            X = np.zeros((len(x_data), 128, 9))
        X[i] = row
    # print(X.shape)
    return X


# This is for parsing the Y data, you can ignore it if you do not need preprocessing
def format_data_y(datafile):
    return np.loadtxt(datafile, dtype=np.int32) - 1


def read_ids(datafile):
    return np.loadtxt(datafile, dtype=np.int32)


# /utils -----------------------------------
# https://github.com/TsingZ0/PFL-Non-IID/tree/master/dataset
# Allocate data to users
def generate_mnist(dir_path, num_clients, num_classes, alpha, niid, balance, partition):
    # Get MNIST data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    trainset = torchvision.datasets.MNIST(
        root=dir_path + "rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(
        root=dir_path + "rawdata", train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    data = np.array([np.reshape(dataset_image[s], (np.product(dataset_image[s].shape),)) for s in range(dataset_image.shape[0])])
    label = dataset_label

    return data, label


# Allocate data to users
def generate_fmnist(dir_path, num_clients, num_classes, alpha, niid, balance, partition):
    # Get FashionMNIST data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    trainset = torchvision.datasets.FashionMNIST(
        root=dir_path + "rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.FashionMNIST(
        root=dir_path + "rawdata", train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    data = np.array([np.reshape(dataset_image[s], (np.product(dataset_image[s].shape),)) for s in range(dataset_image.shape[0])])
    label = dataset_label

    return data, label


def generate_har(dir_path, num_clients, num_classes, alpha, niid, balance, partition):
    tmp_data, tmp_target = load_data_har(dir_path + 'rawdata/')

    data = np.concatenate(np.concatenate(tmp_data, axis=1), axis=1)
    target = tmp_target

    return data, target


def load_data_har(data_folder):
    str_folder = data_folder + 'UCI HAR Dataset/'
    INPUT_SIGNAL_TYPES = [
        "body_acc_x_",
        "body_acc_y_",
        "body_acc_z_",
        "body_gyro_x_",
        "body_gyro_y_",
        "body_gyro_z_",
        "total_acc_x_",
        "total_acc_y_",
        "total_acc_z_"
    ]

    str_train_files = [str_folder + 'train/' + 'Inertial Signals/' + item + 'train.txt' for item in
                       INPUT_SIGNAL_TYPES]
    str_test_files = [str_folder + 'test/' + 'Inertial Signals/' +
                      item + 'test.txt' for item in INPUT_SIGNAL_TYPES]
    str_train_y = str_folder + 'train/y_train.txt'
    str_test_y = str_folder + 'test/y_test.txt'
    str_train_id = str_folder + 'train/subject_train.txt'
    str_test_id = str_folder + 'test/subject_test.txt'

    X_train = format_data_x(str_train_files)
    X_test = format_data_x(str_test_files)
    Y_train = format_data_y(str_train_y)
    Y_test = format_data_y(str_test_y)
    id_train = read_ids(str_train_id)
    id_test = read_ids(str_test_id)

    X_train, X_test = X_train.reshape((-1, 9, 1, 128)), X_test.reshape((-1, 9, 1, 128))

    X = np.concatenate((X_train, X_test), axis=0)
    Y = np.concatenate((Y_train, Y_test), axis=0)
    ID = np.concatenate((id_train, id_test), axis=0)

    XX, YY = [], []
    for i in np.unique(ID):
        idx = ID == i
        XX.append(X[idx])
        YY.append(Y[idx])

    return X, Y
    # return XX, YY


# utils.py --------------------------------
# https://github.com/thupchnsky/mufc/blob/main/utils.py
def load_dataset(filepath):
    """
        Return:
            dataset: dict
    """
    with open(filepath, 'rb') as fr:
        dataset = pickle.load(fr)
    return dataset


def sample_points_in_bin(bin_mid, total_points, quant_eps):
    """
        Input:
            bin_mid: numpy.array (d,)
            total_points: points needed to be generated
            quant_eps: quantization region length
    """
    sampled_shifts = np.random.uniform(-quant_eps / 2.0,
                                       quant_eps / 2.0,
                                       size=[total_points, bin_mid.size])
    sampled_points = sampled_shifts + bin_mid
    return sampled_points

def clustering_loss(data, centroids):
    """
        Computes the clustering loss on a dataset given a fixed set of centroids
        Input:
            centroids: numpy.array (k,d)
            data: numpy.array (n,d)
    """
    loss = 0.0
    for i_data in range(data.shape[0]):
        d = np.linalg.norm(data[i_data, :] - centroids, axis=1)
        loss += np.min(d)**2
    return loss


def induced_loss(data, centroids, assignments):
    """
        Compute the loss based on the induced clustering results
        Input:
            centroids: numpy.array (k,d)
            data: numpy.array (n,d)
            assignments: numpy.array (n,). Values are between [0,k-1]
    """
    loss = 0.0
    for i_data in range(data.shape[0]):
        d = np.linalg.norm(data[i_data, :] - centroids[assignments[i_data], :])
        loss += d**2
    return loss


def induced_loss_return_max(data, centroids, assignments):
    """
        Compute the loss based on the induced clustering results
        Input:
            centroids: numpy.array (k,d)
            data: numpy.array (n,d)
            assignments: numpy.array (n,). Values are between [0,k-1]
    """
    loss = 0.0
    argmax_idx = -1
    max_loss = -1
    for i_data in range(data.shape[0]):
        d = np.linalg.norm(data[i_data, :] - centroids[assignments[i_data], :])
        loss += d**2
        if d > max_loss:
            max_loss = d
            argmax_idx = i_data
    return loss, argmax_idx

def split_data(data_combined,
               num_clusters,
               num_clients=None,
               split='iid',
               k_prime=None):
    json_data = {}
    # K-means optimal loss
    clf = KMeans(n_clusters=num_clusters).fit(data_combined)
    kmeans_loss = clf.inertia_
    kmeans_label = clf.labels_
    json_data['kmeans_loss'] = kmeans_loss

    if num_clients is None:
        num_clients = int(
            data_combined.shape[0] /
            100)  # make sure each client does not have too much data

    # initialize for each client
    for i in range(num_clients):
        json_data['client_' + str(i)] = []

    # iid split
    if split == 'iid':
        for k in range(num_clusters):
            data_cluster = data_combined[kmeans_label == k, :]
            size_per_client = math.floor(data_cluster.shape[0] / num_clients)
            for i in range(num_clients - 1):
                json_data['client_' + str(i)].append(
                    data_cluster[i * size_per_client:(i + 1) *
                                                     size_per_client, :])
            # fill the rest into the last client
            json_data['client_' + str(num_clients - 1)].append(
                data_cluster[(num_clients - 1) * size_per_client:, :])

        tmp_count = 0
        # concatenate the data for all clients
        for i in range(num_clients):
            json_data['client_' + str(i)] = np.concatenate(
                json_data['client_' + str(i)], axis=0)
            tmp_count += json_data['client_' + str(i)].shape[0]
        # have a final check on the sizes
        assert tmp_count == data_combined.shape[
            0], "Error: data size does not match"
    # non-iid split
    elif split == 'non-iid':
        if k_prime is None:
            k_prime = int(num_clusters / 2)
        assert k_prime <= num_clusters, "Error: not valid k_prime"
        # first get data for each cluster
        data_by_cluster = {}
        data_by_cluster_used = [0] * num_clusters
        size_per_client = int(data_combined.shape[0] / num_clients)
        for k in range(num_clusters):
            data_by_cluster[k] = data_combined[kmeans_label == k, :]

        valid_cluster_idx = [k for k in range(num_clusters)]
        # first fill in the data for first n-1 clients
        for i in range(num_clients - 1):
            tmp_client_data = []
            tmp_client_size = 0
            tmp_client_clusters = np.random.choice(valid_cluster_idx,
                                                   min(k_prime,
                                                       len(valid_cluster_idx)),
                                                   replace=False)
            for tmp_client_cluster_idx in tmp_client_clusters:
                # some intermediate variables
                tmp_1 = data_by_cluster_used[tmp_client_cluster_idx]
                tmp_2 = data_by_cluster[tmp_client_cluster_idx].shape[0]
                if tmp_client_size < size_per_client and tmp_1 < tmp_2:
                    tmp_count = min([
                        np.random.randint(
                            int(size_per_client / k_prime) - 1,
                            size_per_client),
                        size_per_client - tmp_client_size, tmp_2 - tmp_1
                    ])
                    tmp_client_data.append(
                        data_by_cluster[tmp_client_cluster_idx][tmp_1:tmp_1 +
                                                                      tmp_count, :])
                    # update each value
                    data_by_cluster_used[tmp_client_cluster_idx] += tmp_count
                    if data_by_cluster_used[tmp_client_cluster_idx] == tmp_2:
                        valid_cluster_idx.remove(
                            tmp_client_cluster_idx
                        )  # will not selected by future clients
                    tmp_client_size += tmp_count
                    if tmp_client_size == size_per_client:
                        break
            json_data['client_' + str(i)] = np.concatenate(tmp_client_data,
                                                           axis=0)
        # leave all other data points to the last client
        cluster_size_last_client = 0
        tmp_client_data = []
        for k in range(num_clusters):
            if data_by_cluster_used[k] < data_by_cluster[k].shape[0]:
                tmp_client_data.append(
                    data_by_cluster[k][data_by_cluster_used[k]:, :])
                cluster_size_last_client += 1
        assert cluster_size_last_client <= k_prime, "Error: k_prime is violated"
        json_data['client_' + str(num_clients - 1)] = np.concatenate(
            tmp_client_data, axis=0)
        # have a final check on the sizes
        tmp_count = 0
        for i in range(num_clients):
            tmp_count += json_data['client_' + str(i)].shape[0]
        assert tmp_count == data_combined.shape[
            0], "Error: data size does not match"
    else:
        raise NotImplementedError

    return json_data
