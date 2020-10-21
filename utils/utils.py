import os
import json

import cvxpy as cp
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from utils.metrics import accuracy, binary_accuracy

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from models.synthetic.linear import LinearModel
from models.femnist.cnn import FemnistCNN
from models.sent140.lstm import LSTMSentiment
from models.shakespeare.gru import NextCharDecoder
from models.inaturalist.resnet import INaturalistCNN

from loaders.synthetic import get_iterator_synthetic
from loaders.sent140 import get_iterator_sent140
from loaders.femnist import get_iterator_femnist
from loaders.shakespeare import get_iterator_shakespeare
from loaders.inaturalist import get_iterator_inaturalist


EXTENSIONS = {"synthetic": ".json",
              "sent140": ".json",
              "femnist": ".pkl",
              "shakespeare": ".txt",
              "inaturalist": ".pkl"}

# Model size in bit
MODEL_SIZE_DICT = {"synthetic": 4354,
                   "shakespeare": 3385747,
                   "femnist": 4843243,
                   "sent140": 19269416,
                   "inaturalist": 44961717}

# Model computation time in ms
COMPUTATION_TIME_DICT = {"synthetic": 1.5,
                         "shakespeare": 389.6,
                         "femnist": 4.6,
                         "sent140": 9.8,
                         "inaturalist": 25.4}

# Tags list
TAGS = ["Train/Loss", "Train/Acc", "Test/Loss", "Test/Acc", "Consensus"]


def args_to_string(args):
    """
    Transform experiment's arguments into a string
    :param args:
    :return: string
    """
    args_string = ""

    args_to_show = ["experiment", "network_name", "fit_by_epoch", "bz",
                    "lr", "decay", "local_steps", "communication_budget"]
    for arg in args_to_show:
        args_string += arg
        args_string += "_" + str(getattr(args, arg)) + "_"

    return args_string[:-1]


def get_optimal_mixing_matrix(adjacency_matrix, method="FDLA"):
    """

    :param adjacency_matrix: np.array()
    :param method:method to construct the mixing matrix weights;
     possible are:
      FMMC (Fast Mixin Markov Chain), see https://web.stanford.edu/~boyd/papers/pdf/fmmc.pdf
      FDLA (Fast Distributed Linear Averaging), https://web.stanford.edu/~boyd/papers/pdf/fastavg.pdf
    :return: optimal mixing matrix as np.array()
    """
    network_mask = 1 - adjacency_matrix
    N = adjacency_matrix.shape[0]

    s = cp.Variable()
    W = cp.Variable((N, N))
    objective = cp.Minimize(s)

    if method == "FDLA":
        constraints = [W == W.T,
                       W @ np.ones((N, 1)) == np.ones((N, 1)),
                       cp.multiply(W, network_mask) == np.zeros((N, N)),
                       -s * np.eye(N) << W - (np.ones((N, 1)) @ np.ones((N, 1)).T) / N,
                       W - (np.ones((N, 1)) @ np.ones((N, 1)).T) / N << s * np.eye(N)
                       ]
    elif method == "FMMC":
        constraints = [W == W.T,
                       W @ np.ones((N, 1)) == np.ones((N, 1)),
                       cp.multiply(W, network_mask) == np.zeros((N, N)),
                       -s * np.eye(N) << W - (np.ones((N, 1)) @ np.ones((N, 1)).T) / N,
                       W - (np.ones((N, 1)) @ np.ones((N, 1)).T) / N << s * np.eye(N),
                       np.zeros((N, N)) <= W
                       ]
    else:
        raise NotImplementedError

    prob = cp.Problem(objective, constraints)
    prob.solve()

    mixing_matrix = W.value

    mixing_matrix *= adjacency_matrix

    if method == "FMMC":
        mixing_matrix = np.multiply(mixing_matrix, mixing_matrix >= 0)

    # Force symmetry
    for i in range(N):
        if np.abs(np.sum(mixing_matrix[i, i:])) >= 1e-20:
            mixing_matrix[i, i:] *= (1 - np.sum(mixing_matrix[i, :(i)])) / np.sum(mixing_matrix[i, i:])
            mixing_matrix[i:, i] = mixing_matrix[i, i:]

    return mixing_matrix


def get_network(network_name, architecture):
    """
    load network  and generate mixing matrix,
    :param network_name: (str) should present in "graph_utils/data"
    :param architecture: possible are: "ring", "complete", "mst", "centralized" and "no_communication"
    :return: nx.DiGraph if architecture is "ring" and nx.DiGraph otherwise
    """
    if (architecture == "no_communication") or (architecture == "matcha") or (architecture == "matcha+"):
        path = os.path.join("graph_utils", "results", network_name, "original.gml")
    else:
        path = os.path.join("graph_utils", "results", network_name, "{}.gml".format(architecture))

    if architecture == "ring":
        network = nx.read_gml(path)
        mixing_matrix = nx.adjacency_matrix(network, weight=None).todense()

        mixing_matrix = mixing_matrix.astype(np.float64)
        mixing_matrix += np.eye(mixing_matrix.shape[0])
        mixing_matrix *= 0.5

        return nx.from_numpy_matrix(mixing_matrix, create_using=nx.DiGraph())

    elif architecture == "mct_plus":
        network = nx.read_gml(path)
        mixing_matrix = nx.adjacency_matrix(network, weight=None).todense()

        n = network.number_of_nodes()
        mixing_matrix = mixing_matrix.astype(np.float64)
        mixing_matrix += np.eye(n)
        mixing_matrix = mixing_matrix / mixing_matrix.sum(axis=0)

        return nx.from_numpy_matrix(mixing_matrix, create_using=nx.DiGraph())

    elif architecture == "complete" or architecture == "matcha":
        network = nx.read_gml(path).to_undirected()

        n = network.number_of_nodes()
        mixing_matrix = np.ones((n, n)) / n

        return nx.from_numpy_matrix(mixing_matrix)

    elif architecture == "matcha+":
        network = nx.read_gml(path).to_undirected()

        return network

    elif architecture == "matcha+mst":
        path = os.path.join("graph_utils", "results", network_name, "mst.gml")
        return nx.read_gml(path).to_undirected()

    elif architecture == "matcha+ring":
        path = os.path.join("graph_utils", "results", network_name, "ring.gml")
        return nx.read_gml(path).to_undirected()

    elif architecture == "matcha+delta_mbst":
        path = os.path.join("graph_utils", "results", network_name, "mct_congest.gml")
        return nx.read_gml(path).to_undirected()

    elif architecture == "no_communication":
        network = nx.read_gml(path)
        mixing_matrix = nx.adjacency_matrix(network, weight=None).todense()

        mixing_matrix = np.eye(network.number_of_nodes())

        return nx.from_numpy_matrix(mixing_matrix)
    else:
        network = nx.read_gml(path)
        mixing_matrix = nx.adjacency_matrix(network, weight=None).todense()

        adjacency_matrix = nx.adjacency_matrix(network, weight=None).todense()
        adjacency_matrix += np.eye(mixing_matrix.shape[0], dtype=np.int64)
        mixing_matrix = get_optimal_mixing_matrix(adjacency_matrix, method="FDLA")

        return nx.from_numpy_matrix(mixing_matrix)


def get_model(name, device, iterator, epoch_size, optimizer_name="adam", lr_scheduler="custom",
              initial_lr=1e-3, seed=1234, coeff=1):
    """
    Load Model object corresponding to the experiment
    :param name: experiment name; possible are: synthetic, shakespeare, sent140, inaturalist, femnist
    :param device:
    :param iterator: torch.utils.DataLoader object representing an iterator of dataset corresponding to name
    :param epoch_size:
    :param optimizer_name: optimizer name, for now only "adam" is possible
    :param lr_scheduler:
    :param initial_lr:
    :param seed:
    :param coeff
    :return: Model object
    """
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if name == "synthetic":
        input_dimension = iterator.dataset.dimension
        num_classes = iterator.dataset.num_classes
        if num_classes == 1:
            metric = binary_accuracy
            criterion = nn.BCEWithLogitsLoss()
        else:
            metric = accuracy
            criterion = nn.CrossEntropyLoss()
        return LinearModel(criterion, metric, device, input_dimension, num_classes,
                           optimizer_name, lr_scheduler, initial_lr, epoch_size)

    elif name == "femnist":
        criterion = nn.CrossEntropyLoss()
        metric = accuracy
        return FemnistCNN(criterion, metric, device, optimizer_name, lr_scheduler, initial_lr, epoch_size)

    elif name == "sent140":
        criterion = nn.BCEWithLogitsLoss()
        metric = binary_accuracy
        return LSTMSentiment(iterator, criterion, metric, device, optimizer_name, lr_scheduler, epoch_size)

    elif name == "shakespeare":
        criterion = nn.CrossEntropyLoss()
        metric = accuracy
        return NextCharDecoder(criterion, metric, device, optimizer_name, lr_scheduler, initial_lr, epoch_size)

    elif name == "inaturalist":
        criterion = nn.CrossEntropyLoss()
        metric = accuracy
        return INaturalistCNN(criterion, metric, device, optimizer_name, lr_scheduler, initial_lr, epoch_size,
                              coeff=coeff)

    else:
        raise NotImplementedError


def get_iterator(name, path, device, batch_size):
    if name == "synthetic":
        return get_iterator_synthetic(path, device, batch_size=batch_size)
    elif name == "sent140":
        all_data_path = os.path.join("data", "sent140", "train", "train.json")
        return get_iterator_sent140(path, all_data_path, device, batch_size)
    elif name == "femnist":
        return get_iterator_femnist(path, device, batch_size)
    elif name == "shakespeare":
        return get_iterator_shakespeare(path, device, batch_size)
    elif name == "inaturalist":
        return get_iterator_inaturalist(path, device, batch_size)
    else:
        raise NotImplementedError


def loggs_to_json(loggs_dir_path):
    """
    Write the results from logs folder as .json format
    :param loggs_dir_path: path to loggs folder
    """

    os.makedirs(os.path.join("results", "json"), exist_ok=True)

    all_results = {"Train/Loss": dict(), "Train/Acc": dict(), "Test/Loss": dict(),
                   "Test/Acc": dict(), "Consensus": dict(), "Round": dict()}

    for dname in os.listdir(loggs_dir_path):
        ea = EventAccumulator(os.path.join(loggs_dir_path, dname)).Reload()
        tags = ea.Tags()['scalars']

        for tag in tags:

            tag_values = []
            steps = []

            for event in ea.Scalars(tag):
                tag_values.append(event.value)
                steps.append(event.step)

            all_results[tag][dname] = tag_values
            all_results["Round"][dname] = steps

    json_path = os.path.join("results", "json", "{}.json".format(os.path.split(loggs_dir_path)[1]))
    with open(json_path, "w") as f:
        json.dump(all_results, f)


def make_plots(args, cycle_time_dict, tag_dict, labels_dict, path_dict, mode=0):
    os.makedirs(os.path.join("results", "plots", args.experiment), exist_ok=True)

    loggs_dir_path = os.path.join("loggs", args_to_string(args))
    path_to_json = os.path.join("results", "json", "{}.json".format(os.path.split(loggs_dir_path)[1]))
    with open(path_to_json, "r") as f:
        data = json.load(f)

    x_lim = np.inf
    for idx, tag in enumerate(TAGS):
        plt.figure(figsize=(12, 10))
        for architecture in ["centralized", "matcha", "mst", "mct_congest", "ring"]:
            values = data[tag][architecture]
            rounds = data["Round"][architecture]

            min_len = min(len(values), len(rounds))

            if rounds[-1] * cycle_time_dict[args.experiment][architecture] < x_lim:
                x_lim = rounds[-1] * cycle_time_dict[args.experiment][architecture]

            if mode == 0:
                plt.plot(cycle_time_dict[args.experiment][architecture] * np.array(rounds) / 1000,
                         values[:min_len], label=labels_dict[architecture],
                         linewidth=5.0)
                plt.grid(True, linewidth=2)
                plt.xlim(0, x_lim / 1000)
                plt.ylabel("{}".format(tag_dict[tag]), fontsize=50)
                plt.xlabel("Time (s)", fontsize=50)
                plt.tick_params(axis='both', labelsize=40)
                plt.tick_params(axis='x')

                if args.experiment == "shakespeare":
                    plt.legend(fontsize=35)

            else:
                min_len = min(len(values), len(rounds))
                plt.plot(rounds[:min_len],
                         values[:min_len], label=labels_dict[architecture],
                         linewidth=5.0)

                plt.ylabel("{}".format(tag_dict[tag]), fontsize=50)
                plt.xlabel("Rounds", fontsize=50)
                plt.tick_params(axis='both', labelsize=40)
                plt.grid(True, linewidth=2)

                if args.experiment == "shakespeare":
                    plt.xlim(0, 1650)
                    plt.legend(fontsize=35)

                if args.experiment == "sent140":
                    plt.xlim(0, 22_000)

        if mode == 0:
            fig_path = os.path.join("results",
                                    "plots",
                                    args.experiment,
                                    "{}_vs_time.png".format(path_dict[tag])
                                    )

        else:
            fig_path = os.path.join("results",
                                    "plots",
                                    args.experiment,
                                    "{}_vs_iteration.png".format(path_dict[tag])
                                    )

        plt.savefig(fig_path, bbox_inches='tight')
