import os
from abc import ABC, abstractmethod

import networkx as nx
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.utils import get_network, get_iterator, get_model, args_to_string, EXTENSIONS
from graph_utils.utils.matcha import RandomTopologyGenerator
from graph_utils.utils.utils import generate_random_ring


class Network(ABC):
    def __init__(self, args):
        """
        Abstract class representing a network of worker collaborating to train a machine learning model,
        each worker has a local model and a local data iterator.
         Should implement `mix` to precise how the communication is done
        :param args: parameters defining the network
        """
        self.args = args
        self.device = args.device
        self.batch_size = args.bz
        self.network = get_network(args.network_name, args.architecture)
        self.n_workers = self.network.number_of_nodes()
        self.local_steps = args.local_steps
        self.log_freq = args.log_freq
        self.fit_by_epoch = args.fit_by_epoch
        self.initial_lr = args.lr
        self.optimizer_name = args.optimizer
        self.lr_scheduler_name = args.decay

        # create logger
        logger_path = os.path.join("loggs", args_to_string(args), args.architecture)
        os.makedirs(logger_path, exist_ok=True)
        self.logger = SummaryWriter(logger_path)

        self.round_idx = 0  # index of the current communication round

        # get data loaders
        if args.experiment == "inaturalist":
            self.train_dir = os.path.join("data", args.experiment, "train_{}".format(args.network_name))
            self.test_dir = os.path.join("data", args.experiment, "test_{}".format(args.network_name))
        else:
            self.train_dir = os.path.join("data", args.experiment, "train")
            self.test_dir = os.path.join("data", args.experiment, "test")

        self.train_path = os.path.join(self.train_dir, "train" + EXTENSIONS[args.experiment])
        self.test_path = os.path.join(self.test_dir, "test" + EXTENSIONS[args.experiment])

        self.train_iterator = get_iterator(args.experiment, self.train_path, self.device, self.batch_size)
        self.test_iterator = get_iterator(args.experiment, self.test_path, self.device, self.batch_size)

        self.workers_iterators = []
        self.local_function_weights = np.zeros(self.n_workers)
        train_data_size = 0
        for worker_id in range(self.n_workers):
            data_path = os.path.join(self.train_dir, str(worker_id) + EXTENSIONS[args.experiment])
            self.workers_iterators.append(get_iterator(args.experiment, data_path, self.device, self.batch_size))
            train_data_size += len(self.workers_iterators[-1])
            self.local_function_weights[worker_id] = len(self.workers_iterators[-1].dataset)

        self.epoch_size = int(train_data_size / self.n_workers)
        self.local_function_weights = self.local_function_weights / self.local_function_weights.sum()

        # create workers models
        if args.use_weighted_average:
            self.workers_models = [get_model(args.experiment, self.device, self.workers_iterators[w_i],
                                             optimizer_name=self.optimizer_name, lr_scheduler=self.lr_scheduler_name,
                                             initial_lr=self.initial_lr, epoch_size=self.epoch_size,
                                             coeff=self.local_function_weights[w_i])
                                   for w_i in range(self.n_workers)]
        else:
            self.workers_models = [get_model(args.experiment, self.device, self.workers_iterators[w_i],
                                             optimizer_name=self.optimizer_name, lr_scheduler=self.lr_scheduler_name,
                                             initial_lr=self.initial_lr, epoch_size=self.epoch_size)
                                   for w_i in range(self.n_workers)]

        # average model of all workers
        self.global_model = get_model(args.experiment,
                                      self.device,
                                      self.train_iterator,
                                      epoch_size=self.epoch_size)

        # write initial performance
        self.write_logs()

    @abstractmethod
    def mix(self):
        pass

    def write_logs(self):
        """
        write train/test loss, train/tet accuracy for average model and local models
         and intra-workers parameters variance (consensus) adn save average model
        """
        train_loss, train_acc = self.global_model.evaluate_iterator(self.train_iterator)
        test_loss, test_acc = self.global_model.evaluate_iterator(self.test_iterator)

        self.logger.add_scalar("Train/Loss", train_loss, self.round_idx)
        self.logger.add_scalar("Train/Acc", train_acc, self.round_idx)
        self.logger.add_scalar("Test/Loss", test_loss, self.round_idx)
        self.logger.add_scalar("Test/Acc", test_acc, self.round_idx)

        # write parameter variance
        average_parameter = self.global_model.get_param_tensor()

        param_tensors_by_workers = torch.zeros((average_parameter.shape[0], self.n_workers))

        for ii, model in enumerate(self.workers_models):
            param_tensors_by_workers[:, ii] = model.get_param_tensor() - average_parameter

        consensus = (param_tensors_by_workers ** 2).mean()
        self.logger.add_scalar("Consensus", consensus, self.round_idx)

        print(f'\t Round: {self.round_idx} |Train Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')


class RingNetwork(Network):
    def __init__(self, args):
        super(RingNetwork, self).__init__(args)
        self.p = args.random_ring_proba
        self.optimal_network = self.network.copy()

    def mix(self, write_results=True):
        """
        :param write_results:
        Mix local model parameters in a gossip fashion
        """
        # update the mixing matrix
        token = np.random.binomial(1, self.p)
        if token:
            generate_random_ring(list(self.network.nodes))
        else:
            self.network = self.optimal_network.copy()

        # update workers
        for worker_id, model in enumerate(self.workers_models):
            model.net.to(self.device)
            if self.fit_by_epoch:
                model.fit_iterator(train_iterator=self.workers_iterators[worker_id],
                                   n_epochs=self.local_steps, verbose=0)
            else:
                model.fit_batches(iterator=self.workers_iterators[worker_id], n_steps=self.local_steps)

        # write logs
        if ((self.round_idx - 1) % self.log_freq == 0) and write_results:
            for param_idx, param in enumerate(self.global_model.net.parameters()):
                param.data.fill_(0.)
                for worker_model in self.workers_models:
                    param.data += (1 / self.n_workers) * list(worker_model.net.parameters())[param_idx].data.clone()

            self.write_logs()

        # mix models
        for param_idx, param in enumerate(self.global_model.net.parameters()):
            temp_workers_param_list = [torch.zeros(param.shape).to(self.device) for _ in range(self.n_workers)]
            for worker_id, model in enumerate(self.workers_models):
                for neighbour in self.network.neighbors(worker_id):
                    coeff = self.network.get_edge_data(worker_id, neighbour)["weight"]
                    temp_workers_param_list[worker_id] += \
                        coeff * list(self.workers_models[neighbour].net.parameters())[param_idx].data.clone()

            for worker_id, model in enumerate(self.workers_models):
                for param_idx_, param_ in enumerate(model.net.parameters()):
                    if param_idx_ == param_idx:
                        param_.data = temp_workers_param_list[worker_id].clone()

        self.round_idx += 1


class CentralizedNetwork(Network):
    def mix(self, write_results=True):
        """
        :param write_results:
        All the local models are averaged, and the average model is re-assigned to each work
        """
        for worker_id, model in enumerate(self.workers_models):
            model.net.to(self.device)
            if self.fit_by_epoch:
                model.fit_iterator(train_iterator=self.workers_iterators[worker_id],
                                   n_epochs=self.local_steps, verbose=0)
            else:
                model.fit_batches(iterator=self.workers_iterators[worker_id], n_steps=self.local_steps)

        for param_idx, param in enumerate(self.global_model.net.parameters()):
            param.data.fill_(0.)
            for worker_model in self.workers_models:
                param.data += (1 / self.n_workers) * list(worker_model.net.parameters())[param_idx].data.clone()

        for ii, model in enumerate(self.workers_models):
            for param_idx, param in enumerate(model.net.parameters()):
                param.data = list(self.global_model.net.parameters())[param_idx].data.clone()

        self.round_idx += 1

        if ((self.round_idx - 1) % self.log_freq == 0) and write_results:
            self.write_logs()


class Peer2PeerNetwork(Network):
    def mix(self, write_results=True):
        """
        :param write_results:
        Mix local model parameters in a gossip fashion
        """
        # update workers
        for worker_id, model in enumerate(self.workers_models):
            model.net.to(self.device)
            if self.fit_by_epoch:
                model.fit_iterator(train_iterator=self.workers_iterators[worker_id],
                                   n_epochs=self.local_steps, verbose=0)
            else:
                model.fit_batches(iterator=self.workers_iterators[worker_id], n_steps=self.local_steps)

        # write logs
        if ((self.round_idx - 1) % self.log_freq == 0) and write_results:
            for param_idx, param in enumerate(self.global_model.net.parameters()):
                param.data.fill_(0.)
                for worker_model in self.workers_models:
                    param.data += (1 / self.n_workers) * list(worker_model.net.parameters())[param_idx].data.clone()

            self.write_logs()

        # mix models
        for param_idx, param in enumerate(self.global_model.net.parameters()):
            temp_workers_param_list = [torch.zeros(param.shape).to(self.device) for _ in range(self.n_workers)]
            for worker_id, model in enumerate(self.workers_models):
                for neighbour in self.network.neighbors(worker_id):
                    coeff = self.network.get_edge_data(worker_id, neighbour)["weight"]
                    temp_workers_param_list[worker_id] += \
                        coeff * list(self.workers_models[neighbour].net.parameters())[param_idx].data.clone()

            for worker_id, model in enumerate(self.workers_models):
                for param_idx_, param_ in enumerate(model.net.parameters()):
                    if param_idx_ == param_idx:
                        param_.data = temp_workers_param_list[worker_id].clone()

        self.round_idx += 1


class MATCHANetwork(Network):
    def __init__(self, args):
        super(MATCHANetwork, self).__init__(args)
        path_to_save_network =\
            os.path.join("loggs", args_to_string(args), args.architecture, "colored_network.gml")

        path_to_matching_history_file =\
            os.path.join("loggs", args_to_string(args), args.architecture, "matching_history.csv")

        self.topology_generator = RandomTopologyGenerator(self.network,
                                                          args.communication_budget,
                                                          network_save_path=path_to_save_network,
                                                          path_to_history_file=path_to_matching_history_file)

    def mix(self, write_results=True):
        """
        :param write_results:
        Mix local model parameters in a gossip fashion
        """
        # update topology
        self.topology_generator.step()
        current_topology = self.topology_generator.current_topology

        # update workers
        for worker_id, model in enumerate(self.workers_models):
            model.net.to(self.device)
            if self.fit_by_epoch:
                model.fit_iterator(train_iterator=self.workers_iterators[worker_id],
                                   n_epochs=self.local_steps, verbose=0)
            else:
                model.fit_batches(iterator=self.workers_iterators[worker_id], n_steps=self.local_steps)

        # write logs
        if ((self.round_idx - 1) % self.log_freq == 0) and write_results:
            for param_idx, param in enumerate(self.global_model.net.parameters()):
                param.data.fill_(0.)
                for worker_model in self.workers_models:
                    param.data += (1 / self.n_workers) * list(worker_model.net.parameters())[param_idx].data.clone()

            self.write_logs()

        # mix models
        for param_idx, param in enumerate(self.global_model.net.parameters()):
            temp_workers_param_list = [torch.zeros(param.shape).to(self.device) for _ in range(self.n_workers)]
            for worker_id, model in enumerate(self.workers_models):
                for neighbour in current_topology.neighbors(worker_id):
                    coeff = current_topology.get_edge_data(worker_id, neighbour)["weight"]
                    temp_workers_param_list[worker_id] += \
                        coeff * list(self.workers_models[neighbour].net.parameters())[param_idx].data.clone()

            for worker_id, model in enumerate(self.workers_models):
                for param_idx_, param_ in enumerate(model.net.parameters()):
                    if param_idx_ == param_idx:
                        param_.data = temp_workers_param_list[worker_id].clone()

        self.round_idx += 1
