import os
from abc import ABC, abstractmethod

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from graph_utils.utils.matcha import RandomTopologyGenerator
from utils.utils import get_network, get_iterator, get_model, args_to_string


EXTENSIONS = {"synthetic": ".json", "sent140": ".json", "femnist": ".pkl", "shakespeare": ".txt"}


class Manager(ABC):
    def __init__(self, args):
        self.device = args.device
        self.batch_size = args.bz
        self.network = get_network(args.network_name, args.architecture)
        self.world_size = self.network.number_of_nodes() + 1  # we add node representing the network manager
        self.log_freq = args.log_freq

        # create logger
        logger_path = os.path.join("loggs", args_to_string(args), args.architecture)
        self.logger = SummaryWriter(logger_path)

        self.round_idx = 0  # index of the current communication round

        self.train_dir = os.path.join("data", args.experiment, "train")
        self.test_dir = os.path.join("data", args.experiment, "test")

        self.train_path = os.path.join(self.train_dir, "train" + EXTENSIONS[args.experiment])
        self.test_path = os.path.join(self.test_dir, "test" + EXTENSIONS[args.experiment])

        self.train_iterator = get_iterator(args.experiment, self.train_path, self.device, self.batch_size)
        self.test_iterator = get_iterator(args.experiment, self.test_path, self.device, self.batch_size)

        self.gather_list = [get_model(args.experiment, self.device, self.train_iterator)
                            for _ in range(self.world_size)]

        self.scatter_list = [get_model(args.experiment, self.device, self.train_iterator)
                             for _ in range(self.world_size)]

        # print initial logs
        self.write_logs()

    def communicate(self):
        for ii, param in enumerate(self.gather_list[-1].net.parameters()):
            param_list = [list(self.gather_list[idx].net.parameters())[ii].data
                          for idx in range(self.world_size)]

            dist.gather(tensor=param.data, dst=self.world_size - 1, gather_list=param_list)

        self.mix()

        if (self.round_idx - 1) % self.log_freq == 0:
            self.write_logs()

        for ii, param in enumerate(self.scatter_list[-1].net.parameters()):
            param_list = [list(self.scatter_list[idx].net.parameters())[ii].data
                          for idx in range(self.world_size)]

            dist.scatter(tensor=param.data, src=self.world_size - 1, scatter_list=param_list)

    @abstractmethod
    def mix(self):
        pass

    def write_logs(self):
        """
        write train/test loss, train/tet accuracy for average model and local models
         and intra-workers parameters variance (consensus) adn save average model
        """
        train_loss, train_acc = self.scatter_list[-1].evaluate_iterator(self.train_iterator)
        test_loss, test_acc = self.scatter_list[-1].evaluate_iterator(self.train_iterator)

        self.logger.add_scalar("Train/Loss", train_loss, self.round_idx)
        self.logger.add_scalar("Train/Acc", train_acc, self.round_idx)
        self.logger.add_scalar("Test/Loss", test_loss, self.round_idx)
        self.logger.add_scalar("Test/Acc", test_acc, self.round_idx)

        # write parameter variance
        average_parameter = self.scatter_list[-1].get_param_tensor()

        param_tensors_by_workers = torch.zeros((average_parameter.shape[0], self.world_size - 1))

        for ii, model in enumerate(self.scatter_list[:-1]):
            param_tensors_by_workers[:, ii] = model.get_param_tensor() - average_parameter

        consensus = (param_tensors_by_workers ** 2).sum()
        self.logger.add_scalar("Consensus", consensus, self.round_idx)

        print(f'\t Round: {self.round_idx} |Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')


class Peer2PeerManager(Manager):
    def mix(self):
        for ii, model in enumerate(self.scatter_list):
            if ii == self.world_size - 1:
                for param_idx, param in enumerate(model.net.parameters()):
                    param.data.fill_(0.)
                    for local_model in self.scatter_list[:-1]:
                        param.data += (1 / (self.world_size - 1)) * list(local_model.net.parameters())[param_idx]
            else:
                for param_idx, param in enumerate(model.net.parameters()):
                    param.data.fill_(0.)
                    for neighbour in self.network.neighbors(ii):
                        coeff = self.network.get_edge_data(ii, neighbour)["weight"]
                        param.data += coeff * list(self.gather_list[neighbour].net.parameters())[param_idx]

        self.round_idx += 1


class MATCHAManager(Manager):
    def __init__(self, args):
        super(MATCHAManager, self).__init__(args)
        path_to_save_network = os.path.join("loggs", args_to_string(args), "matcha", "colored_network.gml")
        path_to_matching_history_file = os.path.join("loggs", args_to_string(args), "matcha", "matching_history.csv")
        self.topology_generator = RandomTopologyGenerator(self.network,
                                                          args.communication_budget,
                                                          network_save_path=path_to_save_network,
                                                          path_to_history_file=path_to_matching_history_file)

    def mix(self):
        # update topology
        self.topology_generator.step()

        for ii, model in enumerate(self.scatter_list):
            if ii == self.world_size - 1:
                for param_idx, param in enumerate(model.net.parameters()):
                    param.data.fill_(0.)
                    for local_model in self.scatter_list[:-1]:
                        param.data += (1 / (self.world_size - 1)) * list(local_model.net.parameters())[param_idx]
            else:
                for param_idx, param in enumerate(model.net.parameters()):
                    param.data.fill_(0.)
                    for neighbour in self.topology_generator.current_topology.neighbors(ii):
                        coeff = self.topology_generator.current_topology.get_edge_data(ii, neighbour)["weight"]
                        param.data += coeff * list(self.gather_list[neighbour].net.parameters())[param_idx]

        self.round_idx += 1


class CentralizedManager(Manager):
    def mix(self):
        for param_idx, param in enumerate(self.scatter_list[-1].net.parameters()):
            param.data.fill_(0.)
            for local_model in self.gather_list[:-1]:
                param.data += (1 / (self.world_size - 1)) * list(local_model.net.parameters())[param_idx]

        for ii, model in enumerate(self.scatter_list[:-1]):
            for param_idx, param in enumerate(model.net.parameters()):
                param.data = list(self.scatter_list[-1].net.parameters())[param_idx]

        self.round_idx += 1
