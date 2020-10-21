import os

import torch
import torch.distributed as dist

from utils.utils import get_network, get_iterator, get_model


EXTENSIONS = {"synthetic": ".json", "sent140": ".json", "femnist": ".pkl", "shakespeare": ".txt"}


class Worker(object):
    def __init__(self, args, rank):
        self.rank = rank
        self.local_steps = args.local_steps
        self.device = args.device
        self.num_gpu = torch.cuda.device_count()
        self.batch_size = args.bz
        self.network = get_network(args.network_name, args.architecture)
        self.world_size = self.network.number_of_nodes() + 1  # we add node representing the network manager
        self.fit_by_epoch = args.fit_by_epoch
        self.initial_lr = args.lr
        self.optimizer_name = args.optimizer
        self.lr_scheduler_name = args.decay

        if self.device == "cuda":
            if torch.cuda.is_available():
                print(f"{rank} get gpu {self.rank % self.num_gpu}")
                self.device = "cuda:"+str(self.rank % self.num_gpu)
            else:
                print("No GPU is available on the system")
                raise TypeError
        elif self.device != "cpu":
            print("Please choose device be either cuda or cpu")
            raise TypeError

        self.data_dir = os.path.join("data", args.experiment, "train")
        self.data_path = os.path.join(self.data_dir, str(rank) + EXTENSIONS[args.experiment])

        self.iterator = get_iterator(args.experiment, self.data_path, self.device, self.batch_size)

        self.model = get_model(args.experiment, self.device, self.iterator,
                               optimizer_name=self.optimizer_name, lr_scheduler=self.lr_scheduler_name,
                               initial_lr=self.initial_lr)

    def communicate(self):

        if self.fit_by_epoch:
            self.model.fit_iterator(train_iterator=self.iterator, n_epochs=self.local_steps)
        else:
            self.model.fit_batches(iterator=self.iterator, n_steps=self.local_steps)

        for ii, param in enumerate(self.model.net.parameters()):
            dist.gather(tensor=param.data, dst=self.world_size - 1)

        for ii, param in enumerate(self.model.net.parameters()):
            dist.scatter(tensor=param.data, src=self.world_size - 1)
