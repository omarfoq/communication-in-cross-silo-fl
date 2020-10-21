import os
from torch.multiprocessing import Process
import torch.distributed as dist
import torch

from utils.args import parse_args
from utils.utils import loggs_to_json, args_to_string
from communication_module.worker import Worker
from communication_module.manager import Peer2PeerManager, CentralizedManager
from communication import CentralizedNetwork, Peer2PeerNetwork, MATCHANetwork, RingNetwork


def run(rank, size, arguments):
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if rank == size - 1:
        if arguments.architecture == "centralized":
            node = CentralizedManager(arguments)
        else:
            node = Peer2PeerManager(arguments)
    else:
        node = Worker(arguments, rank)

    for _ in range(arguments.n_rounds):
        node.communicate()


def init_process(rank, size, arguments, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, arguments)


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse_args()

    if args.parallel:
        print("Run experiment in parallel settings using torch.dist..")

        processes = []
        world_size = args.num_workers + 1  # We add an extra node that plays the role of network manager
        for rank_ in range(world_size):
            p = Process(target=init_process, args=(rank_, world_size, args, run))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    else:
        print("Run experiment in sequential setting..")

        if args.architecture == "centralized":
            network = CentralizedNetwork(args)
        elif args.architecture == "matcha" or args.architecture == "matcha+" or\
                args.architecture == "matcha+mst" or args.architecture == "matcha+ring" or\
                args.architecture == "matcha+delta_mbst":
            network = MATCHANetwork(args)
        elif args.architecture == "dynamic_ring":
            network = RingNetwork(args)
        else:
            network = Peer2PeerNetwork(args)

        for k in range(args.n_rounds):
            network.mix()

        network.write_logs()

    loggs_dir = os.path.join("loggs", args_to_string(args))
    loggs_to_json(loggs_dir)
