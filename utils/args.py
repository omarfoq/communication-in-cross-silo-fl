import argparse
from utils.utils import get_network


def parse_args(args_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'experiment',
        help='name of experiment',
        type=str)
    parser.add_argument(
        "--use_weighted_average",
        help="if used the weighted average will be optimized, otherwise the average is optimized,"
             " i,e, all the local functions are treated the same.",
        action='store_true'
    )
    parser.add_argument(
        '--network_name',
        help='name of the network;',
        type=str
    )
    parser.add_argument(
        '--architecture',
        help='architecture to use, possible: complete, centralized, ring, mst, original and matcha;',
        default='original'
    )
    parser.add_argument(
        '--communication_budget',
        type=float,
        help='used to fix communication budget when architecture is matcha;',
        default=0.5
    )
    parser.add_argument(
        "--random_ring_proba",
        type=float,
        help="the probability of using a random ring at each step; only used if architecture is ring",
        default=0.5
    )
    parser.add_argument(
        '--parallel',
        help='if chosen the training well be run in parallel,'
             'otherwise the training will be run in a sequential fashion;',
        action='store_true'
    )
    parser.add_argument(
        '--fit_by_epoch',
        help='if chosen each local step corresponds to one epoch,'
             ' otherwise each local step corresponds to one gradient step',
        action='store_true'
    )
    parser.add_argument(
        '--n_rounds',
        help='number of communication rounds;',
        type=int,
        default=1
    )
    parser.add_argument(
        '--bz',
        help='batch_size;',
        type=int,
        default=1
    )
    parser.add_argument(
        '--local_steps',
        help='number of local steps before communication;',
        type=int,
        default=1
    )
    parser.add_argument(
        '--log_freq',
        help='number of local steps before communication;',
        type=int,
        default=1
    )
    parser.add_argument(
        '--device',
        help='device to use, either cpu or gpu;',
        type=str,
        default="cpu"
    )
    parser.add_argument(
        '--optimizer',
        help='optimizer to be used for the training;',
        type=str,
        default="adam"
    )
    parser.add_argument(
        "--lr",
        type=float,
        help='learning rate',
        default=1e-3
    )
    parser.add_argument(
        "--decay",
        help='learning rate decay scheme to be used;'
             ' possible are "cyclic", "sqrt", "linear" and "constant"(no learning rate decay);'
             'default is "cyclic"',
        type=str,
        default="constant"
    )

    if args_list:
        args = parser.parse_args(args_list)
    else:
        args = parser.parse_args()

    network = get_network(args.network_name, args.architecture)
    args.num_workers = network.number_of_nodes()

    return args
