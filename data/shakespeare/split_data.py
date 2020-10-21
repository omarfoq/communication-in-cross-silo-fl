import os
import argparse
import random
import time


def iid_divide(l, g):
    """
    divide list l among g groups
    each group has either int(len(l)/g) or int(len(l)/g)+1 elements
    returns a list of groups

    """
    num_elems = len(l)
    group_size = int(len(l)/g)
    num_big_groups = num_elems - g * group_size
    num_small_groups = g - num_big_groups
    glist = []
    for i in range(num_small_groups):
        glist.append(l[group_size * i : group_size * (i + 1)])
    bi = group_size*num_small_groups
    group_size += 1
    for i in range(num_big_groups):
        glist.append(l[bi + group_size * i:bi + group_size * (i + 1)])
    return glist


parser = argparse.ArgumentParser()

parser.add_argument('--num_workers',
                    help=('number of workers/users;'
                          'default: 1;'),
                    type=int,
                    default=1)
parser.add_argument('--iid',
                    help='sample iid;',
                    action="store_true")
parser.add_argument('--niid',
                    help="sample niid;",
                    dest='iid', action='store_false')
parser.add_argument('--s_frac',
                    help='fraction of all data to sample; default: 0.1;',
                    type=float,
                    default=0.1)
parser.add_argument('--tr_frac',
                    help='fraction in training set; default: 0.8;',
                    type=float,
                    default=0.8)
parser.add_argument('--seed',
                    help='args.seed for random partitioning of test/train data',
                    type=int,
                    default=None)

parser.set_defaults(user=False)

args = parser.parse_args()


if __name__ == "__main__":
    print('------------------------------')
    print('generating training and test sets')

    rng_seed = (args.seed if (args.seed is not None and args.seed >= 0) else int(time.time()))
    rng = random.Random(rng_seed)

    train_file = os.path.join("train", "train.txt")
    test_file = os.path.join("test", "test.txt")

    data_dir = os.path.join('raw_data', 'by_play_and_character')

    if args.iid:
        # TO DO: Factorize this part
        all_lines = []
        for file_name in os.listdir(data_dir):
            file_path = os.path.join(data_dir, file_name)
            with open(file_path, "r") as f:
                lines = f.readlines()
            all_lines += lines

        tot_num_samples = len(all_lines)
        num_new_samples = int(args.s_frac * tot_num_samples)

        indices = [i for i in range(tot_num_samples)]
        new_indices = rng.sample(indices, num_new_samples)

        indices_groups = iid_divide(new_indices, args.num_workers)

        for id_w, worker_indices in enumerate(indices_groups):
            curr_num_samples = len(worker_indices)

            num_train_samples = max(1, int(args.tr_frac * curr_num_samples))
            num_test_samples = curr_num_samples - num_train_samples

            train_indices = rng.sample(worker_indices, num_train_samples)
            test_indices = list(set(worker_indices) - set(train_indices))

            local_train_file = os.path.join("train", "{}.txt".format(id_w))

            for (file_, indices) in [(train_file, train_indices),
                                     (local_train_file, train_indices),
                                     (test_file, test_indices)]:

                for sample_idx in indices:
                    sample = all_lines[sample_idx]

                    with open(file_, "a") as f:
                        f.write(sample)
    else:
        writers = os.listdir(data_dir)

        rng.shuffle(writers)
        writers_by_workers = iid_divide(writers, args.num_workers)

        for id_w, worker_writers in enumerate(writers_by_workers):
            all_worker_lines = []
            for writer in worker_writers:
                file_path = os.path.join(data_dir, writer)
                with open(file_path, "r") as f:
                    lines = f.readlines()

                all_worker_lines += lines

            tot_num_samples = len(all_worker_lines)
            num_new_samples = int(args.s_frac * tot_num_samples)

            indices = [i for i in range(tot_num_samples)]
            new_indices = rng.sample(indices, num_new_samples)

            new_worker_lines = [all_worker_lines[i] for i in new_indices]

            num_train_samples = max(1, int(args.tr_frac * num_new_samples))
            num_test_samples = num_new_samples - num_train_samples

            train_indices = rng.sample(new_indices, num_train_samples)
            test_indices = list(set(new_indices) - set(train_indices))

            local_train_file = os.path.join("train", "{}.txt".format(id_w))

            for (file_, indices) in [(train_file, train_indices),
                                     (local_train_file, train_indices),
                                     (test_file, test_indices)]:

                for sample_idx in indices:
                    sample = all_worker_lines[sample_idx]

                    with open(file_, "a") as f:
                        f.write(sample)
