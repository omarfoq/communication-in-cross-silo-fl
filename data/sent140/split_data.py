import os
import csv
import argparse
import json
import random
import time
import numpy as np


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
                    default=0.01)
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
    np.random.seed(rng_seed)

    train_file = os.path.join("train", "train.json")
    test_file = os.path.join("test", "test.json")

    data_dir = os.path.join('raw_data', 'all_data.csv')
    with open(data_dir, 'rt', encoding='ISO-8859-1') as f:
        reader = csv.reader(f)
        data = list(reader)

    data = sorted(data, key=lambda x: x[4])

    if args.iid:
        tot_num_samples = len(data)
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

            local_train_file = os.path.join("train", "{}.json".format(id_w))

            for (file_, indices) in [(local_train_file, train_indices),
                                     (train_file, train_indices),
                                     (test_file, test_indices)]:

                for sample_idx in indices:
                    sample = data[sample_idx]
                    row = dict()

                    row['idx'] = sample[1]
                    row["time"] = sample[2]
                    row['query'] = sample[3]
                    row["user"] = sample[4]
                    row["text"] = sample[5]
                    row["label"] = 1 if sample[0] == "4" else 0

                    with open(file_, "a") as f:
                        json.dump(row, f)
                        f.write("\n")

    else:
        all_writers = set()

        for i in range(len(data)):
            row = data[i]
            all_writers.add(row[4])

        all_writers = list(all_writers)

        data_by_writers = {k: [] for k in all_writers}

        for i in range(len(data)):
            row = data[i]
            data_by_writers[row[4]].append(row)

        num_writers_by_user = np.random.lognormal(5, 1.5, args.num_workers) + 5
        num_writers_by_user *= (len(all_writers) / num_writers_by_user.sum())
        num_samples = np.floor(num_writers_by_user).astype(np.int64)

        writers_by_workers = []
        current_idx = 0
        for worker_id in range(args.num_workers):
            writers_by_workers.append(all_writers[current_idx: current_idx + num_samples[worker_id]])
            current_idx = num_samples[worker_id]

        for id_w, writers in enumerate(writers_by_workers):
            all_worker_data = []
            for writer in writers:
                all_worker_data += data_by_writers[writer]

            tot_num_samples = len(all_worker_data)
            curr_num_samples = int(args.s_frac * tot_num_samples)

            indices = [i for i in range(tot_num_samples)]
            worker_indices = rng.sample(indices, curr_num_samples)

            num_train_samples = max(1, int(args.tr_frac * curr_num_samples))
            num_test_samples = curr_num_samples - num_train_samples

            train_indices = rng.sample(worker_indices, num_train_samples)
            test_indices = list(set(worker_indices) - set(train_indices))

            local_train_file = os.path.join("train", "{}.json".format(id_w))

            for (file_, indices) in [(local_train_file, train_indices),
                                     (train_file, train_indices),
                                     (test_file, test_indices)]:

                for sample_idx in indices:
                    sample = data[sample_idx]
                    row = dict()

                    row['idx'] = sample[1]
                    row["time"] = sample[2]
                    row['query'] = sample[3]
                    row["user"] = sample[4]
                    row["text"] = sample[5]
                    row["label"] = 1 if sample[0] == "4" else 0

                    with open(file_, "a") as f:
                        json.dump(row, f)
                        f.write("\n")


