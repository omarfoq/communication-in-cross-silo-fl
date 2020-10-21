import os
import pickle
import argparse
import random
import time
import numpy as np


def relabel_class(c):
    """
    maps hexadecimal class value (string) to a decimal number
    returns:
    - 0 through 9 for classes representing respective numbers
    - 10 through 35 for classes representing respective uppercase letters
    - 36 through 61 for classes representing respective lowercase letters
    """
    if c.isdigit() and int(c) < 40:
        return int(c) - 30
    elif int(c, 16) <= 90:  # uppercase
        return int(c, 16) - 55
    else:
        return int(c, 16) - 61


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

    data_dir = os.path.join('intermediate', 'images_by_writer.pkl')
    with open(data_dir, 'rb') as f:
        all_data = pickle.load(f)

    if args.iid:
        combined_data = []

        for (writer_id, l) in all_data:
            combined_data += l

        for ii, (path, c) in enumerate(combined_data):
            combined_data[ii] = (path, relabel_class(c))

        tot_num_samples = len(combined_data)
        num_new_samples = int(args.s_frac * tot_num_samples)

        indices = [i for i in range(tot_num_samples)]
        new_indices = rng.sample(indices, num_new_samples)

        indices_groups = iid_divide(new_indices, args.num_workers)

        train_data = []
        test_data = []

        for id_w, worker_indices in enumerate(indices_groups):
            curr_num_samples = len(worker_indices)

            num_train_samples = max(1, int(args.tr_frac * curr_num_samples))
            num_test_samples = curr_num_samples - num_train_samples

            train_indices = rng.sample(worker_indices, num_train_samples)
            test_indices = list(set(indices) - set(train_indices))

            worker_data = [combined_data[ii] for ii in train_indices]
            train_data += [combined_data[ii] for ii in train_indices]
            test_data += [combined_data[ii] for ii in test_indices]

            with open('train/{}.pkl'.format(id_w), 'wb') as f:
                pickle.dump(worker_data, f, pickle.HIGHEST_PROTOCOL)

        with open('train/train.pkl', 'wb') as f:
            pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)

        with open('test/test.pkl', 'wb') as f:
            pickle.dump(test_data, f, pickle.HIGHEST_PROTOCOL)

    else:
        writer_ids = [i for i in range(len(all_data))]
        rng.shuffle(writer_ids)

        num_writers_by_user = np.random.lognormal(5, 1.5, args.num_workers) + 5
        num_writers_by_user *= (len(writer_ids) / num_writers_by_user.sum())
        num_samples = np.floor(num_writers_by_user).astype(np.int64)

        writers_by_workers = []
        current_idx = 0
        for worker_id in range(args.num_workers):
            writers_by_workers.append(writer_ids[current_idx: current_idx + num_samples[worker_id]])
            current_idx = num_samples[worker_id]

        train_data = []
        test_data = []

        for id_w, writer_indices in enumerate(writers_by_workers):
            all_worker_data = []
            for writer_id in writer_indices:
                all_worker_data += all_data[writer_id][1]

            for ii, (path, c) in enumerate(all_worker_data):
                all_worker_data[ii] = (path, relabel_class(c))

            tot_num_samples = len(all_worker_data)
            curr_num_samples = int(args.s_frac * tot_num_samples)

            indices = [i for i in range(tot_num_samples)]
            worker_indices = rng.sample(indices, curr_num_samples)

            num_train_samples = max(1, int(args.tr_frac * curr_num_samples))
            num_test_samples = curr_num_samples - num_train_samples

            train_indices = rng.sample(worker_indices, num_train_samples)
            test_indices = list(set(worker_indices) - set(train_indices))

            worker_data = [all_worker_data[ii] for ii in train_indices]
            train_data += [all_worker_data[ii] for ii in train_indices]
            test_data += [all_worker_data[ii] for ii in test_indices]

            with open('train/{}.pkl'.format(id_w), 'wb') as f:
                pickle.dump(worker_data, f, pickle.HIGHEST_PROTOCOL)

        with open('train/train.pkl', 'wb') as f:
            pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)

        with open('test/test.pkl', 'wb') as f:
            pickle.dump(test_data, f, pickle.HIGHEST_PROTOCOL)
