import os
import argparse
import json
import random
import time
import numpy as np
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()


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
    rng_seed = (args.seed if (args.seed is not None and args.seed >= 0) else int(time.time()))
    rng = random.Random(rng_seed)

    data_file = os.path.join('all_data', 'all_data.json')

    with open(data_file, 'r') as inf:
        data = json.load(inf)

    X_list = {"train": [], "test": []}
    y_list = {"train": [], "test": []}

    num_classes = data['num_classes']

    for worker in data['users']:
        train_file = os.path.join("train", "{}.json".format(worker))

        worker_data = data['user_data'][worker]
        X = np.array(worker_data['x'])
        y = np.array(worker_data['y'])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=args.tr_frac, random_state=args.seed)

        X_list["train"].append(X_train)
        y_list["train"].append(y_train)
        X_list["test"].append(X_test)
        y_list["test"].append(y_test)

        json_data_train = {"x": X_train.tolist(), "y": y_train.tolist(), "num_classes": num_classes}

        with open(train_file, 'w') as outfile:
            json.dump(json_data_train, outfile)

    for key in ["train", "test"]:
        X = np.vstack(X_list[key])
        y = np.concatenate(y_list[key])

        file = os.path.join(key, "{}.json".format(key))
        json_data = {"x": X.tolist(), "y": y.tolist(), "num_classes": num_classes}
        with open(file, 'w') as outfile:
            json.dump(json_data, outfile)

