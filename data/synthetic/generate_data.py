""" From https://github.com/TalwalkarLab/leaf/blob/master/data/synthetic/"""
import argparse
import json
import os
import numpy as np
from scipy.special import softmax

NUM_DIM = 10
PROB_CLUSTERS = [1.0]


class SyntheticDataset:
    def __init__(
            self,
            num_classes=2,
            seed=931231,
            num_dim=NUM_DIM,
            prob_clusters=[0.5, 0.5]):

        np.random.seed(seed)

        self.num_classes = num_classes
        self.num_dim = num_dim
        self.num_clusters = len(prob_clusters)
        self.prob_clusters = prob_clusters

        self.side_info_dim = self.num_clusters

        self.Q = np.random.normal(
            loc=0.0, scale=1.0, size=(self.num_dim + 1, self.num_classes, self.side_info_dim))

        self.Sigma = np.zeros((self.num_dim, self.num_dim))
        for i in range(self.num_dim):
            self.Sigma[i, i] = (i + 1) ** (-1.2)

        self.means = self._generate_clusters()

    def get_task(self, num_samples):
        cluster_idx = np.random.choice(
            range(self.num_clusters), size=None, replace=True, p=self.prob_clusters)
        new_task = self._generate_task(self.means[cluster_idx], cluster_idx, num_samples)
        return new_task

    def _generate_clusters(self):
        means = []
        for i in range(self.num_clusters):
            loc = np.random.normal(loc=0, scale=1., size=None)
            mu = np.random.normal(loc=loc, scale=1., size=self.side_info_dim)
            means.append(mu)
        return means

    def _generate_x(self, num_samples):
        B = np.random.normal(loc=0.0, scale=1.0, size=None)
        loc = np.random.normal(loc=B, scale=1.0, size=self.num_dim)

        samples = np.ones((num_samples, self.num_dim + 1))
        samples[:, 1:] = np.random.multivariate_normal(
            mean=loc, cov=self.Sigma, size=num_samples)

        return samples

    def _generate_y(self, x, cluster_mean):
        model_info = np.random.normal(loc=cluster_mean, scale=0.1, size=cluster_mean.shape)
        w = np.matmul(self.Q, model_info)

        num_samples = x.shape[0]
        prob = softmax(np.matmul(x, w) + np.random.normal(loc=0., scale=0.1, size=(num_samples, self.num_classes)),
                       axis=1)

        y = np.argmax(prob, axis=1)
        return y, w, model_info

    def _generate_task(self, cluster_mean, cluster_id, num_samples):
        x = self._generate_x(num_samples)
        y, w, model_info = self._generate_y(x, cluster_mean)

        # now that we have y, we can remove the bias coeff
        x = x[:, 1:]

        return {'x': x, 'y': y, 'w': w, 'model_info': model_info, 'cluster': cluster_id}


def main():
    args = parse_args()
    np.random.seed(args.seed)

    num_samples = get_num_samples(args.num_workers)
    dataset = SyntheticDataset(
        num_classes=args.num_classes, prob_clusters=PROB_CLUSTERS, num_dim=args.dimension, seed=args.seed)
    tasks = [dataset.get_task(s) for s in num_samples]
    users, num_samples, user_data = to_leaf_format(tasks)
    save_json('all_data', 'all_data.json', users, num_samples, user_data, args.num_classes)


def get_num_samples(num_tasks, min_num_samples=5, max_num_samples=1000):
    num_samples = np.random.lognormal(3, 2, (num_tasks)).astype(int)
    num_samples = [min(s + min_num_samples, max_num_samples) for s in num_samples]
    return num_samples


def to_leaf_format(tasks):
    users, num_samples, user_data = [], [], {}

    for i, t in enumerate(tasks):
        x, y = t['x'].tolist(), t['y'].tolist()
        u_id = str(i)

        users.append(u_id)
        num_samples.append(len(y))
        user_data[u_id] = {'x': x, 'y': y}

    return users, num_samples, user_data


def save_json(json_dir, json_name, users, num_samples, user_data, num_classes):
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)

    json_file = {
        'users': users,
        'num_samples': num_samples,
        'user_data': user_data,
        "num_classes": num_classes
    }

    with open(os.path.join(json_dir, json_name), 'w') as outfile:
        json.dump(json_file, outfile)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--num_workers',
        help='number of workers;',
        type=int,
        required=True)
    parser.add_argument(
        '--num_classes',
        help='number of classes;',
        type=int,
        required=True)
    parser.add_argument(
        '--dimension',
        help='data dimension;',
        type=int,
        required=True)
    parser.add_argument(
        '--seed',
        help='seed for the random processes;',
        type=int,
        default=931231,
        required=False)
    return parser.parse_args()


if __name__ == '__main__':
    main()
