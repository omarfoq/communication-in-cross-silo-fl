import argparse
import json
import os
import pickle
import time
import random
from collections import Counter

import networkx as nx
import numpy as np

import geopy.distance
from geopy.geocoders import Nominatim


class FileException(FileNotFoundError):
    def __init__(self, message):
        super().__init__(message)


parser = argparse.ArgumentParser()

parser.add_argument('--network',
                    help="name of the network to use, should be present in /graph_utils/data; default: amazon_us",
                    type=str,
                    default="amazon_us")
parser.add_argument('--num_categories',
                    help="number of classes to include, default: 80",
                    type=int,
                    default="80")
parser.add_argument('--s_frac',
                    help='fraction of all data to sample; default: 0.1;',
                    type=float,
                    default=1)
parser.add_argument('--tr_frac',
                    help='fraction in training set; default: 0.8;',
                    type=float,
                    default=0.9)
parser.add_argument('--seed',
                    help='args.seed for random partitioning of test/train data',
                    type=int,
                    default=None)

args = parser.parse_args()


if __name__ == "__main__":
    network_path = os.path.abspath(os.path.join(os.getcwd(), "..", "..", "graph_utils/data", args.network + ".gml"))

    if not os.path.isfile(network_path):
        raise FileException("The network with name {} is not found!".format(network_path))

    rng_seed = (args.seed if (args.seed is not None and args.seed >= 0) else int(time.time()))
    rng = random.Random(rng_seed)
    np.random.seed(rng_seed)

    # Get workers locations
    network_path = os.path.abspath(os.path.join(os.getcwd(), "..", "..", "graph_utils/data", args.network + ".gml"))
    workers_network = nx.read_gml(network_path, label="label")
    nodes_locs = []
    geolocator = Nominatim(user_agent="delay", timeout=20)
    for node in workers_network.nodes():
        time.sleep(1.0)  # To avoid Service time out Error
        geo = geolocator.geocode(node, timeout=20)
        nodes_locs.append((geo.latitude, geo.longitude))

    # Get the information for images and locations
    with open(os.path.join("raw_data", "train2018_locations.json")) as f:
        train_imgs_locations = json.load(f)

    with open(os.path.join("raw_data", "val2018_locations.json")) as f:
        val_imgs_locations = json.load(f)

    with open(os.path.join("raw_data", "train2018.json")) as f:
        train_images_data = json.load(f)

    with open(os.path.join("raw_data", "val2018.json")) as f:
        val_images_data = json.load(f)

    all_data = dict()
    for images_data in [train_images_data, val_images_data]:
        for img, annotation in zip(images_data["images"], images_data["annotations"]):
            img_id = img["id"]
            img_path = ["raw_data/"] + img["file_name"].split("/")[1:]
            img_path = "/".join(img_path)
            category_id = annotation["category_id"]

            all_data[img_id] = {"path": img_path, "class": category_id}

    for imgs_locations in [train_imgs_locations, val_imgs_locations]:
        for location in imgs_locations:
            img_id = location["id"]
            all_data[img_id]["lat"] = location["lat"]
            all_data[img_id]["lon"] = location["lon"]

    # Get most common categories
    all_categories = []
    for img_id in all_data:
        all_categories.append(all_data[img_id]['class'])

    c = Counter(all_categories)
    most_common_categories = c.most_common(args.num_categories)
    most_common_categories = [i for (i, j) in most_common_categories]

    relabel_categories = {category: idx for idx, category in enumerate(most_common_categories)}
    most_common_categories = set(most_common_categories)

    # Assign images to closest workers
    imgs_by_workers = {worker_id: [] for worker_id in range(workers_network.number_of_nodes())}

    for img_id in all_data:
        category = all_data[img_id]['class']
        if category in most_common_categories:
            # Get closest worker to node
            coord_img = (all_data[img_id]['lat'], all_data[img_id]['lon'])
            distances = np.array([geopy.distance.distance(coord_img, coord_node).km for coord_node in nodes_locs])
            worker_id = np.argmin(distances)

            img_data = (all_data[img_id]["path"], relabel_categories[category])

            imgs_by_workers[worker_id].append(img_data)

    # Split data to train and test
    train_data = []
    test_data = []

    for worker_id in imgs_by_workers.keys():
        all_worker_data = imgs_by_workers[worker_id]

        tot_num_samples = len(all_worker_data)
        num_new_samples = int(args.s_frac * tot_num_samples)

        indices = [i for i in range(tot_num_samples)]
        new_indices = rng.sample(indices, num_new_samples)

        num_train_samples = max(1, int(args.tr_frac * num_new_samples))
        num_test_samples = num_new_samples - num_train_samples

        train_indices = rng.sample(new_indices, num_train_samples)
        test_indices = list(set(new_indices) - set(train_indices))

        worker_data = [all_worker_data[ii] for ii in train_indices]
        train_data += [all_worker_data[ii] for ii in train_indices]
        test_data += [all_worker_data[ii] for ii in test_indices]

        with open('train/{}.pkl'.format(worker_id), 'wb') as f:
            pickle.dump(worker_data, f, pickle.HIGHEST_PROTOCOL)

    with open('train/train.pkl', 'wb') as f:
        pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)

    with open('test/test.pkl', 'wb') as f:
        pickle.dump(test_data, f, pickle.HIGHEST_PROTOCOL)
