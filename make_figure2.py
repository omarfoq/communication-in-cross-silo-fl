import os
import json

import numpy as np
import matplotlib.pyplot as plt

from utils.utils import args_to_string, loggs_to_json
from utils.args import parse_args


cycle_time_dict = {"gaia": {"ring": 522.8,
                            "centralized": 9293.3,
                            "mst": 1442.0,
                            "mct_congest": 1018.8,
                            "matcha": 2612.8},
                   "amazon_us": {"ring": 485.9,
                                 "centralized": 18983.2,
                                 "mst": 1385.7,
                                 "mct_congest": 952.8,
                                 "matcha": 5036.7},
                   "geantdistance": {"ring": 491.1,
                                     "centralized": 35188.4,
                                     "mst": 2753.8,
                                     "mct_congest": 984.7,
                                     "matcha": 2658.9},
                   "exodus": {"ring": 488.1,
                              "centralized": 70350.7,
                              "mst": 3176.9,
                              "mct_congest": 1023.5,
                              "matcha": 2874.3},
                   "ebone": {"ring": 482.2,
                             "centralized": 77462.5,
                             "mst": 4123.4,
                             "mct_congest": 984.8,
                             "matcha": 2660.3}}

EXTENSIONS = {"synthetic": ".json",
              "sent140": ".json",
              "femnist": ".pkl",
              "shakespeare": ".txt",
              "inaturalist": ".pkl"}

# Model size in bit
MODEL_SIZE_DICT = {"synthetic": 4354,
                   "shakespeare": 3385747,
                   "femnist": 4843243,
                   "sent140": 19269416,
                   "inaturalist": 44961717}

# Model computation time in ms
COMPUTATION_TIME_DICT = {"synthetic": 1.5,
                         "shakespeare": 389.6,
                         "femnist": 4.6,
                         "sent140": 9.8,
                         "inaturalist": 25.4}

# Tags list
TAGS = ["Train/Loss", "Train/Acc", "Test/Loss", "Test/Acc", "Consensus"]

labels_dict = {"matcha": "MATCHA$^{+}$",
               "mst": "MST",
               "centralized": "STAR",
               'mct_congest': "$\delta$-MBST",
               "ring": "RING"}

tag_dict = {"Train/Loss": "Train loss",
            "Train/Acc": "Train acc",
            "Test/Loss": "Test loss",
            "Test/Acc": "Test acc",
            "Consensus": "Consensus"}

path_dict = {"Train/Loss": "Train_loss",
             "Train/Acc": "Train_acc",
             "Test/Loss": "Test_loss",
             "Test/Acc": "Test_acc",
             "Consensus": "Consensus"}

trsh_dict = {"gaia": 0.65,
             "amazon_us": 0.55,
             "geantdistance": 0.55,
             "exodus": 0.5,
             "ebone": 0.5}

lr_dict = {"gaia": "1e-3",
           "amazon_us": "1e-3",
           "geantdistance": "1e-3",
           "exodus": "1e-1",
           "ebone": "1e-1"}

bz_dict = {"shakespeare": 512,
           "femnist": 128,
           "sent140": 512,
           "inaturalist": 16}


def make_plots(args, mode=0):
    os.makedirs(os.path.join("results", "plots", args.experiment), exist_ok=True)

    loggs_dir_path = os.path.join("loggs", args_to_string(args))
    path_to_json = os.path.join("results", "json", "{}.json".format(os.path.split(loggs_dir_path)[1]))
    with open(path_to_json, "r") as f:
        data = json.load(f)

    # fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    x_lim = np.inf
    for idx, tag in enumerate(TAGS):
        fig = plt.figure(figsize=(12, 10))
        for architecture in ["centralized", "matcha", "mst", "mct_congest", "ring"]:
            try:
                values = data[tag][architecture]
                rounds = data["Round"][architecture]
            except:
                continue

            if mode == 0:
                min_len = min(len(values), len(rounds))

                if rounds[-1] * cycle_time_dict[network_name][architecture] < x_lim:
                    x_lim = rounds[-1] * cycle_time_dict[network_name][architecture]

                plt.plot(cycle_time_dict[network_name][architecture] * np.array(rounds) / 1000,
                         values[:min_len], label=labels_dict[architecture],
                         linewidth=5.0)
                plt.grid(True, linewidth=2)
                plt.xlim(0, x_lim / 1000)
                plt.ylabel("{}".format(tag_dict[tag]), fontsize=50)
                plt.xlabel("time (s)", fontsize=50)
                plt.tick_params(axis='both', labelsize=40)
                plt.tick_params(axis='x')
                plt.legend(fontsize=35)

            else:
                min_len = min(len(values), len(rounds))

                if rounds[:min_len][-1] < x_lim:
                    x_lim = rounds[:min_len][-1]

                plt.plot(rounds[:min_len],
                         values[:min_len], label=labels_dict[architecture],
                         linewidth=5.0)
                plt.ylabel("{}".format(tag_dict[tag]), fontsize=50)
                plt.xlabel("Rounds", fontsize=50)
                plt.tick_params(axis='both', labelsize=40)
                plt.legend(fontsize=35)
                plt.grid(True, linewidth=2)
                plt.xlim(0, x_lim)

        if mode == 0:
            fig_path = os.path.join("results", "plots", args.experiment,
                                    "{}_{}_vs_time.png".format(args.network_name, path_dict[tag]))
            plt.savefig(fig_path, bbox_inches='tight')
        else:
            fig_path = os.path.join("results", "plots", args.experiment,
                                    "{}_{}_vs_iteration.png".format(args.network_name, path_dict[tag]))
            plt.savefig(fig_path, bbox_inches='tight')


if __name__ == "__main__":
    network_name = "amazon_us"

    for experiment in [ "inaturalist", "shakespeare", "sent140", "femnist"]:
        args = parse_args([experiment,
                           "--network", network_name,
                           "--bz", str(bz_dict[experiment]),
                           "--lr", str(lr_dict[network_name]),
                           "--decay", "sqrt",
                           "--local_steps", "1"])

        args_string = args_to_string(args)

        loggs_dir = os.path.join("loggs", args_to_string(args))
        loggs_to_json(loggs_dir)

        print("{}:".format(experiment))

        make_plots(args, mode=0)
        make_plots(args, mode=1)

        print("#" * 10)




