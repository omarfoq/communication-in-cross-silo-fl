from utils.utils import args_to_string, loggs_to_json
from utils.args import parse_args

import os
import json


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

for network_name in ["gaia", "amazon_us", "geantdistance", "exodus", "ebone"]:
    print("{}:".format(network_name))
    args = parse_args(["inaturalist",
                       "--network", network_name,
                       "--bz", "16",
                       "--lr", lr_dict[network_name],
                       "--decay", "sqrt",
                       "--local_steps", "1"])

    args_string = args_to_string(args)

    loggs_dir = os.path.join("loggs", args_to_string(args))
    loggs_to_json(loggs_dir)

    loggs_dir_path = os.path.join("loggs", args_to_string(args))
    path_to_json = os.path.join("results", "json", "{}.json".format(os.path.split(loggs_dir_path)[1]))
    with open(path_to_json, "r") as f:
        data = json.load(f)

    for architecture in ["centralized", "ring", "matcha"]:
        values = data['Train/Acc'][architecture]
        rounds = data["Round"][architecture]

        ii = -1
        for ii, value in enumerate(values):
            if value > trsh_dict[network_name]:
                break

        try:
            print("Number of steps to achieve {}% is {} on {} using {}".format(int(trsh_dict[network_name] * 100),
                                                                               rounds[ii], network_name, architecture))
        except IndexError:
            print("Number of steps to achieve {}% is {} on {} using {}".format(int(trsh_dict[network_name] * 100),
                                                                               rounds[-1], network_name, architecture))

    print("#" * 10)
