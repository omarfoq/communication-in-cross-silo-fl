# Graph Generator

Generate different overlays given a connectivity graph. The connectivity
graph should be stored in ``data`` as a ``.gml`` file
## Setup Instructions

Run ```generate_network.py``` with a choice of the following arguments:

- ```name```: name of the used network;
- ```--experiment```: name of the experiment that will be run on the
  network; possible are femnist, inaturalist, synthetic, shakespeare,
  sent140; if not precised --model_size will be used as model size;
- ``--model_size``: size of the model that will be transmitted on the
  network in bit; will be ignored if --experiment is precised; default
  is 1e8;
- ``--default_capacity``: default capacity (in bit/s) to use on links
  with unknown capacity; default is 1e9;
- ```--centrality``` : Type of centrality to use in order to select the
  central node of the network; possible values are: "load", "distance"
  and "information"; default is "load";


i.e. 
- ```python3 generate_network.py amazon_us --experiment inaturalist```
  (generate different overlays with Amazon North America as connectivity
  graph for iNaturalist experiment)<br/>

To generate all the topologies for all the networks run

```
.\generate_all_networks.sh
```