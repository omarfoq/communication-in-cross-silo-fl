# Optimal Topology Design for Cross-Silo Federated Learning

This repository is the official implementation of Optimal Topology
Design for Cross-Silo Federated Learning.

Federated learning usually employs a master-slave architecture where an
orchestrator iteratively aggregates model updates from remote clients
and pushes them back a refined model. This approach may be inefficient
in cross-silo settings, as close-by data silos with high-speed access
links may exchange information faster than with the orchestrator, and
the orchestrator may become a communication bottleneck. In this paper we
define the problem of topology design for cross-silo federated learning
using the theory of max-plus linear systems to compute the system
throughput---number of communication rounds per time unit. We also
propose practical algorithms that, under the knowledge of measurable
network characteristics, find a topology with the largest throughput or
with provable throughput guarantees. In realistic Internet networks with
10 Gbps access links for silos, our algorithms speed up training by a
factor 9 and 1.5 in comparison to the master-slave architecture and to
state-of-the-art MATCHA, respectively. Speedups are even larger with
slower access links.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Datasets

We provide four datasets that are used in the paper under corresponding
folders. For all datasets, see the README files in separate
data/$dataset folders for instructions on preprocessing and/or sampling
data.

## Networks and Topologies

A main part of the paper is related to topology design. In
`graph_utils/` details on generating different topologies for each
network are provided. Scripts to compute the cycle time of each topology
are also provided in `graph_utils/`

## Training

Run on one dataset, with a specific topology choice on on network.
Specify the name of the dataset (experiment), the name of the network
and the used architecture, and configure all other hyper-parameters (see
all hyper-parameters values in the appendix of the paper)

```train
python3  main.py experiment ----network_name  \
         --architecture=original (--parallel) (--fit_by_epoch) \
         --n_rounds=1 --bz=1 
         --local_steps=1 --log_freq=1 \
         --device="cpu" --lr=1e-3\
         --optimizer='adam' --decay="constant"
```

And the test and training accuracy and loss will be saved in the log files.

## Evaluation

### iNaturalist Speed-ups
To evaluate the speed-ups obtained when training iNaturalist on the proposed topology architectures (generate Table 3) fora given network, run

```eval
python3 main.py inaturalist --network_name gaia --architecture $ARCHITECTURE --n_rounds 5600 --bz 16 --device cuda --log_freq 100 --local_steps 1 --lr 0.001 --decay sqrt
python3 main.py inaturalist --network_name amazon_us --architecture $ARCHITECTURE --n_rounds 1600 --bz 16 --device cuda --log_freq 40 --local_steps 1 --lr 0.001 --decay sqrt
python3 main.py inaturalist --network_name geantdistance --architecture $ARCHITECTURE --n_rounds 4000 --bz 16 --device cuda --log_freq 100 --local_steps 1 --lr 0.001 --decay sqrt
python3 main.py inaturalist --network_name exodus --architecture $ARCHITECTURE --n_rounds 4800 --bz 16 --device cuda --log_freq 100 --local_steps 1 --lr 0.1 --decay sqrt --optimizer sgd
python3 main.py inaturalist --network_name ebone --architecture $ARCHITECTURE --n_rounds 6000 --bz 16 --device cuda --log_freq 100 --local_steps 1 --lr 0.1 --decay sqrt --optimizer sgd
```

And the test and training accuracy and loss for the corresponding experiment will be saved in the log files.

Do this operation for all architectures ($ARCHITECTURE=ring, centralized, matcha, exodus, ebone).  
Remind that for every network, a new generation of dataset (data/$dataset folders) is required to distribute data into silos. 

Then run

```eval
python3 make_table3.py
```

To generate the values from Table 3.

### Effect of the topology on the convergence

To evaluate the influence of topology on the training evolution for the different datasets when trained on AWS-NA network, run

```eval
python  main.py inaturalist --network_name amazon_us --architecture ring --n_rounds 1600 --bz 16 --device cuda --log_freq 40 --local_steps 1 --lr 0.001 --decay sqrt
python  main.py inaturalist --network_name amazon_us --architecture centralized --n_rounds 1600 --bz 16 --device cuda --log_freq 40 --local_steps 1 --lr 0.001 --decay sqrt
python  main.py inaturalist --network_name amazon_us --architecture matcha --n_rounds 1600 --bz 16 --device cuda --log_freq 40 --local_steps 1 --lr 0.001 --decay sqrt
python  main.py inaturalist --network_name amazon_us --architecture mst --n_rounds 1600 --bz 16 --device cuda --log_freq 40 --local_steps 1 --lr 0.001 --decay sqrt
python  main.py inaturalist --network_name amazon_us --architecture mct_congest --n_rounds 1600 --bz 16 --device cuda --log_freq 40 --local_steps 1 --lr 0.001 --decay sqrt

python main.py femnist --network_name amazon_us --architecture ring --n_rounds 6400 --bz 128 --device cuda --log_freq 80 --local_steps 1 --lr 0.001 --decay sqrt
python main.py femnist --network_name amazon_us --architecture centralized --n_rounds 6400 --bz 128 --device cuda --log_freq 80 --local_steps 1 --lr 0.001 --decay sqrt
python main.py femnist --network_name amazon_us --architecture matcha --n_rounds 6400 --bz 128 --device cuda --log_freq 80 --local_steps 1 --lr 0.001 --decay sqrt
python main.py femnist --network_name amazon_us --architecture mst --n_rounds 6400 --bz 128 --device cuda --log_freq 80 --local_steps 1 --lr 0.001 --decay sqrt
python main.py femnist --network_name amazon_us --architecture mct_congest --n_rounds 6400 --bz 128 --device cuda --log_freq 80 --local_steps 1 --lr 0.001 --decay sqrt

python main.py sent140 --network_name amazon_us --architecture ring --n_rounds 20000 --bz 512 --device cuda --log_freq 100 --local_steps 1 --lr 0.001 --decay sqrt
python main.py sent140 --network_name amazon_us --architecture centralized --n_rounds 20000 --bz 512 --device cuda --log_freq 100 --local_steps 1 --lr 0.001 --decay sqrt
python main.py sent140 --network_name amazon_us --architecture matcha --n_rounds 20000 --bz 512 --device cuda --log_freq 100 --local_steps 1 --lr 0.001 --decay sqrt
python main.py sent140 --network_name amazon_us --architecture mst --n_rounds 20000 --bz 512 --device cuda --log_freq 100 --local_steps 1 --lr 0.001 --decay sqrt
python main.py sent140 --network_name amazon_us --architecture mct_congest --n_rounds 20000 --bz 512 --device cuda --log_freq 100 --local_steps 1 --lr 0.001 --decay sqrt

python main.py shakespeare --network_name amazon_us --architecture ring --n_rounds 1500 --bz 512 --decay sqrt --lr 1e-3 --device cuda --local_steps 1 --log_freq 30
python main.py shakespeare --network_name amazon_us --architecture centralized --n_rounds 1500 --bz 512 --decay sqrt --lr 1e-3 --device cuda --local_steps 1 --log_freq 30
python main.py shakespeare --network_name amazon_us --architecture matcha --n_rounds 1500 --bz 512 --decay sqrt --lr 1e-3 --device cuda --local_steps 1 --log_freq 30
python main.py shakespeare --network_name amazon_us --architecture mst --n_rounds 1500 --bz 512 --decay sqrt --lr 1e-3 --device cuda --local_steps 1 --log_freq 30
python main.py shakespeare --network_name amazon_us --architecture mct_congest --n_rounds 1500 --bz 512 --decay sqrt --lr 1e-3 --device cuda --local_steps 1 --log_freq 30
```

to generate the log files for each experiment. Tne run

```eval
python3 make_figure2.py
```

to generate Figure 2. (Figures will be found in `results/plots`)

## Results

### iNaturalist Speed-ups
Our topology design achieves the following speed-ups when training
iNaturalist dataset over different networks:


|Network Name         | Silos  | Links | Ring vs Star speed-up | Ring vs MATCHA speed-up|
| ------------------  |  ------|-------|----------------       | --------------         |
| Gaia    |     11       |      55              |2.65        | 1.54 |
| AWS NA    |    22      |      321              |3.41          |1.47|
| Géant   |     40        |      61             |4.85          |0.81|
| Exodus    |     79        |      147              |8.78          |1.37|
| Ebone    |     87        |      161              |8.83          |1.29|

### Effect of the topology on the convergence

Effect of overlays on the convergence w.r.t. communication rounds  (top row)  
and wall-clock time(bottom row) when training four different datasets on  
 AWS North America underlay.1Gbps core links capacities, 100Mbps access  
 links capacities,s= 1.

![](https://user-images.githubusercontent.com/42912620/84382812-7e215780-abeb-11ea-94f5-e08e506ace89.PNG)
