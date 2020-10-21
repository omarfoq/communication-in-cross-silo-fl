# iNaturalist Dataset

## Setup Instructions

* Download iNaturalist
  [here](https://storage.googleapis.com/inat_data_2018_eu/train_val2018.tar.gz),
  unzip it and place its content in ``raw_data`` folder.

* Run preprocess.sh with a choice of the following tags:

  - ```--network```:= name of the network to use, should be present in
    ``/graph_utils/data``, default is us-amzaon
  - ```--sf``` := fraction of data to sample, written as a decimal;
    default is 0.1
  - ```--tf``` := fraction of data in training set, written as a decimal; default is 0.9
  - ```--seed``` := seed to be used before random sampling of data

i.e.
- ```./preprocess.sh --sf 1.0 --tf 0.9 --seed 1234``` (full-sized
  dataset partitioned on Gaia)<br/>
