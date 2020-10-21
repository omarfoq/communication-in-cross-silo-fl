# Synthetic Dataset

## Setup Instructions

Run generate_data.sh with a choice of the following tags:

- ```-nw```: number of workers, written as integer
- ```-nc``` : number of classes, written as integer
- ```-dim```: dimension of the data, written as integer
- ```--tf``` := fraction of data in training set, written as a decimal; default is 0.9
- ```--seed``` := seed to be used before random sampling of data

i.e. 
- ```./generate_data.sh -s -nw 11 -nc 2 -dim 10 -tf 0.8 -seed 1234``` 
