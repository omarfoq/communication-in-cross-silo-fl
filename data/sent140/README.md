# Sentiment140 Dataset

## Setup Instructions

Run preprocess.sh with a choice of the following tags:

- ```-nw```: number of workers, written as integer
- ```-s``` := 'iid' to sample in an i.i.d. manner, or 'niid' to sample
  in a non-i.i.d. manner; more information on i.i.d. versus non-i.i.d.
  is included in the 'Notes' section
- ```--sf``` := fraction of data to sample, written as a decimal;
  default is 0.1
- ```--tf``` := fraction of data in training set, written as a decimal; default is 0.9
- ```--seed``` := seed to be used before random sampling of data

i.e. 
- ```./preprocess.sh -s iid -nw 11--sf 1.0 -t sample``` (full-sized
  dataset partitioned on Gaia)<br/>


