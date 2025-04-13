# DAR: Dimension-Adaptive Recommendation with Multi-Granular Noise Control, SIGIR 2025

PyTorch Implementation for DAR: Dimension-Adaptive Recommendation with Multi-Granular Noise Control

--- Based on "MixGCF: An Improved Training Method for Graph Neural Network-based Recommender Systems", https://github.com/huangtinglin/MixGCF


#### Environment Requirement

The code has been tested running under Python 3.7.6. The required packages are as follows:

- pytorch == 1.7.0
- numpy == 1.20.2
- scipy == 1.6.3
- sklearn == 0.24.1
- prettytable == 2.1.0



#### Training

The training commands are as following:

```
python main.py --dataset=$1 --t=$2 --a=$3 --context_hops=$4
```

or use run.sh

```
./run.sh dataset t a context_hops
```

The output will be in the ./logs/ folder.
