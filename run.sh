#!/bin/bash

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <dataset> <t_value> <a_value> <c_value>"
    exit 1
fi

DATASET=$1
T_VALUE=$2
A_VALUE=$3
C_VALUE=$4

python main.py --dataset=$DATASET --t=$T_VALUE --a=$A_VALUE --context_hops=$C_VALUE
EOF
