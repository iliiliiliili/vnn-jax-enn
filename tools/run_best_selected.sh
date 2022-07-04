#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate variational

CUDA_VISIBLE_DEVICES="" python ./enn/experiments/neurips_2021/run_testbed_best_selected.py "$@"
