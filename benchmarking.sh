#!/bin/bash

source venv/bin/activate

# ENV OPTIONS FOR RUNNING ON GPU
# export CUDA_VISIBLE_DEVICES=1
# export XLA_PYTHON_CLIENT_ALLOCATOR=platform

python benchmark_openml.py --missing None MNAR MCAR MAR --imputation None simple iterative miceforest --save --corrupt
