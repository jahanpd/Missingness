#!/bin/bash

source venv/bin/activate

repeats=30

# run distance experiments
# python spiral_experiment.py $repeats
# python spiral_experiment2.py $repeats

# run imputation/missingness benchmark experiments

# datasets with artifically removed data
# python imputation_benchmark.py --repeats $repeats --dataset spiral --missing None --imputation None --save
# python imputation_benchmark.py --repeats $repeats --dataset spiral --missing MCAR --p 0.1 --imputation None simple iterative miceforest --save
# python imputation_benchmark.py --repeats $repeats --dataset spiral --missing MAR --p 0.5 --imputation None simple iterative miceforest --save
# python imputation_benchmark.py --repeats $repeats --dataset spiral --missing MNAR --p 0.3 --imputation None simple iterative miceforest --save

# python imputation_benchmark.py --repeats $repeats --dataset thoracic --missing None --imputation None --save
# python imputation_benchmark.py --repeats $repeats --dataset thoracic --missing MCAR --p 0.1 --imputation None simple iterative miceforest --save
# python imputation_benchmark.py --repeats $repeats --dataset thoracic --missing MAR --p 0.1 --imputation None simple iterative miceforest --save
# python imputation_benchmark.py --repeats $repeats --dataset thoracic --missing MNAR --p 0.1 --imputation None simple iterative miceforest --save

# python imputation_benchmark.py --repeats $repeats --dataset abalone --missing None --imputation None --save
# python imputation_benchmark.py --repeats $repeats --dataset abalone --missing MCAR --p 0.1 --imputation None simple iterative miceforest --save
# python imputation_benchmark.py --repeats $repeats --dataset abalone --missing MAR --p 0.1 --imputation None simple iterative miceforest --save
# python imputation_benchmark.py --repeats $repeats --dataset abalone --missing MNAR --p 0.1 --imputation None simple iterative miceforest --save

# datasets with already missing data
# python imputation_benchmark.py --repeats $repeats --dataset anneal --missing None --imputation None simple iterative --save
# python imputation_benchmark.py --repeats $repeats --dataset banking --missing None --imputation None simple iterative miceforest --save

# python benchmark_openml.py --repeats $repeats --missing None MCAR MAR MNAR --imputation None simple iterative miceforest --corrupt --save
export CUDA_VISIBLE_DEVICES=1
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
# export XLA_PYTHON_CLIENT_PREALLOCATE=true
# export XLA_PYTHON_CLIENT_MEM_FRACTION=.70
python benchmark_openml.py --missing None MNAR MCAR MAR --imputation None simple iterative miceforest --save --corrupt
#python benchmark_openml.py --repeats $repeats --missing None MCAR MAR MNAR --imputation None simple iterative miceforest --save --dataset 2 
#python benchmark_openml.py --repeats $repeats --missing None MCAR MAR MNAR --imputation None simple iterative miceforest --save --dataset 3
#python benchmark_openml.py --repeats $repeats --missing None MCAR MAR MNAR --imputation None simple iterative miceforest --save --dataset 0 
#python benchmark_openml.py --repeats $repeats --missing None MCAR MAR MNAR --imputation None simple iterative miceforest --save --dataset 4 
#python benchmark_openml.py --repeats $repeats  --missing None MCAR MAR MNAR --imputation None simple iterative miceforest --save --dataset 5 
#python benchmark_openml.py --repeats $repeats --missing None MCAR MAR MNAR --imputation None simple iterative miceforest --save --dataset 6 
#python benchmark_openml.py --repeats $repeats --missing None MCAR MAR MNAR --imputation None simple iterative miceforest --save --dataset 7 
#python benchmark_openml.py --repeats $repeats --missing None MCAR MAR MNAR --imputation None simple iterative miceforest --save --dataset 8 
#python benchmark_openml.py --repeats $repeats --missing None MCAR MAR MNAR --imputation None simple iterative miceforest --save --dataset 9 
#python benchmark_openml.py --repeats $repeats --missing None MCAR MAR MNAR --imputation None simple iterative miceforest --save --dataset 10 
#python benchmark_openml.py --repeats $repeats --missing None MCAR MAR MNAR --imputation None simple iterative miceforest --save --dataset 11
#python benchmark_openml.py --repeats $repeats --missing None MCAR MAR MNAR --imputation None simple iterative miceforest --save --dataset 12
# python benchmark_openml.py --repeats $repeats --missing None MCAR MAR MNAR --imputation None simple iterative miceforest --save --dataset 16 17 18 19 20 21 22 23 24
