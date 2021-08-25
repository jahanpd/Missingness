#!/bin/bash

source venv/bin/activate

repeats=30

# run distance experiments
# python spiral_experiment.py --repeats $repeats
# python spiral_experiment2.py --repeats $repeats --missing None, MNAR

# run imputation/missingness benchmark experiments

# datasets with artifically removed data
# python imputation_benchmark.py --repeats $repeats --dataset spiral --missing None --imputation None --save
python imputation_benchmark.py --repeats $repeats --dataset spiral --missing MCAR --p 0.1 --imputation None simple iterative miceforest --save
python imputation_benchmark.py --repeats $repeats --dataset spiral --missing MAR --p 0.5 --imputation None simple iterative miceforest --save
python imputation_benchmark.py --repeats $repeats --dataset spiral --missing MNAR --p 0.5 --imputation None simple iterative miceforest --save

python imputation_benchmark.py --repeats $repeats --dataset thoracic --missing None --imputation None --save
python imputation_benchmark.py --repeats $repeats --dataset thoracic --missing MCAR --p 0.1 --imputation None simple iterative miceforest --save
python imputation_benchmark.py --repeats $repeats --dataset thoracic --missing MAR --p 0.1 --imputation None simple iterative miceforest --save
python imputation_benchmark.py --repeats $repeats --dataset thoracic --missing MNAR --p 0.1 --imputation None simple iterative miceforest --save

python imputation_benchmark.py --repeats $repeats --dataset abalone --missing None --imputation None --save
python imputation_benchmark.py --repeats $repeats --dataset abalone --missing MCAR --p 0.1 --imputation None simple iterative miceforest --save
python imputation_benchmark.py --repeats $repeats --dataset abalone --missing MAR --p 0.1 --imputation None simple iterative miceforest --save
python imputation_benchmark.py --repeats $repeats --dataset abalone --missing MNAR --p 0.1 --imputation None simple iterative miceforest --save

python imputation_benchmark.py --repeats $repeats --dataset mnist --missing None --imputation None simple iterative --save
python imputation_benchmark.py --repeats $repeats --dataset mnist --missing MCAR --imputation None simple iterative --save
python imputation_benchmark.py --repeats $repeats --dataset mnist --missing MAR --imputation None simple iterative --save

# datasets with already missing data
python imputation_benchmark.py --repeats $repeats --dataset anneal --imputation None simple iterative --save
python imputation_benchmark.py --repeats $repeats --dataset banking --imputation None simple iterative miceforest --save