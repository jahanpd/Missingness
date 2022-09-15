#!/bin/bash

# Ensure in environment with requirements.txt installed before running
# SYNTHETIC EXPERIMENTS
repeats=10

python spiral_experiment1.py $repeats
python spiral_experiment2.py $repeats

# BENCHMARKING ON REAL WORLD DATASETS
# pulls datasets from OpenML with flags to control settings
python get_datainfo.py --corrupt --min_classes 2 --min_features 30 --max_features 250  --min_instances 1000 --max_instances 10000 --save
python get_datainfo.py --min_p_missing 0.05 --not_suite --min_classes 2 --min_features 30 --max_features 250  --min_instances 1000 --maxembedding_layers_instances 10000 --save

# run hyperparameter optimization with bayesian optimization using wandb platform
python sweeps.py --corrupt
python sweeps.py

# run two benchmarks
## first on datasets with established missingess
python benchmark.py --missing None --imputation None simple iterative miceforest --seed 92653
## then on datasets with no missing data that are corrupted according to predefined missingess patterns 
python benchmark.py --missing None MCAR MAR MNAR --imputation None simple iterative miceforest --corrupt --seed 31415

# get results from wandb database and prep for analysis
python process_results.py --corrupt --save
python process_results.py --noncorrupt --save

# format into tables and save
cd aux
python view_tables.py
python view_tables.py --corrupt

# generate critical difference diagrams
cd ../cd_diagram
python run_diagram.py 
python run_diagram.py --corrupt

# perform meta-learning and analysis
