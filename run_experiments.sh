#!/bin/bash

source venv/bin/activate

# python spiral_experiment.py 10
# python spiral_experiment2.py 10 --missing None
# python spiral_experiment2.py 10 --missing MNAR

# python imputation_benchmark.py --repeats 10 --dataset spiral --epochs 200 --save
# python imputation_benchmark.py --repeats 10 --dataset spiral --epochs 200 --missing MCAR --p 0.8 --save
# python imputation_benchmark.py --repeats 10 --dataset spiral --epochs 200 --missing MAR --p 0.99 --save
# python imputation_benchmark.py --repeats 10 --dataset spiral --epochs 200 --missing MNAR --p 0.99 --save

 python imputation_benchmark.py --repeats 10 --dataset anneal --imputation None simple iterative --epochs 200 --save