#!/bin/bash

source venv/bin/activate

repeats=30

python spiral_experiment1.py $repeats
python spiral_experiment2.py $repeats
