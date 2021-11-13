import pandas as pd
import numpy as np

datasets = pd.read_csv('./results/openml/corrupted_tasklist.csv')

print(list(datasets))

view = ['tid', 'did', 'name', 'NumberOfInstances', 'task_type', 'MajorityClassSize', 'MaxNominalAttDistinctValues', 'MinorityClassSize', 'NumberOfClasses', 'NumberOfFeatures', 
        'NumberOfInstancesWithMissingValues', 'NumberOfMissingValues', 'NumberOfNumericFeatures', 'NumberOfSymbolicFeatures', 'number_samples']

print(datasets[view])
