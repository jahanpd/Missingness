import pandas as pd
import numpy as np 

datasets = pd.read_csv('./results/openml/corrupted_tasklist.csv')

print(list(datasets))

view = ['tid', 'ttid', 'did', 'name', 'task_type', 'MajorityClassSize', 'MaxNominalAttDistinctValues', 'MinorityClassSize', 'NumberOfClasses', 'NumberOfFeatures', 'NumberOfInstances',
        'NumberOfInstancesWithMissingValues', 'NumberOfMissingValues', 'NumberOfNumericFeatures', 'NumberOfSymbolicFeatures', 'number_samples']

print(datasets[view])
