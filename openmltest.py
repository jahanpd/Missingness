import numpy as np
import pandas as pd
from scipy.io.arff import loadarff
# functions to prep dataset
import os
import openml

def get_dids(missing, sample, key=42, test=lambda x, m: x > m):
    rng = np.random.default_rng(key)
    # to generate list of datasets with high proportion of missingness
    class_tasks = openml.tasks.list_tasks(task_type=openml.tasks.TaskType(1), output_format='dataframe')
    reg_tasks = openml.tasks.list_tasks(task_type=openml.tasks.TaskType(2), output_format='dataframe')
    # ensure stats present
    class_tasks = class_tasks.dropna(subset=['evaluation_measures'])
    reg_tasks = reg_tasks.dropna(subset=['evaluation_measures'])
    # where there are greater than 50% rows with missing data
    class_subset = class_tasks[test(class_tasks.NumberOfInstancesWithMissingValues / class_tasks.NumberOfInstances, missing)]
    reg_subset = reg_tasks[test(reg_tasks.NumberOfInstancesWithMissingValues / reg_tasks.NumberOfInstances, missing)]
    # limit number of features to less than 500 for tractability
    class_subset = class_subset[(class_subset.NumberOfFeatures < 500) & (class_subset.NumberOfInstances < 100000)].drop_duplicates(subset=["name"])
    reg_subset = reg_subset[(reg_subset.NumberOfFeatures < 500) & (reg_subset.NumberOfInstances < 100000)].drop_duplicates(subset=["name"])
    
    # sample sample number from remaining
    k1 = rng.integers(9999)
    class_subset = class_subset.sample(n=sample, random_state=k1)
    k2 = rng.integers(9999)
    reg_subset = reg_subset.sample(n=sample, random_state=k2)
    return class_subset, reg_subset


df_class_miss, df_reg_miss = get_dids(0.5, 10)
df_class, df_reg = get_dids(0, 3, test=lambda x, m: x == m)
print(df_class_miss)
print(df_class)

for i, did in enumerate(df_reg.did.values):
    ds = openml.datasets.get_dataset(dataset_id=int(did))
    label = ds.retrieve_class_labels()  # if none, then regression
    print("label", label)
    print(ds.default_target_attribute)
    X, y, categorical_indicator, attribute_names = ds.get_data(target = df_reg.target_feature.values[i])
    print(X)
    print(y)
    print(categorical_indicator)
    print(attribute_names)