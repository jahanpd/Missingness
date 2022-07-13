import UAT.datasets as data
import pandas as pd
import argparse

parser = argparse.ArgumentParser()


parser.add_argument("--corrupt", action='store_true')
parser.add_argument("--min_p_missing", default=0.1, type=float,
                    help="Minimum percentage of missing data in dataset. Only relevant if corrupt flag is ABSENT.")
parser.add_argument("--min_features", default=3, type=int)
parser.add_argument("--max_features", default=250, type=int)
parser.add_argument("--min_instances", default=100, type=int)
parser.add_argument("--max_instances", default=50000, type=int)

args = parser.parse_args()
if args.corrupt:
    data_list = data.get_list(0, min_features=args.min_features, max_features=args.max_features, 
                              min_instances=args.min_instances, max_instances=args.max_instances,
                              test=lambda x, m: x == m)
    data_list["task_type"] = ["Supervised Classification" if x > 0 else "Supervised Regression" for x in data_list.NumberOfClasses]
    data_list = data_list.reset_index().sort_values(by=['NumberOfInstances'])[['did', 'task_type', 'name', 'NumberOfFeatures', 'NumberOfInstances','NumberOfClasses', 'NumberOfNumericFeatures', 'NumberOfSymbolicFeatures', 'NumberOfInstancesWithMissingValues', 'NumberOfMissingValues']]
    data_list.to_csv("results/openml/corrupted_tasklist.csv")
else:
    data_list = data.get_list(args.min_p_missing, min_features=args.min_features, max_features=args.max_features, 
                              min_instances=args.min_instances, max_instances=args.max_instances,
                              test=lambda x, m: x > m)
    data_list["task_type"] = ["Supervised Classification" if x > 0 else "Supervised Regression" for x in data_list.NumberOfClasses]
    data_list = data_list.reset_index().sort_values(by=['NumberOfInstances'])[['did', 'task_type', 'name', 'NumberOfFeatures', 'NumberOfInstances','NumberOfClasses', 'NumberOfNumericFeatures', 'NumberOfSymbolicFeatures', 'NumberOfInstancesWithMissingValues', 'NumberOfMissingValues']]
    data_list.to_csv("results/openml/noncorrupted_tasklist.csv")

print(data_list)
