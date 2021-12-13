import pickle
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd

path = "../results/openml/hyperparams"
results = [join(path, f) for f in listdir(path) if isfile(join(path, f))]

for result in results:
    with open(result, 'rb') as f:
        y = pickle.load(f)
    # print(result, y[np.argmin([l[1] for l in y])], [l[1] for l in y])
    print(result, y["max"])

# data_list = pd.read_csv("/home/jahan/missing/results/openml/noncorrupted_tasklist.csv")
# print(data_list)
