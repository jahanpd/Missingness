import pandas as pd
from os import listdir
from os.path import isfile, join

path = "results/imputation"
results = [join(path, f) for f in listdir(path) if isfile(join(path, f))]

for f in results:
    data = pd.read_pickle(f)
    print(data.mean())