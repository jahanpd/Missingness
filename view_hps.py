import pickle
from os import listdir
from os.path import isfile, join

path = "results/openml/hyperparams"
results = [join(path, f) for f in listdir(path) if isfile(join(path, f))]

for result in results:
    with open(result, 'rb') as f:
        y = pickle.load(f)
    print(y)
