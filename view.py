import pandas as pd
from os import listdir
from os.path import isfile, join
import numpy as np

path = "results/imputation"
results = [join(path, f) for f in listdir(path) if isfile(join(path, f))]

# datasets = ["spiral", "thoracic", "abalone"]
datasets = ["spiral"]
missingness = ["None", "MCAR", "MAR", "MNAR"]
# missingness = ["MAR"]
# imputation = ["None", "Dropped", "Simple", "Iterative", "Miceforest"]
imputation = ["None"]
metrics = ["accuracy", "nll", "rmse"]

multiindex = pd.MultiIndex.from_product([datasets, missingness, imputation], names=["Dataset", "Missingness", "Imputation"])
print(len((multiindex.values)))

multiindex_filter = []
# results = []
for idx in multiindex.values:
    try:
        print(idx)
        fs = [f for f in results if idx[0].lower() in f.lower() and (idx[1]+"_"+idx[2]).lower() in f.lower()]
        data = pd.read_pickle(fs[0])
        # if data.values.shape[1] == 
        print(fs)
        print(data)
    except Exception as e:
        print(e)
        # pass

for f in results:
    data = pd.read_pickle(f)

test = {}
for d in datasets:
    test[d] = {}
    for m in missingness:
        test[d][m] ={}
        for i in imputation:
            test[d][m][i] ={}
            for met in metrics:
                test[d][m][i] = [1,2,3]



# print(pd.DataFrame(test, orient="index"))
