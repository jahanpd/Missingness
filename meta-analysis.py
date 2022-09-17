import copy
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# import data
base = ['dsname', 'missingness', 'imputation', 'nll_lsam', 'acc_lsam', 'nll_gbm', 'acc_gbm']
ccols = ['NumberOfFeatures', 'NumberOfInstances', 'NumberOfClasses', 'NumberOfNumericFeatures', 'NumberOfSymbolicFeatures']
ncols = ['NumberOfFeatures', 'NumberOfInstances', 'NumberOfClasses', 'NumberOfNumericFeatures', 'NumberOfSymbolicFeatures', 'NumberOfMissingValues']
corrupt = pd.read_csv('./results/openml/benchmark_results_corrupt.csv')
noncorrupt = pd.read_csv('./results/openml/benchmark_results_noncorrupt.csv')
cdata = pd.read_csv("results/openml/corrupted_tasklist.csv")
cdata.rename(columns={'name':'dsname'}, inplace=True)
ndata = pd.read_csv("results/openml/noncorrupted_tasklist.csv")
ndata.rename(columns={'name':'dsname'},inplace=True)
corrupt = corrupt.merge(cdata, on='dsname')[base+ccols]
noncorrupt = noncorrupt.merge(ndata, on='dsname')[base+ncols]
# add derived cols
noncorrupt['FractionMissingValues'] = noncorrupt.NumberOfMissingValues / (noncorrupt.NumberOfFeatures * noncorrupt.NumberOfInstances)
noncorrupt["NumericRatio"]=noncorrupt.NumberOfNumericFeatures.values / noncorrupt.NumberOfFeatures.values
noncorrupt["FeatureInstanceRatio"]=noncorrupt.NumberOfFeatures.values / noncorrupt.NumberOfInstances.values
corrupt["NumericRatio"]=corrupt.NumberOfNumericFeatures.values / corrupt.NumberOfFeatures.values
corrupt["FeatureInstanceRatio"]=corrupt.NumberOfFeatures.values / corrupt.NumberOfInstances.values
# dummify missingness type
corrupt = pd.get_dummies(corrupt, columns=['missingness'])
# redefine predictor col arrays
ncols = ['NumberOfFeatures', 'NumberOfInstances', 'NumberOfClasses', 'NumberOfNumericFeatures', 'NumberOfSymbolicFeatures', 'NumericRatio', 'FeatureInstanceRatio', 'FractionMissingValues']
ccols = ['NumberOfFeatures', 'NumberOfInstances', 'NumberOfClasses', 'NumberOfNumericFeatures', 'NumberOfSymbolicFeatures', 'NumericRatio', 'FeatureInstanceRatio', 'missingness_MAR', 'missingness_MCAR', 'missingness_MNAR']
allfeat = ['NumberOfFeatures', 'NumberOfInstances', 'NumberOfClasses', 'NumberOfNumericFeatures', 'NumberOfSymbolicFeatures', 'NumericRatio', 'FeatureInstanceRatio', 'FractionMissingValues', 'missingness_MAR', 'missingness_MCAR', 'missingness_MNAR']
print(noncorrupt[['dsname', 'FractionMissingValues']])
print(list(corrupt))

# prep dict to store analysis
store_corrupted = {
    ("", "Dataset Characteristic"):ccols,
    ("Overall: LightGBM vs LSAM", "Accuracy"):[],
    ("Overall: LightGBM vs LSAM", "NLL"):[],
    ("LSAM: Simple vs None", "Accuracy"):[],
    ("LSAM: Simple vs None", "NLL"):[],
    ("LSAM: Iterative vs None", "Accuracy"):[],
    ("LSAM: Iterative vs None", "NLL"):[],
    ("LSAM: Miceforest vs None", "Accuracy"):[],
    ("LSAM: Miceforest vs None", "NLL"):[],
}

store_noncorrupted = copy.deepcopy(store_corrupted)
store_noncorrupted[("", "Dataset Characteristic")] = ncols

# define feature importance function
def feature_importance(X, y, names):
    rf = RandomForestClassifier()
    rf.fit(X, y.flatten())
    return sorted(list(zip(rf.feature_importances_,names)), key=lambda x: x[0], reverse=True)

def add_to_dict(sdict, slist, idx):
    keys = list(sdict.keys())
    for char in sdict[keys[0]]:
        for i, (fi, name) in enumerate(slist):
            if name == char:
                sdict[keys[idx]].append(fi)
        
# LightGBM vs LSAM
## corrupted data
### NLL
X = corrupt[ccols]
y = (corrupt.nll_lsam < corrupt.nll_gbm).astype(float)
fi = feature_importance(X.values, y.values, list(X))
add_to_dict(store_corrupted, fi, 2)
X = corrupt[ccols]
y = (corrupt.acc_lsam > corrupt.acc_gbm).astype(float)
fi = feature_importance(X.values, y.values, list(X))
add_to_dict(store_corrupted, fi, 1)

## noncorrupted data
X = noncorrupt[ncols]
y = (noncorrupt.nll_lsam < noncorrupt.nll_gbm).astype(float)
fi = feature_importance(X.values, y.values, list(X))
add_to_dict(store_noncorrupted, fi, 2)
X = noncorrupt[ncols]
y = (noncorrupt.acc_lsam > noncorrupt.acc_gbm).astype(float)
fi = feature_importance(X.values, y.values, list(X))
add_to_dict(store_noncorrupted, fi, 1)

# function to deal with repeated pattern
def get_fi_lsam(ds, imp):
    if "missingness_None" in list(ds):
        t = ds.loc[
            (ds.imputation == imp) &
            (ds.missingness_None == 0)
            ]
        n = ds.loc[
            (ds.imputation == 'None') &
            (ds.missingness_None == 0)
            ]
        X = t[ccols]
    else:
        t = ds.loc[
            (ds.imputation == imp) 
            ]
        n = ds.loc[
            (ds.imputation == 'None')
            ]
        X = t[ncols]

    y = (n.acc_lsam.values > t.acc_lsam.values).astype(float)
    fi_acc = feature_importance(X.values, y, list(X))
    y = (n.nll_lsam.values < t.nll_lsam.values).astype(float)
    fi_nll = feature_importance(X.values, y, list(X))
    return fi_acc, fi_nll

# LSAM None vs Simple
## corrupted data
fi_acc, fi_nll = get_fi_lsam(corrupt, "simple")
add_to_dict(store_corrupted, fi_acc, 3)
add_to_dict(store_corrupted, fi_nll, 4)

## noncorrupted data
fi_acc, fi_nll = get_fi_lsam(noncorrupt, "simple")
add_to_dict(store_noncorrupted, fi_acc, 3)
add_to_dict(store_noncorrupted, fi_nll, 4)

# LSAM None vs Iterative
## corrupted data
fi_acc, fi_nll = get_fi_lsam(corrupt, "iterative")
add_to_dict(store_corrupted, fi_acc, 5)
add_to_dict(store_corrupted, fi_nll, 6)

## noncorrupted data
fi_acc, fi_nll = get_fi_lsam(noncorrupt, "iterative")
add_to_dict(store_noncorrupted, fi_acc, 5)
add_to_dict(store_noncorrupted, fi_nll, 6)

# LSAM None vs Miceforest
## corrupted data
fi_acc, fi_nll = get_fi_lsam(corrupt, "miceforest")
add_to_dict(store_corrupted, fi_acc, 7)
add_to_dict(store_corrupted, fi_nll, 8)

## noncorrupted data
fi_acc, fi_nll = get_fi_lsam(noncorrupt, "miceforest")
add_to_dict(store_noncorrupted, fi_acc, 7)
add_to_dict(store_noncorrupted, fi_nll, 8)

nondf = pd.DataFrame(store_noncorrupted)
snon = nondf.style.hide().format(precision=2).highlight_max(axis=0, props="textbf:--rwrap;")
print(nondf)
print(snon.to_latex())

corrdf = pd.DataFrame(store_corrupted)
cnon = corrdf.style.hide().format(precision=2).highlight_max(axis=0, props="textbf:--rwrap;")
print(corrdf)
print(cnon.to_latex())
