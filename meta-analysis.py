import numpy as np
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
    ("Overall - LightGBM vs LSAM", "Accuracy"):[],
    ("Overall - LightGBM vs LSAM", "NLL"):[],
    ("LSAM - Simple Imputation vs None", "Accuracy"):[],
    ("LSAM - Simple Imputation vs None", "NLL"):[],
    ("LSAM - Iterative Imputation vs None", "Accuracy"):[],
    ("LSAM - Iterative Imputation vs None", "NLL"):[],
    ("LSAM - Miceforest Imputation vs None", "Accuracy"):[],
    ("LSAM - Miceforest Imputation vs None", "NLL"):[],
}

store_noncorrupted = store_corrupted.copy()
store_noncorrupted[("", "Dataset Characteristic")] = ncols

# define feature importance function
def feature_importance(X, y, names):
    rf = RandomForestClassifier()
    rf.fit(X, y.flatten())
    return sorted(list(zip(rf.feature_importances_,names)), key=lambda x: x[0], reverse=True)

# LightGBM vs LSAM
## corrupted data
### NLL
X = corrupt[ccols]
y = (corrupt.nll_lsam < corrupt.nll_gbm).astype(float)
fi = feature_importance(X.values, y.values, list(X))
print('NLL')
print(fi)
X = corrupt[ccols]
y = (corrupt.acc_lsam > corrupt.acc_gbm).astype(float)
fi = feature_importance(X.values, y.values, list(X))
print('Accuracy')
print(fi)

## noncorrupted data
X = noncorrupt[ncols]
y = (noncorrupt.nll_lsam < noncorrupt.nll_gbm).astype(float)
fi = feature_importance(X.values, y.values, list(X))
print('NLL')
print(fi)
X = noncorrupt[ncols]
y = (noncorrupt.acc_lsam > noncorrupt.acc_gbm).astype(float)
fi = feature_importance(X.values, y.values, list(X))
print('Accuracy')
print(fi)

# LSAM None vs Simple
## corrupted data
t = corrupt.loc[

    (corrupt.imputation == 'simple') &
    (corrupt.missingness_None == 0)
    ]
n = corrupt.loc[

    (corrupt.imputation == 'None') &
    (corrupt.missingness_None == 0)
    ]
X = t[ccols]
y = (n.acc_lsam.values > t.acc_lsam.values).astype(float)
fi = feature_importance(X.values, y, list(X))
print('Accuracy')
print(fi)
y = (n.nll_lsam.values < t.nll_lsam.values).astype(float)
fi = feature_importance(X.values, y, list(X))
print('NLL')
print(fi)
## noncorrupted data
t = noncorrupt.loc[

    (noncorrupt.imputation == 'simple') 
    ]
n = noncorrupt.loc[

    (noncorrupt.imputation == 'None') 
    ]
X = t[ncols]
y = (n.acc_lsam.values > t.acc_lsam.values).astype(float)
fi = feature_importance(X.values, y, list(X))
print('Accuracy')
print(fi)
y = (n.nll_lsam.values < t.nll_lsam.values).astype(float)
fi = feature_importance(X.values, y, list(X))
print('NLL')
print(fi)


# LSAM None vs Iterative
## corrupted data

## noncorrupted data

# LSAM None vs Miceforest
## corrupted data

## noncorrupted data



predictors = ["NumberOfFeatures", "NumberOfInstances", "NumberOfClasses", "NumberOfNumericFeatures", "NumberOfSymbolicFeatures", ]
outcomes = ["winner_lsam_acc", "winner_lsam_nll"]
df_hps=pd.DataFrame(store_results)
print(df_hps)
X=df_hps[predictors]
y=df_hps[outcomes]
X["NumericRatio"]=X.NumberOfNumericFeatures.values / X.NumberOfFeatures.values
X["FeatureInstanceRatio"]=X.NumberOfFeatures.values / X.NumberOfInstances.values
rf_acc = RandomForestClassifier()
rf_acc.fit(X.values, y.winner_lsam_acc.values)
acc_feat_importance = sorted(list(zip(rf_acc.feature_importances_,list(X))), key=lambda x: x[0])
rf_nll = RandomForestClassifier()
rf_nll.fit(X.values, y.winner_lsam_nll.values)
nll_feat_importance = sorted(list(zip(rf_nll.feature_importances_,list(X))),key=lambda x: x[0])
print(acc_feat_importance)
print(nll_feat_importance)
