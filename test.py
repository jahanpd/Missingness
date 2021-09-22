import numpy as np
import pandas as pd
from scipy.stats import t
import jax
import jax.numpy as jnp
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.datasets import fetch_openml
import UAT.datasets as data
import xgboost as xgb

# test imputation script
# for missing in [None, "MCAR", "MAR", "MNAR"]:
#     for imputation in [None]: #, "simple", "iterative", "miceforest"]:
#         X_train, X_valid, X_test, y_train, y_valid, y_test, (x_a, x_b), classes = data.spiral(
#             1000,
#             missing=missing,
#             imputation=imputation,  # one of none, simple, iterative, miceforest
#             train_complete=False,
#             test_complete=False,
#             split=0.33,
#             rng_key=0,
#             p=0.3,
#             cols_miss=100
#         )
#         print("spiral", missing, imputation, X_train.shape, X_valid.shape, X_test.shape)


task_list = data.get_list(0.2, 4, key=42, test=lambda x, m: x > m)
task_list["task_type"] = ["Supervised Classification" if x > 0 else "Supervised Regression" for x in task_list.NumberOfClasses]
# task_list = data.get_tasklist(0.5, 5, key=42, test=lambda x, m: x > m)
# task_list = data.get_list(0, 3, key=24, test=lambda x, m: x == m)
# task_list["task_type"] = ["Supervised Classification" if x > 0 else "Supervised Regression" for x in task_list.NumberOfClasses]
print(task_list)

for row in task_list[['did', 'task_type', 'name']].values[4:, :]:
    for missing in [None]:
    # for missing in [None, "MCAR", "MAR", "MNAR"]:
    # for missing in ["MCAR", "MAR", "MNAR"]:
        for imputation in ["simple"]: #, "iterative", "miceforest"]:
            X_train, X_valid, X_test, y_train, y_valid, y_test, diagnostics, classes = data.openml_ds(
                row[0],
                row[1],
                missing=missing,
                imputation=imputation,  # one of none, simple, iterative, miceforest
                train_complete=False,
                test_complete=False,
                split=0.33,
                rng_key=0,
                prop=0.35,
                cols_miss=100
            )
            print(diagnostics)
            print("{}".format(row[2]), missing, imputation, X_train.shape, X_valid.shape, X_test.shape)

# for missing in [None, "MCAR", "MAR", "MNAR"]:
#     for imputation in [None, "simple"]: # "iterative", "miceforest"]:
#         X_train, X_valid, X_test, y_train, y_valid, y_test, classes = data.thoracic(
#             missing=missing,
#             imputation=imputation,  # one of none, simple, iterative, miceforest
#             train_complete=False,
#             test_complete=False,
#             split=0.2,
#             rng_key=0,
#             p=0.15,
#             cols_miss=100
#         )
#         print("thoracic", missing, imputation, X_train.shape, X_valid.shape, X_test.shape)

# for missing in [None, "MCAR", "MAR", "MNAR"]:
#     for imputation in [None, "simple"]: #, "iterative", "miceforest"]:
#         X_train, X_valid, X_test, y_train, y_valid, y_test, classes = data.abalone(
#             missing=missing,
#             imputation=imputation,  # one of none, simple, iterative, miceforest
#             train_complete=False,
#             test_complete=False,
#             split=0.33,
#             rng_key=0,
#             p=0.1,
#             cols_miss=100
#         )
#         print("abalone", missing, imputation, X_train.shape, X_valid.shape, X_test.shape)

# for imputation in [None, "simple", 'iterative', "miceforest"]:
#     X_train, X_valid, X_test, y_train, y_valid, y_test, classes = data.anneal(
#         imputation=imputation,  # one of none, simple, iterative, miceforest
#         rng_key=1
#     )
    # param = {'objective':'multi:softprob', 'num_class':classes}
    # param['max_depth'] = 2
    # param['eta'] = 1
    # dtrain = xgb.DMatrix(X_train, label=y_train)
    # dvalid = xgb.DMatrix(X_valid, label=y_valid)
    # dtest = xgb.DMatrix(X_test)
    # evallist = [(dvalid, 'eval'), (dtrain, 'train')]
    # num_round = 100
    # bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=10)
    # output_xgb = bst.predict(dtest)
    # print("anneal", imputation, X_train.shape, X_valid.shape, X_test.shape)

# for imputation in [None, "simple", "iterative", "miceforest"]:
#     X_train, X_valid, X_test, y_train, y_valid, y_test, classes = data.banking(
#         imputation=imputation,  # one of none, simple, iterative, miceforest
#         rng_key=0
#     )
#     print("banking", imputation, X_train.shape, X_valid.shape, X_test.shape)

# for imputation in [None, "simple", "iterative", "miceforest"]:
#     for missing in [None, "MCAR", "MAR"]:
#         X_train, X_valid, X_test, y_train, y_valid, y_test, classes = data.mnist(
#             missing=missing,
#             imputation=imputation,  # one of none, simple, iterative, miceforest
#             train_complete=False,
#             test_complete=False,
#             split=0.33,
#             rng_key=0,
#             p=0.5
#         )
#         print("mnist", imputation, X_train.shape, X_valid.shape, X_test.shape)