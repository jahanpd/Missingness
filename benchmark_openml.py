# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import argparse
import pickle
import numpy as np
import jax
import jax.numpy as jnp
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import UAT.datasets as data
import lightgbm as lgb
import itertools
from os import listdir
from os.path import isfile, join
import sys
from benchmarkaux.openmlrun import run
from benchmarkaux.gridsearch import gridsearchgbm, gridsearchattn

IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
  from tqdm.notebook import tqdm
else:
  from tqdm import tqdm

from jax.config import config
config.update("jax_debug_nans", False)

devices = jax.local_device_count()
# xgb.set_config(verbosity=0)

if __name__ ==  "__main__":
    parser = argparse.ArgumentParser("train model")
    parser.add_argument("--missing", choices=["None", "MCAR", "MAR", "MNAR"], default="None", nargs='+')
    parser.add_argument("--imputation", choices=["None", "Drop", "simple", "iterative", "miceforest"], nargs='+')
    parser.add_argument("--epochs", default=10000, type=int)
    parser.add_argument("--dataset", type=int, nargs='+')
    parser.add_argument("--p", default=0.35, type=float)
    parser.add_argument("--corrupt", action='store_true')
    parser.add_argument("--train_complete", action='store_true') # default is false
    parser.add_argument("--test_complete", action='store_false') # default is true
    parser.add_argument("--load_params", action='store_false') # default is true
    parser.add_argument("--save", action='store_true')
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--inverse", action='store_true')
    parser.add_argument("--gbm_gpu", type=int, default=-1)
    args = parser.parse_args()
    
    if args.corrupt:
        data_list = data.get_list(0, key=12, test=lambda x, m: x == m)
        data_list["task_type"] = ["Supervised Classification" if x > 0 else "Supervised Regression" for x in data_list.NumberOfClasses]
        data_list.to_csv("results/openml/corrupted_tasklist.csv")
        missing_list = args.missing
    else:
        data_list = data.get_list(0.2, key=13, test=lambda x, m: x > m)
        data_list["task_type"] = ["Supervised Classification" if x > 0 else "Supervised Regression" for x in data_list.NumberOfClasses]
        data_list.to_csv("results/openml/noncorrupted_tasklist.csv")
        missing_list = ["None"]

    data_list = data_list.reset_index().sort_values(by=['NumberOfInstances'])

    rng = np.random.default_rng(1234)
    key = rng.integers(9999)
    ros = RandomOverSampler(random_state=key)
    class_filter = np.array([x == "Supervised Classification" for x in data_list.task_type])
    if args.dataset is not None:
        selection = np.array(args.dataset)
    else:
        selection = np.arange(len(data_list))[class_filter]
        if args.inverse:
            selection = np.flip(selection)

    print(
        data_list[
            ['name', 'NumberOfInstances', 'NumberOfFeatures', 'NumberOfClasses', 'NumberOfNumericFeatures', 'NumberOfSymbolicFeatures']
            ].loc[data_list.index[selection]])
    for row in data_list[['did', 'task_type', 'name', 'NumberOfFeatures']].values[selection,:]:
        # attempt to get hyperparams from file
        # if empty then perform grid search or bayesian optimisation
        ## set up transformer model hp search
        path = "results/openml/hyperparams"
        filenames = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
        subset = [f for f in filenames if row[2] in f]

        try:
            trans_subset = [f for f in subset if 'trans' in f]
            with (open(trans_subset[0], "rb")) as handle:
                trans_results = pickle.load(handle)
            loaded_hps_trans = True
        except Exception as e:
            loaded_hps_trans = False
        try:
            gbm_subset = [f for f in subset if 'gbm' in f]
            with (open(gbm_subset[0], "rb")) as handle:
                gbm_results = pickle.load(handle)
            loaded_hps_gbm = True
        except Exception as e:
            loaded_hps_gbm = False

        hpk = 4
        if not loaded_hps_trans:
            searchspace = [
                ("d_model",[32, 128]),
                ("lr_max", [5e-3]),
                ("reg", [1e-3, 0]),
                ("start_es", [0.3]),  # start early stopping
                ("depth", [4, 2]),
                ("nndepth", [2,1]),
                # ("nnwidth", [4]), # change to 2**(nndepth + 1)
            ]
            trans_results = gridsearchattn(searchspace, 4, row, args)
            with open('results/openml/hyperparams/{},{},trans_hyperparams.pickle'.format(row[2], "None"), 'wb') as handle:
                    pickle.dump(trans_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if not loaded_hps_gbm:
            searchspace = [
                ("max_depth", [6,12,24]),
                ("learning_rate", [1e-2]),
                ("max_bin", [10, 50])
            ]
            gbm_results = gridsearchgbm(searchspace, 4, row, args)
            with open('results/openml/hyperparams/{},{},gbm_hyperparams.pickle'.format(row[2], "None"),'wb') as handle:
                    pickle.dump(gbm_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        for missing in missing_list:
            # BOOTSTRAP PERFORMANCE if file not already present
            path = "results/openml"
            result_files = [f for f in listdir(path) if isfile(join(path, f))]
            def result_exists(filename, ds, mis, imp):
                splt = filename[:-7].split(",")
                try:
                    return ds == splt[0] and str(mis) == splt[2] and str(imp) == splt[3]
                except:
                    return False
            for imputation in args.imputation:
                # we do not want to impute data if there is None missing and data is not corrupted
                if imputation != "None" and (missing is None or missing =="None") and args.corrupt:
                    continue
                # if results file already exists then skip
                sub = [f for f in result_files if result_exists(f, row[2], missing, imputation)]
                if len(sub) > 0:
                    continue

                key = rng.integers(9999)

                if imputation == "None":
                    imputation = None
                if missing == "None":
                    missing = None
                m1, perc_missing = run(
                    dataset=row[0],
                    task=row[1],
                    missing=missing,
                    train_complete=args.train_complete,
                    test_complete=args.test_complete,
                    imputation=imputation,
                    trans_params = trans_results["best"][2],
                    gbm_params =  gbm_results["best"][2],
                    corrupt=args.corrupt,
                    row_data=row,
                    rng_init=key,
                    folds=4,
                    repeats=1
                    )
                print(row[2], missing, imputation)
                print(m1.mean())
                if args.save:
                    m1.to_pickle("results/openml/{},{:2f},{},{},{},{}.pickle".format(
                    row[2], perc_missing, str(missing), str(imputation), args.test_complete, args.corrupt))

