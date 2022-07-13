from LSAM.aux import oversampled_Kfold
import LSAM.datasets as data
from .makemodel import create_make_model
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
import lightgbm as lgb
import jax
import jax.numpy as jnp
import sys

devices = jax.local_device_count()
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
  from tqdm.notebook import tqdm
else:
  from tqdm import tqdm

def run(
    dataset=61,  # iris
    task="Supervised Classification",
    missing=None,
    imputation=None,
    train_complete=True,
    test_complete=True,
    rng_init=12345,
    lsam_params = None,
    gbm_params =  None,
    corrupt = False,
    row_data=None,
    folds=5,
    repeats=1,
    cols_miss=0.9,
    perc_missing=0.8,
    wandb=None
    ):
    """
    Args:
        dataset: int, referring to an OpenML dataset ID
        task: str, one of "Supervised Classification" or "Supervised Regression"
        missing: str, one of "None" "MCAR", "MAR", "MNAR" to define missingness pattern if corrupting data
        imputation: str, one of "Simple", "Iterative", "Miceforest" method of dealing with missingness
        train_complete: bool, whether the training set is to be complete if corrupting data
        test_complete: bool, whether the test set is to be complete if corrupting data
        rng_init: int, initial rng key
        lsam_params: dict, hyperparams for the transformer
        gbm_params: dict, hyperparams for the GBM (light gbm)
        corrupt: bool, whether we are corrupting the OpenML dataset with missingness
        row_data: optional basic information about the dataset for printing/debugging,
        folds: int, number for K folds,
        repeats: int, number of repeats for validation 
    Returns:
        Pandas Dataframe: performance data
        Float: percentage missing (for diagnostic purposes)
    """
    metrics = {
        ("accuracy", "lsam"):[],
        ("accuracy", "gbm"):[],
        ("nll", "lsam"):[],
        ("nll", "gbm"):[],
        ("rmse", "lsam"):[],
        ("rmse", "gbm"):[],
    }
    rng = np.random.default_rng(rng_init)

    key = rng.integers(9999)
    # resample argument will randomly oversample training set
    X, y, classes, cat_bin = data.prepOpenML(dataset, task)
    resample = True if classes > 1 else False
    kfolds = oversampled_Kfold(folds, key=int(key), n_repeats=repeats, resample=resample)
    splits = kfolds.split(X, y)
    count = 0

    for train, test in splits:
        if row_data is not None:
            print("name: {}, features: {}".format(row_data[2], row_data[3]))
        key = rng.integers(9999)

        X_train, X_test, X_valid, y_train, y_test, y_valid, diagnostics = data.openml_ds(
                X[train,:],
                y[train],
                X[test,:],
                y[test],
                task,
                cat_bin=cat_bin,
                classes=classes,
                missing=missing,
                imputation=imputation,  # one of none, simple, iterative, miceforest
                train_complete=train_complete,
                test_complete=test_complete,
                split=0.2,
                rng_key=key,
                prop=perc_missing,
                corrupt=corrupt,
                cols_miss=int(X.shape[1] * cols_miss)
            )
        print(diagnostics)
        count += 1

        # equalise training classes if categorical
        if task == "Supervised Classification":
            relevant_metrics = ["accuracy", "nll"]
            key = rng.integers(9999)
            ros = RandomOverSampler(random_state=key)
            X_train, y_train = ros.fit_resample(X_train, y_train)
        else:
            relevant_metrics = ["rmse"]

        # sanity check
        if imputation is not None:
            print("assertion is ", np.all(~np.isnan(X_train)))
            assert np.all(~np.isnan(X_train))

        # create dropped dataset baseline and implement missingness strategy
        def drop_nans(xarray, yarray):
            row_mask = ~np.any(np.isnan(xarray), axis=1)
            xdrop = xarray[row_mask, :]
            ydrop = yarray[row_mask]
            return xdrop, ydrop, 1.0 - (np.sum(row_mask) / len(row_mask))

        X_train_drop, y_train_drop, perc_missing = drop_nans(X_train, y_train)
        X_test_drop, y_test_drop, _ = drop_nans(X_test, y_test)
        X_valid_drop, y_valid_drop, _ = drop_nans(X_valid, y_valid)

        print("dataset sizes")
        print(X_train.shape, X_valid.shape, X_test.shape)
        if (len(y_train_drop) < devices) or (len(y_test_drop) < devices) or (len(y_valid_drop) < devices):
            print("no rows left in dropped dataset...")
            empty=True
        else:
            print("dropped dataset sizes")
            print(X_train_drop.shape, X_valid_drop.shape, X_test_drop.shape)
            empty=False
        key = rng.integers(9999)
        print("key: {}, k: {}/{}, dataset: {}, missing: {}, impute: {}".format(key, count, len(splits), dataset, missing, imputation))
        # MAKE AND TRAIN lsam
        if lsam_params is not None:
            make_model = create_make_model(X_train.shape[1], X_train.shape[0], task, key)
            print(lsam_params)
            model, batch_size_base2, loss_fun = make_model(
                    X_valid=X_valid, y_valid=y_valid, classes=classes,
                    **lsam_params
            )

            model.fit(X_train, y_train)
            # assess performance of models on test set and store metrics
            # predict prob will output 
            if task == "Supervised Regression":
                output = model.predict(X_test, sample=False)
                # output = model.predict(X_test)
            elif task == "Supervised Classification":
                output = model.predict_proba(X_test, sample=False)
                # output = model.predict_proba(X_test)
            # calculate performance metrics
            for rm in relevant_metrics:
                if rm == "accuracy":
                    class_o = np.argmax(output, axis=1)
                    correct_o = class_o == y_test
                    acc = np.sum(correct_o) / y_test.shape[0]
                    metrics[("accuracy","lsam")].append(acc)
                    if wandb is not None:
                        wandb.log({"lsam_accuracy_fold{}".format(count):acc}, commit=False)
                    tqdm.write("strategy:{}, acc lsam:{}".format(imputation, acc))
                if rm == "nll":
                    nll = -(jnp.log(output + 1e-8) * jax.nn.one_hot(y_test, classes) +
                            jnp.log(1 - output + 1e-8) * jax.nn.one_hot(1 - y_test, classes)
                            ).sum(axis=-1).mean()
                    metrics[("nll","lsam")].append(nll)
                    if wandb is not None:
                        wandb.log({"lsam_nll_fold{}".format(count):nll}, commit=False)
                    tqdm.write("strategy:{}, nll lsam:{}".format(imputation, nll))
                if rm == "rmse":
                    rmse = np.sqrt(np.square(output - y_test).mean())
                    metrics[("rmse","lsam")].append(rmse)
                    if wandb is not None:
                        wandb.log({"lsam_rmse_fold{}".format(count):rmse}, commit=False)
                    tqdm.write("strategy:{}, rmse lsam:{}".format(imputation, rmse))

        if gbm_params is not None:
            # turn off verbosity for LGB
            gbm_params['verbose']=-1
            gbm_params['num_class']=classes
            dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=list(np.argwhere(cat_bin == 1)), free_raw_data=False)
            dvalid = lgb.Dataset(X_valid, label=y_valid, categorical_feature=list(np.argwhere(cat_bin == 1)), reference=dtrain, free_raw_data=False)
            dtest = lgb.Dataset(X_test, label=y_test, categorical_feature=list(np.argwhere(cat_bin == 1)), free_raw_data=False)
            evallist = [(dvalid, 'eval'), (dtrain, 'train')]
            num_round = int(5e3)
            print("training gbm for {} epochs".format(num_round))
            for k in ["max_depth", "max_bin"]:
                gbm_params[k] = int(gbm_params[k])
            gbm_params["learning_rate"] = gbm_params["learning_rate"]
            print(gbm_params)
            bst = lgb.train(
                gbm_params, dtrain, num_round, valid_sets=[dvalid],
                categorical_feature=list(np.argwhere(cat_bin == 1)),
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100),
                    lgb.log_evaluation(1000)
                ]
            )
            output_gbm = bst.predict(X_test)

            # calculate performance metrics
            for rm in relevant_metrics:
                if rm == "accuracy":
                    class_x = np.argmax(output_gbm, axis=1)
                    correct_x = class_x == y_test
                    acc_gbm = np.sum(correct_x) / y_test.shape[0]
                    metrics[("accuracy","gbm")].append(acc_gbm)
                    if wandb is not None:
                        wandb.log({"gbm_accuracy_fold{}".format(count):acc_gbm}, commit=False)
                    tqdm.write("strategy:{}, acc gbm: {}".format(imputation, acc_gbm))
                if rm == "nll":
                    nll_gbm = -(jnp.log(output_gbm + 1e-8) * jax.nn.one_hot(y_test, classes) +
                                jnp.log(1 - output_gbm + 1e-8) * jax.nn.one_hot(1 - y_test, classes)
                                ).sum(axis=-1).mean()
                    metrics[("nll","gbm")].append(nll_gbm)
                    if wandb is not None:
                        wandb.log({"gbm_nll_fold{}".format(count):nll_gbm}, commit=False)
                    tqdm.write("strategy:{}, nll xbg:{}".format(imputation, nll_gbm))
                if rm == "rmse":
                    rmse_gbm = np.sqrt(np.square(output_gbm - y_test).mean())
                    metrics[("rmse","gbm")].append(rmse_gbm)
                    if wandb is not None:
                        wandb.log({"gbm_rmse_fold{}".format(count):rmse_gbm}, commit=False)
                    tqdm.write("strategy:{}, rmse xbg:{}".format(imputation, rmse_gbm))
        
    # convert metrics dict to dataframe and determine % change
    # get rid of unused metrics
    dict_keys = list(metrics.keys())
    
    for k in dict_keys:
        if len(metrics[k]) == 0:
            _ = metrics.pop(k, None)

    metrics = pd.DataFrame(metrics)
    metrics.columns = pd.MultiIndex.from_tuples(metrics.columns, names=['metric','dataset'])
    
    # iterate over metrics to determing % change for each metric
    #metrics_list = list(metrics.columns.levels[0])
    #print(metrics)
    #for m in metrics_list:
    #    metrics[m, "delta"] = (metrics[m, "full"].values - metrics[m, "drop"].values
    #    ) / (metrics[m, "full"].values + metrics[m, "drop"].values) * 100
    metrics = metrics.sort_index(axis=1)

    return metrics, perc_missing
