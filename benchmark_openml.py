# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import argparse
import pickle
from jax.interpreters.batching import batch
import numpy as np
import pandas as pd
from scipy.stats import t
import jax
import jax.numpy as jnp
from jax.experimental.optimizers import exponential_decay
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from imblearn.over_sampling import RandomOverSampler
import UAT.datasets as data
from UAT import UAT, create_early_stopping
from UAT import binary_cross_entropy, cross_entropy, mse
from UAT.aux import oversampled_Kfold
from UAT.training.lr_schedule import attention_lr
# import xgboost as xgb
import lightgbm as lgb
from tqdm import tqdm
import itertools
from os import listdir
from os.path import isfile, join
from numba import cuda
from bayes_opt import BayesianOptimization
from bayes_opt import SequentialDomainReductionTransformer



devices = jax.local_device_count()
# xgb.set_config(verbosity=0)

def create_make_model(features, rows, task, key):
    def make_model(
            d_model,
            batch_size,
            lr_param,
            reg,
            X_valid,
            y_valid
        ):
        print("dims: {}, lr_param: {}, reg: {}".format(d_model, lr_param, reg))
        batch_size_base2 = 2 ** int(np.round(np.log2(batch_size)))
        model_kwargs_uat = dict(
                features=features,
                d_model=int(np.round(d_model)),
                embed_hidden_size=64,
                embed_hidden_layers=2,
                embed_activation=jax.nn.leaky_relu,
                encoder_layers=2,
                encoder_heads=4,
                enc_activation=jax.nn.leaky_relu,
                decoder_layers=4,
                decoder_heads=4,
                dec_activation=jax.nn.leaky_relu,
                net_hidden_size=64,
                net_hidden_layers=2,
                net_activation=jax.nn.leaky_relu,
                last_layer_size=32,
                out_size=classes,
                W_init = jax.nn.initializers.glorot_uniform(),
                b_init = jax.nn.initializers.normal(1e-5),
                )
        steps_per_epoch = rows // batch_size_base2
        max_steps = 1e8
        epochs = int(max_steps // steps_per_epoch)
        start_steps = steps_per_epoch * 5 # wait at least 5 epochs before early stopping
        stop_steps_ = 200
  
        early_stopping = create_early_stopping(start_steps, stop_steps_, metric_name="loss", tol=1e-8)
        training_kwargs_uat = dict(
                    optim="adam",
                    frequency=20,
                    batch_size=batch_size_base2,
                    lr=attention_lr(lr_param, 1000),
                    epochs=epochs,
                    early_stopping=early_stopping
                )
        if task == "Supervised Classification":
            loss_fun = cross_entropy(classes, l2_reg=10.0**-reg, dropout_reg=1e-5) 
        elif task == "Supervised Regression":
            loss_fun = mse(l2_reg=10.0**-reg, dropout_reg=1e-5)
        training_kwargs_uat["X_test"] = X_valid
        training_kwargs_uat["y_test"] = y_valid 
        model = UAT(
            model_kwargs=model_kwargs_uat,
            training_kwargs=training_kwargs_uat,
            loss_fun=loss_fun,
            rng_key=key,
            )
        return model, batch_size_base2, loss_fun
    return make_model


def run(
    dataset=61,  # iris
    task="Supervised Classification",
    missing=None,
    imputation=None,
    train_complete=True,
    test_complete=True,
    epochs=10,
    prop=0.35,
    rng_init=12345,
    trans_params = None,
    gbm_params =  None,
    corrupt = False
    ):
    """ 
    dataset: int, referring to an OpenML dataset ID
    task: str, one of "Supervised Classification" or "Supervised Regression"
    target: str, colname of target variable
    missing: str, one of "MCAR", "MAR", "MNAR" to define missingness pattern if corrupting data, "None" if not
    train_complete: bool, whether the training set is to be complete if corrupting data
    test_complete: bool, whether the test set is to be complete if corrupting data



    strategy: str, one of "Simple", "Iterative", "Miceforest" method of dealing with missingness
    epochs: int, number of epochs to train model
    """
    metrics = {
        ("accuracy", "full"):[],
        ("accuracy", "drop"):[],
        ("accuracy", "gbmoost"):[],
        ("accuracy", "gbmoost_drop"):[],
        ("nll", "full"):[],
        ("nll", "drop"):[],
        ("nll", "gbmoost"):[],
        ("nll", "gbmoost_drop"):[],
        ("rmse", "full"):[],
        ("rmse", "drop"):[],
        ("rmse", "gbmoost"):[],
        ("rmse", "gbmoost_drop"):[],
    }
    rng = np.random.default_rng(rng_init)

    key = rng.integers(9999)
    # resample argument will randomly oversample training set
    X, y, classes, cat_bin = data.prepOpenML(dataset, task)
    resample = True if classes > 1 else False
    kfolds = oversampled_Kfold(5, key=int(key), n_repeats=1, resample=resample)
    splits = kfolds.split(X, y)
    count = 0
    # turn off verbosity for LGB
    gbm_params['verbosity']=-1
    for train, test in splits:
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
                prop=prop,
                corrupt=corrupt
            )
        count += 1
        print("key: {}, k: {}/{}, dataset: {}, missing: {}, impute: {}".format(key, count, len(splits), dataset, missing, imputation))
        # import dataset
        key = rng.integers(9999)
        make_model = create_make_model(X_train.shape[1], X_train.shape[0], task, key)
        print(trans_params)
        model, batch_size_base2, loss_fun = make_model(
                trans_params["d_model"], trans_params["batch_size"], trans_params["lr_param"], trans_params["reg"], X_valid, y_valid)
                
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
        
        model.fit(X_train, y_train)
        # XGBoost comparison
        # build XGBoost model
        # dtrain = xgb.DMatrix(X_train, label=y_train)
        dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=list(np.argwhere(cat_bin == 1)), free_raw_data=False)
        # dvalid = xgb.DMatrix(X_valid, label=y_valid)
        dvalid = lgb.Dataset(X_valid, label=y_valid, categorical_feature=list(np.argwhere(cat_bin == 1)), reference=dtrain, free_raw_data=False)
        # dtest = xgb.DMatrix(X_test)
        dtest = lgb.Dataset(X_test, label=y_test, categorical_feature=list(np.argwhere(cat_bin == 1)), free_raw_data=False)
        evallist = [(dvalid, 'eval'), (dtrain, 'train')]
        num_round = 1000
        print("training gbmoost for {} epochs".format(num_round))
        for k in ["max_depth", "max_bin"]:
            gbm_params[k] = int(gbm_params[k])
        bst = lgb.train(gbm_params, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=50, verbose_eval=100)
        output_gbm = bst.predict(X_test)

        # 1. there is no point dropping if the training set is complete
        # 2. there is no point dropping if when dropped the dataset is empty
        # 3. there is no point dropping if we have imputed the training set
        # 4a. if we are corrupting we want to drop when missing is not none OR
        # 4b. if we are not corrupting we want to drop when missing is none
        if (not train_complete) and (not empty) and (imputation is None) and ((missing is not None and corrupt) or (missing is None and not corrupt)) and False:
            if X_train_drop.shape[0] < trans_params["batch_size"]:
                new_bs = 2
                while new_bs <= X_train_drop.shape[0] - new_bs:
                    new_bs *= 2
            else:
                new_bs = trans_params["batch_size"]
            model_drop, batch_size_base2, loss_fun = make_model(
                trans_params["d_model"], new_bs, trans_params["lr_param"], trans_params["reg"], X_valid, y_valid)
            model_drop.fit(X_train_drop, y_train_drop)
            # XGBoost comparison
            # build XGBoost model
            dtrain_drop = lgb.Dataset(X_train_drop, label=y_train_drop, categorical_feature=list(np.argwhere(cat_bin == 1)), free_raw_data=False)
            # dvalid = xgb.DMatrix(X_valid, label=y_valid)
            dvalid_drop = lgb.Dataset(X_valid_drop, label=y_valid_drop, categorical_feature=list(np.argwhere(cat_bin == 1)), reference=dtrain,
                    free_raw_data=False)
            # dtest = xgb.DMatrix(X_test)
            dtest_drop = lgb.Dataset(X_test_drop, label=y_test_drop, categorical_feature=list(np.argwhere(cat_bin == 1)),free_raw_data=False)
#             train_drop = gbm.DMatrix(X_train_drop, label=y_train_drop)
            # dvalid_drop = gbm.DMatrix(X_valid_drop, label=y_valid_drop)
#             dtest_drop = gbm.DMatrix(X_test_drop)
            num_round = 1000
            print("training gbmoost for {} epochs".format(num_round))
            bst_drop = lgb.train(
                    gbm_params, dtrain_drop, num_round, valid_sets=[dvalid], early_stopping_rounds=50, verbose_eval=100)
            output_gbm_drop = bst_drop.predict(X_test_drop)
        else:
            empty = True # in order to set dropped metrics to NA

        
        # assess performance of models on test set and store metrics
        # predict prob will output 
        if task == "Supervised Regression":
            output = model.predict(X_test)
            if not train_complete and not empty:
                output_drop = model_drop.predict(X_test_drop)
        elif task == "Supervised Classification":
            output = model.predict_proba(X_test)
            if not train_complete and not empty:
                output_drop = model_drop.predict_proba(X_test_drop)

        # calculate performance metrics
        for rm in relevant_metrics:
            if rm == "accuracy":
                class_o = np.argmax(output, axis=1)
                correct_o = class_o == y_test
                acc = np.sum(correct_o) / y_test.shape[0]
                
                class_x = np.argmax(output_gbm, axis=1)
                correct_x = class_x == y_test
                acc_gbm = np.sum(correct_x) / y_test.shape[0]
                if not train_complete and not empty:
                    class_d = np.argmax(output_drop, axis=1)
                    correct_d = class_d == y_test_drop
                    acc_drop = np.sum(correct_d) / y_test_drop.shape[0]
                    
                    class_d = np.argmax(output_gbm_drop, axis=1)
                    correct_d = class_d == y_test_drop
                    acc_drop_gbm = np.sum(correct_d) / y_test_drop.shape[0]
                else:
                    acc_drop = np.nan
                    acc_drop_gbm = np.nan
                metrics[("accuracy","full")].append(acc)
                metrics[("accuracy","drop")].append(acc_drop)
                metrics[("accuracy","gbmoost")].append(acc_gbm)
                metrics[("accuracy","gbmoost_drop")].append(acc_drop_gbm)
                tqdm.write("strategy:{}, acc full:{}, acc drop:{}, acc gbm: {}".format(imputation, acc, acc_drop, acc_gbm))
            if rm == "nll":
                nll = (- jnp.log(output + 1e-8) * jax.nn.one_hot(y_test, classes)).sum(axis=-1).mean()
                nll_gbm = (- jnp.log(output_gbm + 1e-8) * jax.nn.one_hot(y_test, classes)).sum(axis=-1).mean()
                if not train_complete and not empty:
                    nll_drop = (- jnp.log(output_drop + 1e-8) * jax.nn.one_hot(y_test_drop, classes)).sum(axis=-1).mean()
                    nll_drop_gbm = (- jnp.log(output_gbm_drop + 1e-8) * jax.nn.one_hot(y_test_drop, classes)).sum(axis=-1).mean()
                else:
                    nll_drop = np.nan
                    nll_drop_gbm = np.nan
                metrics[("nll","full")].append(nll)
                metrics[("nll","drop")].append(nll_drop)
                metrics[("nll","gbmoost")].append(nll_gbm)
                metrics[("nll","gbmoost_drop")].append(nll_drop_gbm)
                tqdm.write("strategy:{}, nll full:{}, nll drop:{}, nll xbg:{}".format(imputation, nll, nll_drop, nll_gbm))
            if rm == "rmse":
                rmse = np.sqrt(np.square(output - y_test).mean())
                rmse_gbm = np.sqrt(np.square(output_gbm - y_test).mean())
                if not train_complete and not empty:
                    rmse_drop = np.sqrt(np.square(output_drop - y_test_drop).mean())
                    rmse_gbm_drop = np.sqrt(np.square(output_gbm_drop - y_test_drop).mean())
                else:
                    rmse_drop = np.nan
                    rmse_gbm_drop = np.nan
                metrics[("rmse","full")].append(rmse)
                metrics[("rmse","drop")].append(rmse_drop)
                metrics[("rmse","gbmoost")].append(rmse_gbm)
                metrics[("rmse","gbmoost_drop")].append(rmse_gbm_drop)
                tqdm.write("strategy:{}, rmse full:{}, rmse drop:{}, rmse xbg:{}".format(imputation, rmse, rmse_drop, rmse_gbm))
    
    # convert metrics dict to dataframe and determine % change
    # get rid of unused metrics
    dict_keys = list(metrics.keys())
    
    for k in dict_keys:
        if len(metrics[k]) == 0:
            _ = metrics.pop(k, None)

    metrics = pd.DataFrame(metrics)
    metrics.columns = pd.MultiIndex.from_tuples(metrics.columns, names=['metric','dataset'])
    
    # iterate over metrics to determing % change for each metric
    metrics_list = list(metrics.columns.levels[0])
    for m in metrics_list:
        metrics[m, "delta"] = (metrics[m, "full"].values - metrics[m, "drop"].values
        ) / (metrics[m, "full"].values + metrics[m, "drop"].values) * 100
    metrics = metrics.sort_index(axis=1)

    return metrics, perc_missing


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
    parser.add_argument("--iters", type=int, default=15)
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

    data_list = data_list.reset_index()
    print(data_list)
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

    for row in data_list[['did', 'task_type', 'name']].values[selection,:]:
        # BAYESIAN HYPERPARAMETER  SEARCH
        # will search if cannot load params from file
        print(row[1], row[2])
        key = rng.integers(9999)
        X, y, classes, cat_bin = data.prepOpenML(row[0], row[1])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=key)
        key = rng.integers(9999)
        X_train, X_test, X_valid, y_train, y_test, y_valid, diagnostics = data.openml_ds(
                X_train,
                y_train,
                X_test,
                y_test,
                row[1],
                cat_bin=cat_bin,
                classes=classes,
                missing=None,
                imputation=None,  # one of none, simple, iterative, miceforest
                train_complete=False,
                test_complete=True,
                split=0.2,
                rng_key=key,
                prop=args.p,
                corrupt=args.corrupt
            )
        key = rng.integers(9999)
        if row[1] == "Supervised Classification":
            objective = 'softmax'
            X_train, y_train = ros.fit_resample(X_train, y_train)
        else:
            objective = 'regression'
            resample=False
        ## set up transformer model grid search
        path = "results/openml/hyperparams"
        filenames = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
        subset = [f for f in filenames if row[2] in f]
        
        # attempt to get params from file
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

        if not loaded_hps_trans:
            # implement bayesian hyperparameter optimization with sequential domain reduction
            key = rng.integers(9999)
            make_model = create_make_model(X_train.shape[1], X_train.shape[0], row[1], key)
            def black_box(d_model, batch_size, lr_param, reg):
                model, batch_size_base2, loss_fun = make_model(d_model, batch_size, lr_param, reg, X_valid, y_valid)
                model.fit(X_train, y_train)
                # break test into 'batches' to avoid OOM errors
                test_mod = X_test.shape[0] % batch_size_base2
                test_rows = np.arange(X_test.shape[0] - test_mod)
                test_batches = np.split(test_rows,
                            np.maximum(1, X_test.shape[0] // batch_size_base2))

                loss_loop = 0
                pbar1 = tqdm(total=len(test_batches), position=0, leave=False)
                @jax.jit
                def loss_calc(params, x_batch, y_batch, rng):
                    out = model.apply_fun(params, x_batch, rng, False)
                    loss, _ = loss_fun(params, out, y_batch)
                    return loss
                for tbatch in test_batches:
                    key_ = jnp.ones((X_test[np.array(tbatch), ...].shape[0], 2))
                    loss = loss_calc(model.params, X_test[np.array(tbatch), ...], y_test[np.array(tbatch)], key_)
                    loss_loop += loss
                    pbar1.update(1)
                # average metrics over test batches
                return - loss_loop / len(test_batches)

            pbounds={
                    "d_model":(16,128),
                    "batch_size":(8, 128),
                    "lr_param":(100, 10000),
                    "reg":(3, 12)
                    }
        
            bounds_transformer = SequentialDomainReductionTransformer()
            key = rng.integers(9999)
            mutating_optimizer = BayesianOptimization(
                f=black_box,
                pbounds=pbounds,
                verbose=0,
                random_state=int(key),
                bounds_transformer=bounds_transformer
            )
            mutating_optimizer.probe(params={"d_model":64, "batch_size":64, "lr_param":3000, "reg":6})
            mutating_optimizer.maximize(init_points=1, n_iter=args.iters)
            print(mutating_optimizer.res)
            print(mutating_optimizer.max)
            trans_results = {"max": mutating_optimizer.max, "all":mutating_optimizer.res}
            
            with open('results/openml/hyperparams/{},trans_hyperparams.pickle'.format(row[2]), 'wb') as handle:
                pickle.dump(trans_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(classes, objective, row[1])    
        if not loaded_hps_gbm:
            def black_box_gbm(max_depth, learning_rate, max_bin):
                param = {'objective':objective, 'num_class':classes}
                param['max_depth'] = int(max_depth)
                param['num_leaves'] = int(0.8 * (2**max_depth))
                param['learning_rate'] = learning_rate
                param['max_bin']=int(max_bin)
                param['verbosity']=-1
                dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=list(np.argwhere(cat_bin == 1)))
                history = lgb.cv(
                    params=param,
                    train_set=dtrain,
                    num_boost_round=1000,
                    nfold=5,
                    early_stopping_rounds=50,
                    stratified=False,
                    categorical_feature=list(np.argwhere(cat_bin == 1))
                    )
                loss = np.mean(history[list(history.keys())[0]])
                print(loss)
                return - loss
            pbounds_gbm={
                    "max_depth":(3,12),
                    "learning_rate":(0.001, 1),
                    "max_bin":(10, 100)
                    }
        
            key = rng.integers(9999)
            mutating_optimizer_gbm = BayesianOptimization(
                f=black_box_gbm,
                pbounds=pbounds_gbm,
                verbose=0,
                random_state=int(key),
            )
            mutating_optimizer_gbm.maximize(init_points=5, n_iter=args.iters)
            print(mutating_optimizer_gbm.res)
            print(mutating_optimizer_gbm.max)
            best_params_gbm = mutating_optimizer_gbm.max
            best_params_gbm["params"]["objective"]=objective
            best_params_gbm["params"]["num_class"]=classes
            best_params_gbm["params"]['num_leaves'] = int(0.8 * (2**best_params_gbm["params"]["max_depth"]))
            gbm_results = {"max": best_params_gbm, "all":mutating_optimizer_gbm.res} 

            with open('results/openml/hyperparams/{},gbm_hyperparams.pickle'.format(row[2]), 'wb') as handle:
                pickle.dump(gbm_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
 
        # BOOTSTRAP PERFORMANCE if file not already present
        path = "results/openml"
        result_files = [f for f in listdir(path) if isfile(join(path, f))]
        def result_exists(filename, ds, mis, imp):
            splt = filename[:-7].split(",")
            try:
                return ds == splt[0] and str(mis) == splt[2] and str(imp) == splt[3]
            except:
                return False
        for missing in missing_list:
            if missing == "None":
                missing = None
            for imputation in args.imputation:
                # try:
                # we do not want to impute data if there is None missing and data is not corrupted
                if imputation != "None" and missing is None and args.corrupt:
                    continue
                # if results file already exists then skip
                sub = [f for f in result_files if result_exists(f, row[2], missing, imputation)]
                if len(sub) > 0:
                    continue

                if imputation == "None":
                    imputation = None

                m1, perc_missing = run(
                    dataset=row[0],
                    task=row[1],
                    missing=missing,
                    train_complete=args.train_complete,
                    test_complete=args.test_complete,
                    imputation=imputation,
                    epochs=args.epochs,
                    prop=args.p,
                    trans_params = trans_results["max"]["params"],
                    gbm_params =  gbm_results["max"]["params"],
                    corrupt=args.corrupt
                    )
                print(row[2], missing, imputation)
                print(m1.mean())
                if args.save:
                    m1.to_pickle("results/openml/{},{:2f},{},{},{},{}.pickle".format(
                    row[2], perc_missing, str(missing), str(imputation), args.test_complete, args.corrupt))

                # except Exception as e:
                #     print(e)
