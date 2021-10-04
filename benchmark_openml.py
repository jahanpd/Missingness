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
import xgboost as xgb
from tqdm import tqdm
import itertools
from os import listdir
from os.path import isfile, join
from numba import cuda

devices = jax.local_device_count()
xgb.set_config(verbosity=0)

def run(
    repeats=5,
    dataset=61,  # iris
    task="Supervised Classification",
    target='class',
    missing=None,
    imputation=None,
    train_complete=True,
    test_complete=True,
    epochs=10,
    prop=0.35,
    rng_init=12345,
    trans_params = None,
    xgb_params =  None,
    l2 = 1e-5,
    corrupt = False
    ):
    """ 
    repeats: int, number of times to repeat for bootstrapping
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
        ("accuracy", "xgboost"):[],
        ("accuracy", "xgboost_drop"):[],
        ("nll", "full"):[],
        ("nll", "drop"):[],
        ("nll", "xgboost"):[],
        ("nll", "xgboost_drop"):[],
        ("rmse", "full"):[],
        ("rmse", "drop"):[],
        ("rmse", "xgboost"):[],
        ("rmse", "xgboost_drop"):[],
    }
    rng = np.random.default_rng(rng_init)

    key = rng.integers(9999)
    # resample argument will randomly oversample training set
    X, y, classes, cat_bin = data.prepOpenML(dataset, task)
    resample = True if classes > 1 else False
    kfolds = oversampled_Kfold(5, key=int(key), n_repeats=1, resample=resample)
    splits = kfolds.split(X, y)
    count = 0
    for train, test in splits:
        key = rng.integers(9999)

        X_train, X_test, X_valid, y_train, y_test, y_valid, diagnostics = data.openml_ds(
                X[train,:],
                y[train],
                X[test,:],
                y[test],
                task,
                cat_bin=cat_bin,
                missing=missing,
                imputation=imputation,  # one of none, simple, iterative, miceforest
                train_complete=False,
                test_complete=False,
                split=0.1,
                rng_key=key,
                prop=prop,
                corrupt=corrupt
            )
        count += 1
        print("key: {}, k: {}/{}, dataset: {}, missing: {}, impute: {}".format(key, count, len(splits), dataset, missing, imputation))
        # import dataset

        # set params for models
        if task == "Supervised Classification":
            loss_fun = cross_entropy(classes, l2_reg=l2, dropout_reg=1e-5)
            objective = 'multi:softprob'
        else:
            loss_fun = mse(l2_reg=l2, dropout_reg=1e-5)
            objective = 'reg:squarederror'
        if trans_params is None:
            model_kwargs_uat = dict(
            features=X.shape[1],
            d_model=None,
            embed_hidden_size=64,
            embed_hidden_layers=2,
            embed_activation=jax.nn.leaky_relu,
            encoder_layers=3,
            encoder_heads=10,
            enc_activation=jax.nn.leaky_relu,
            decoder_layers=6,
            decoder_heads=20,
            dec_activation=jax.nn.leaky_relu,
            net_hidden_size=64,
            net_hidden_layers=2,
            net_activation=jax.nn.leaky_relu,
            last_layer_size=32,
            out_size=classes,
            W_init = jax.nn.initializers.glorot_uniform(),
            b_init = jax.nn.initializers.normal(1e-5),
            )
        else:
            model_kwargs_uat = trans_params[0]
        
        if xgb_params is None:
            param = {'objective':objective, 'num_class':classes}
            param['max_depth'] = 12
            param['min_child_weight'] = 3
            param['eta'] = 0.01
        else:
            param = xgb_params

        # set training params
        # define training parameters
        training_kwargs_uat = trans_params[1]
        steps_per_epoch = X_train.shape[0] // training_kwargs_uat["batch_size"]
        start_steps = steps_per_epoch * 5 // 10 # wait at least 5 epochs before early stopping

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

        # get valdation for early stopping and add to training kwargs
        training_kwargs_uat["X_test"] = X_valid
        training_kwargs_uat["y_test"] = y_valid
        max_steps = 1e5
        epochs = int(max_steps // steps_per_epoch)
        training_kwargs_uat["epochs"] = epochs
        stop_steps_ = 200
        early_stopping = create_early_stopping(start_steps, stop_steps_, metric_name="loss", tol=1e-8)
        training_kwargs_uat["early_stopping"] = early_stopping
        
        # create dropped dataset baseline and implement missingness strategy
        def drop_nans(xarray, yarray):
            row_mask = ~np.any(np.isnan(xarray), axis=1)
            xdrop = xarray[row_mask, :]
            ydrop = yarray[row_mask]
            return xdrop, ydrop
        
        X_train_drop, y_train_drop = drop_nans(X_train, y_train)
        X_test_drop, y_test_drop = drop_nans(X_test, y_test)
        X_valid_drop, y_valid_drop = drop_nans(X_valid, y_valid)
        
        print("dataset sizes")
        print(X_train.shape, X_valid.shape, X_test.shape)
        if (len(y_train_drop) < devices) or (len(y_test_drop) < devices) or (len(y_valid_drop) < devices):
            print("no rows left in dropped dataset...")
            empty=True
        else:
            print("dropped dataset sizes")
            print(X_train_drop.shape, X_valid_drop.shape, X_test_drop.shape)
            empty=False
        
        # initialize drop model and strategy model and train
        key = rng.integers(9999)
        model = UAT(
            model_kwargs=model_kwargs_uat,
            training_kwargs=training_kwargs_uat,
            loss_fun=loss_fun,
            rng_key=key,
        )
        model.fit(X_train, y_train)
        # XGBoost comparison
        # build XGBoost model
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_valid, label=y_valid)
        dtest = xgb.DMatrix(X_test)
        evallist = [(dvalid, 'eval'), (dtrain, 'train')]
        num_round = 1000
        print("training xgboost for {} epochs".format(num_round))
        bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=50, verbose_eval=100)
        output_xgb = bst.predict(dtest)

        # 1. there is no point dropping if the training set is complete
        # 2. there is no point dropping if when dropped the dataset is empty
        # 3. there is no point dropping if we have imputed the training set
        # 4a. if we are corrupting we want to drop when missing is not none OR
        # 4b. if we are not corrupting we want to drop when missing is none
        if (not train_complete) and (not empty) and (imputation is None) and ((missing is not None and corrupt) or (missing is None and not corrupt)):
            training_kwargs_uat["X_test"] = X_valid_drop
            training_kwargs_uat["y_test"] = y_valid_drop
            training_kwargs_uat_ = training_kwargs_uat.copy()
            if X_train_drop.shape[0] < training_kwargs_uat["batch_size"]:
                new_bs = 2
                while new_bs <= X_train_drop.shape[0] - new_bs:
                    new_bs *= 2
                training_kwargs_uat_["batch_size"] = new_bs
            model_drop = UAT(
                model_kwargs=model_kwargs_uat,
                training_kwargs=training_kwargs_uat_,
                loss_fun=loss_fun,
                rng_key=key,
            )
            model_drop.fit(X_train_drop, y_train_drop)
            # XGBoost comparison
            # build XGBoost model
            dtrain_drop = xgb.DMatrix(X_train_drop, label=y_train_drop)
            dvalid_drop = xgb.DMatrix(X_valid_drop, label=y_valid_drop)
            dtest_drop = xgb.DMatrix(X_test_drop)
            evallist = [(dvalid, 'eval'), (dtrain, 'train')]
            num_round = 500
            print("training xgboost for {} epochs".format(num_round))
            bst_drop = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=10, verbose_eval=100)
            output_xgb_drop = bst_drop.predict(dtest_drop)
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
                
                class_x = np.argmax(output_xgb, axis=1)
                correct_x = class_x == y_test
                acc_xgb = np.sum(correct_x) / y_test.shape[0]
                if not train_complete and not empty:
                    class_d = np.argmax(output_drop, axis=1)
                    correct_d = class_d == y_test_drop
                    acc_drop = np.sum(correct_d) / y_test_drop.shape[0]
                    
                    class_d = np.argmax(output_xgb_drop, axis=1)
                    correct_d = class_d == y_test_drop
                    acc_drop_xgb = np.sum(correct_d) / y_test_drop.shape[0]
                else:
                    acc_drop = np.nan
                    acc_drop_xgb = np.nan
                metrics[("accuracy","full")].append(acc)
                metrics[("accuracy","drop")].append(acc_drop)
                metrics[("accuracy","xgboost")].append(acc_xgb)
                metrics[("accuracy","xgboost_drop")].append(acc_drop_xgb)
                tqdm.write("strategy:{}, acc full:{}, acc drop:{}, acc xgb: {}".format(imputation, acc, acc_drop, acc_xgb))
            if rm == "nll":
                nll = (- jnp.log(output + 1e-8) * jax.nn.one_hot(y_test, classes)).sum(axis=-1).mean()
                nll_xgb = (- jnp.log(output_xgb + 1e-8) * jax.nn.one_hot(y_test, classes)).sum(axis=-1).mean()
                if not train_complete and not empty:
                    nll_drop = (- jnp.log(output_drop + 1e-8) * jax.nn.one_hot(y_test_drop, classes)).sum(axis=-1).mean()
                    nll_drop_xgb = (- jnp.log(output_xgb_drop + 1e-8) * jax.nn.one_hot(y_test_drop, classes)).sum(axis=-1).mean()
                else:
                    nll_drop = np.nan
                    nll_drop_xgb = np.nan
                metrics[("nll","full")].append(nll)
                metrics[("nll","drop")].append(nll_drop)
                metrics[("nll","xgboost")].append(nll_xgb)
                metrics[("nll","xgboost_drop")].append(nll_drop_xgb)
                tqdm.write("strategy:{}, nll full:{}, nll drop:{}, nll xbg:{}".format(imputation, nll, nll_drop, nll_xgb))
            if rm == "rmse":
                rmse = np.sqrt(np.square(output - y_test).mean())
                rmse_xgb = np.sqrt(np.square(output_xgb - y_test).mean())
                if not train_complete and not empty:
                    rmse_drop = np.sqrt(np.square(output_drop - y_test_drop).mean())
                    rmse_xgb_drop = np.sqrt(np.square(output_xgb_drop - y_test_drop).mean())
                else:
                    rmse_drop = np.nan
                    rmse_xgb_drop = np.nan
                metrics[("rmse","full")].append(rmse)
                metrics[("rmse","drop")].append(rmse_drop)
                metrics[("rmse","xgboost")].append(rmse_xgb)
                metrics[("rmse","xgboost_drop")].append(rmse_xgb_drop)
                tqdm.write("strategy:{}, rmse full:{}, rmse drop:{}, rmse xbg:{}".format(imputation, rmse, rmse_drop, rmse_xgb))
    
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

    return metrics

if __name__ ==  "__main__":
    parser = argparse.ArgumentParser("train model")
    parser.add_argument("--repeats", default=5, type=int)
    parser.add_argument("--folds", default=3, type=int)
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
    args = parser.parse_args()
    
    if args.corrupt:
        data_list = data.get_list(0,2, key=12, test=lambda x, m: x == m)
        data_list["task_type"] = ["Supervised Classification" if x > 0 else "Supervised Regression" for x in data_list.NumberOfClasses]
        data_list.to_csv("results/openml/corrupted_tasklist.csv")
        missing_list = args.missing
    else:
        data_list = data.get_list(0.2, 4, key=13, test=lambda x, m: x > m)
        data_list["task_type"] = ["Supervised Classification" if x > 0 else "Supervised Regression" for x in data_list.NumberOfClasses]
        data_list.to_csv("results/openml/noncorrupted_tasklist.csv")
        missing_list = ["None"]

    data_list = data_list.reset_index()
    print(data_list)
    rng = np.random.default_rng(1234)
    key = rng.integers(9999)
    ros = RandomOverSampler(random_state=key)
    if args.dataset is not None:
        selection = np.array(args.dataset)
    else:
        selection = np.arange(len(data_list))

    for row in data_list[['did', 'task_type', 'name']].values[selection,:]:
        ## GRID SEARCH
        folds = args.folds
        # do a grid search for optimal HP for selected dataset
        X, y, classes, cat_bin = data.prepOpenML(row[0], row[1])
        key = rng.integers(9999)
        if row[1] == "Supervised Classification":
            objective = 'multi:softprob'
            resample=True
        else:
            objective = 'reg:squarederror'
            resample=False
        ## set up transformer model grid search
        key = rng.integers(9999)
        kfolds = oversampled_Kfold(2, key=int(key), n_repeats=1, resample=resample)
        splits = kfolds.split(X, y)
        trans_param_list = []
        loss_list = []
        hps_list = []
        path = "results/openml/hyperparams"
        filenames = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
        subset = [f for f in filenames if row[2] in f]
        # get batch size array
        model_kwargs_uat = dict(
            features=X.shape[1],
            d_model=None,
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
        training_kwargs_uat = dict(
                optim="adam",
                frequency=10
            )
        if len(subset) > 1 and args.load_params:
            trans_subset = [f for f in subset if 'trans' in f]
            xgb_subset = [f for f in subset if 'xgb' in f]
            with (open(trans_subset[0], "rb")) as handle:
                temp_best_trans_params = pickle.load(handle)
            with (open(xgb_subset[0], "rb")) as handle:
                temp_best_xgb_params = pickle.load(handle)
            # load up params
            temp_best_trans_params = temp_best_trans_params[np.argmin([l[1] for l in temp_best_trans_params])]
            temp_best_xgb_params = temp_best_xgb_params[np.argmin([l[1] for l in temp_best_xgb_params])]
            # set trans params
            model_kwargs_uat["d_model"] = temp_best_trans_params[0][0]
            training_kwargs_uat["lr"] = attention_lr(temp_best_trans_params[0][3], 1000)
            training_kwargs_uat["batch_size"] = temp_best_trans_params[0][1]
            best_trans_params = (model_kwargs_uat, training_kwargs_uat, temp_best_trans_params[0][2])
            # set xgb params
            best_xgb_params = temp_best_xgb_params[0]
        else:
            for hps in itertools.product([256], [16, 64], [1e-4, 1e-8], [4000, 10000]):
                print("hp search", row[2])
                if hps[1] < X.shape[0] // 2:
                    hps_list.append(hps)
                    try:
                        model_kwargs_uat["d_model"] = hps[0]
                        # generic training parameters
                        steps_per_epoch = (X.shape[0] // 2) // hps[1]
                        max_steps = 1e5
                        epochs = int(max_steps // steps_per_epoch)
                        start_steps = steps_per_epoch * 5 // 10 # wait at least 5 epochs before early stopping
                        stop_steps_ = 200
                        early_stopping = create_early_stopping(start_steps, stop_steps_, metric_name="loss", tol=1e-8)
                        training_kwargs_uat["early_stopping"] = early_stopping
                        training_kwargs_uat['batch_size'] = hps[1]
                        training_kwargs_uat['lr'] = attention_lr(hps[3], 1000)
                        training_kwargs_uat["epochs"] = epochs
                        if row[1] == "Supervised Classification":
                            print("Supervised Classification", hps[2], hps[0])
                            loss_fun = cross_entropy(classes, l2_reg=hps[2], dropout_reg=1e-5)
                        else:
                            print("Supervised Regression", hps[2], hps[0])
                            loss_fun = mse(l2_reg=hps[2], dropout_reg=1e-5)
                        trans_param_list.append((model_kwargs_uat.copy(), training_kwargs_uat.copy(), hps[2]))
                        losses = []
                        for train, test in splits:
                            X_train, X_test, X_valid, y_train, y_test, y_valid, diagnostics = data.openml_ds(
                                X[train,:],
                                y[train],
                                X[test,:],
                                y[test],
                                row[1],
                                cat_bin=cat_bin,
                                missing=None,
                                imputation=None,  # one of none, simple, iterative, miceforest
                                train_complete=False,
                                test_complete=False,
                                split=0.1,
                                rng_key=key,
                                prop=args.p,
                                corrupt=False
                            )
                            training_kwargs_uat["X_test"] = X_valid
                            training_kwargs_uat["y_test"] = y_valid
                            key = rng.integers(9999)
                            model = UAT(
                                model_kwargs=model_kwargs_uat,
                                training_kwargs=training_kwargs_uat,
                                loss_fun=loss_fun,
                                rng_key=key,
                            )

                            model.fit(X_train, y_train)
                            # break test into 'batches' to avoid OOM errors
                            test_mod = X_test.shape[0] % (hps[1])
                            test_rows = np.arange(X_test.shape[0] - test_mod)
                            test_batches = np.split(test_rows,
                                        np.maximum(1, X_test.shape[0] // (hps[1])))

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
                            losses.append(loss_loop / len(test_batches))
                        loss_list.append(np.mean(losses))
                    except Exception as e:
                        print(e)
                        for i, dev in enumerate(cuda.gpus):
                            cuda.select_device(dev)
                            cuda.close()
                        loss_list.append(np.inf)
            hp_search = list(zip(hps_list, loss_list))
            print(hp_search)
            best_trans_params = trans_param_list[np.argmin(loss_list)]

            xgb_param_list = []
            for max_depth in [10,20,50]:
                for child_weight in [3, 9]:
                    for eta in [0.01, 1]:
                        print("hp search")
                        param = {'objective':objective, 'num_class':classes}
                        param['max_depth'] = max_depth
                        param['min_child_weight'] = child_weight
                        param['eta'] = eta
                        param['verbose_eval']=100
                        xgb_param_list.append(param.copy())
            dtrain = xgb.DMatrix(X, label=y)
            num_round = 1000
            performance = []
            for params in xgb_param_list:
                history = xgb.cv(
                    params=param,
                    dtrain=dtrain,
                    num_boost_round=num_round,
                    folds=splits,
                    early_stopping_rounds=50,
                    verbose_eval=100)
                performance.append(history.values.flatten()[2])
            hp_search_xgb = list(zip(xgb_param_list, performance))
            best_xgb_params = xgb_param_list[np.argmin(performance)]
            
            with open('results/openml/hyperparams/{},trans_hyperparams.pickle'.format(row[2]), 'wb') as handle:
                pickle.dump(hp_search, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open('results/openml/hyperparams/{},xgb_hyperparams.pickle'.format(row[2]), 'wb') as handle:
                pickle.dump(hp_search_xgb, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # BOOTSTRAP PERFORMANCE if file not already present
        path = "results/openml"
        result_files = [f for f in listdir(path) if isfile(join(path, f))]
        def result_exists(filename, ds, rpts, mis, imp):
            splt = filename[:-7].split(",")
            try:
                return ds == splt[0] and rpts == int(splt[1]) and str(mis) == splt[2] and str(imp) == splt[3]
            except:
                return False
        for missing in missing_list:
            if missing == "None":
                missing = None
            for imputation in args.imputation:
                try:
                    # we do not want to impute data if there is None missing and data is not corrupted
                    if imputation != "None" and missing is None and args.corrupt:
                        continue
                    # if results file already exists then skip
                    sub = [f for f in result_files if result_exists(f, row[2], args.repeats, missing, imputation)]
                    if len(sub) > 0:
                        continue

                    if imputation == "None":
                        imputation = None

                    m1 = run(
                        repeats=args.repeats,
                        dataset=row[0],
                        task=row[1],
                        target=row[2],
                        missing=missing,
                        train_complete=args.train_complete,
                        test_complete=args.test_complete,
                        imputation=imputation,
                        epochs=args.epochs,
                        prop=args.p,
                        trans_params = best_trans_params,
                        xgb_params =  best_xgb_params,
                        l2 = best_trans_params[2],
                        corrupt=args.corrupt
                        )
                    print(row[2], missing, imputation)
                    print(m1.mean())
                    if args.save:
                        m1.to_pickle("results/openml/{},{},{},{},{},{}.pickle".format(
                        row[2], args.repeats, str(missing), str(imputation), args.test_complete, args.corrupt))

                except Exception as e:
                    print(e)
