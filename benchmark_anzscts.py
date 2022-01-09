# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import argparse
import pickle
from jax.interpreters.batching import batch
from matplotlib.pyplot import get
import numpy as np
import pandas as pd
from scipy.stats import t
import jax
import jax.numpy as jnp
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from imblearn.over_sampling import RandomOverSampler
import UAT.datasets as data
from UAT.datasets.data import simple, iterative, miceforest
from UAT import UAT, create_early_stopping
from UAT import binary_cross_entropy, cross_entropy, mse, brier
from UAT.aux import oversampled_Kfold
from UAT.training.lr_schedule import attention_lr, linear_increase
from optax import linear_onecycle_schedule, join_schedules, piecewise_constant_schedule, linear_schedule
import lightgbm as lgb
import itertools
from os import listdir
from os.path import isfile, join
from bayes_opt import BayesianOptimization
from bayes_opt import SequentialDomainReductionTransformer
import sys
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
  from tqdm.notebook import tqdm
else:
  from tqdm import tqdm

from jax.config import config
config.update("jax_debug_nans", False)

devices = jax.local_device_count()
# xgb.set_config(verbosity=0)

def create_make_model(features, rows, task, key):
    """
        Create a function to make a transformer based model with args in closure.
        Args:
            features: int, number of features
            rows: int, number of rows in training dataset
            task: str, one of 'Supervised Classification' or 'Supervised Regression'
            key: int, an rng key
        Returns:
            Callable to create model
    """
    def make_model(
            X_valid,
            y_valid,
            classes,
            batch_size=5,
            max_steps=1e5,
            lr_max=None,
            embed_depth=128,
            depth=10,
            early_stop=True,
            b2=0.99,
            reg=5,
        ):
        """
        Args:
            X_valid: ndarray, for early stopping
            y_valid: ndarray,
            classes: int, number of possible classes in outcome,
            batch_size: int, number of samples in each batch,
            max_steps: int, total number of iterations to train for
            lr_max: float, maximum learning rate
            embed_depth: int, depth of the embedding neural networks,
            depth: int, depth of the decoder in the transformer,
            early_stop: bool, whether to do early stopping,
            b2: float, interval (0, 1) hyperparameter for adam/adabelief,
            reg: int, exponent in regularization (1e-reg)
        Returns:
            model: Object
            batch_size_base2: int
            loss_fun: Callable
        """
        # use a batch size to get around 10-20 iterations per epoch
        # this means you cycle over the datasets a similar number of times
        # regardless of dataset size. 
        batch_size_base2 = min(2 ** int(np.round(np.log2(rows/20))), 256)
        # batch_size_base2 = 64
        steps_per_epoch = max(rows // batch_size_base2, 1)
        epochs = max_steps // steps_per_epoch
        while epochs < 100:
            if batch_size_base2 > 256:
                break
            batch_size_base2 *= 2
            steps_per_epoch = max(rows // batch_size_base2, 1)
            epochs = max_steps // steps_per_epoch
            print(epochs)

        freq = 5
        print("lr: {}, d: {}, depth: {}, reg: {}, b2: {}".format(
            np.exp(lr_max), int(embed_depth), int(depth), reg, b2))
        model_kwargs_uat = dict(
                features=features,
                d_model=16,
                embed_hidden_size=16,
                embed_hidden_layers=int(embed_depth),
                embed_activation=jax.nn.gelu,
                encoder_layers=int(depth),
                encoder_heads=5,
                enc_activation=jax.nn.gelu,
                decoder_layers=int(depth),
                decoder_heads=5,
                dec_activation=jax.nn.gelu,
                net_hidden_size=16,
                net_hidden_layers=5,
                net_activation=jax.nn.gelu,
                last_layer_size=16,
                out_size=classes,
                W_init = jax.nn.initializers.he_normal(),
                b_init = jax.nn.initializers.zeros,
                )
        epochs = int(max_steps // steps_per_epoch)
        start_steps = 10*steps_per_epoch # wait at least 0 epochs before early stopping
        stop_steps_ = steps_per_epoch * (epochs // 4) / min(steps_per_epoch, freq)

        # definint learning rate schedule
        m = max_steps // 2
        n_cycles = 3
        decay = piecewise_constant_schedule(
            np.exp(lr_max),
            boundaries_and_scales={
                int(15 * steps_per_epoch):0.1,
            })
        warmup = linear_schedule(
            init_value=1e-20,
            end_value=np.exp(lr_max),
            transition_steps=100
        )
        schedule=join_schedules(
            [warmup, decay],
            [50]
        )
        optim_kwargs=dict(
            b1=0.9, b2=0.99,
            eps=1e-9,
            weight_decay=10.0**-reg,
        )
        early_stopping = create_early_stopping(start_steps, stop_steps_, metric_name="loss", tol=1e-8)
        training_kwargs_uat = dict(
                    optim="adam",
                    frequency=min(steps_per_epoch, freq),
                    batch_size=batch_size_base2,
                    lr=decay,
                    #lr=1e-4,
                    epochs=epochs,
                    early_stopping=early_stopping,
                    optim_kwargs=optim_kwargs,
                    early_stop=early_stop,
                    steps_til_samp=0
                )
        if task == "Supervised Classification":
            loss_fun = cross_entropy(classes, l2_reg=0, dropout_reg=1e-7)
            # loss_fun = brier(l2_reg=0.0, dropout_reg=1e-7)
        elif task == "Supervised Regression":
            loss_fun = mse(l2_reg=0.0, dropout_reg=1e-7)
        training_kwargs_uat["X_test"] = X_valid
        training_kwargs_uat["y_test"] = y_valid
        model = UAT(
            model_kwargs=model_kwargs_uat,
            training_kwargs=training_kwargs_uat,
            loss_fun=loss_fun,
            rng_key=key,
            classes=classes,
            unsupervised_pretraining=dict(
                lr=1e-4,
                batch_size=batch_size_base2,
                cut_off=0
                )
            )
        return model, batch_size_base2, loss_fun
    return make_model

# convenience function to get ANZSCTS data from VPN source
def get_data(path, imputation, keyinit=121):
    rng = np.random.default_rng(keyinit)
    dataset = pd.read_csv(path)
    # orders dataset in ascending order based on date of procedure
    dataset.sort_values(by='DOP', inplace=True)
    dataset.dropna(subset=["MORT30"], inplace=True)

    cols = ['ICU', 'VENT', 'TP', 'PROCNO', 'AGE', 'Sex', 'Race1', 'Insur', 'SMO_H', 'SMO_C', 'DB',
            'DB_CON', 'HCHOL', 'PRECR', 'DIAL', 'TRANS', 'HG', 'HYT', 'CBVD', 'CBVD_T', 'CVA_W',
            'CART', 'PVD', 'LD', 'LD_T', 'IE', 'IE_T', 'IMSRX', 'MI', 'MI_T', 'MI_W', 'CCS', 'ANGRXG',
            'ANGRXH', 'ANGRXC', 'ANG_T', 'CHF', 'CHF_C', 'NYHA', 'SHOCK', 'RESUS', 'ARRT', 'ARRT_A',
            'ARRT_AT', 'ARRT_H', 'ARRT_V', 'PACE', 'MEDIN', 'MEDAC', 'MEDST', 'MED_ASP', 'MED_CLOP',
            'MED_TICA', 'POP', 'PCS', 'PTAVR', 'PTCA', 'PTCA_ADM', 'PTCA_H', 'CATH', 'EF', 'EF_EST',
            'LMD', 'DISVES', 'BMI', 'eGFR', 'PROC', 'STAT', 'DTCATH', 'TRAUMA', 'TUMOUR', 'CT', 'LAA',
            'AO', 'AOP', 'MIN', 'CPB', 'CCT', 'PERF', 'MINHT', 'IABP', 'IABP_W', 'IABP_I', 'ECMO',
            'ECMO_W', 'ECMO_I', 'VAD', 'VAD_W', 'VAD_I', 'IOTOE', 'ANTIFIB', 'ANTIFIB_T', 'IDGCA',
            'ITA', 'DAN_AC', 'DANV', 'DAN', 'AOPROC', 'AOPATH', 'MIPROC', 'MIPATH', 'TRPROC', 'TRPATH',
            'PUPROC', 'PUPATH', 'AOPLN', 'RBC', 'RBCUnit', 'NRBC', 'PlateUnit', 'NovoUnit', 'CryoUnit',
            'FFPUnit', 'DRAIN_4']
    cont = ['ICU', 'VENT', 'AGE', 'PRECR', 'HG','EF','BMI', 'eGFR', 'CCT', 'PERF', 'MINHT', 'DRAIN_4']
    cat_bin = [0 if x in cont else 1 for x in cols]
    outcome = ['MORT30']
    X = dataset[cols]
    y = dataset[outcome]
    idx = int(len(dataset) * 0.80)
    X_train = X.values[:idx,:]
    X_test = X.values[idx:, :]
    y_train = y.values.flatten()[:idx]
    y_test = y.values.flatten()[idx:]

    # perform desired imputation strategy
    if imputation == "simple":
        X_train, _, X_test = simple(
            X_train,
            dtypes=cat_bin,
            valid=None,
            test=X_test,
            all_cat=False
            )

    key = rng.integers(9999)
    if imputation == "iterative":
        X_train, _, X_test = iterative(
            X_train,
            key,
            dtypes=cat_bin,
            valid=None,
            test=X_test)

    key = rng.integers(9999)
    if imputation == "miceforest":
        test_input = X_test
        X_train, _, test_input = miceforest(
            X_train,
            int(key),
            dtypes=cat_bin,
            valid=None,
            test=test_input)
        X_test = test_input

    return X_train, X_test, y_train, y_test, cat_bin



# function to run an experiment on an openML dataset 
def run(
    path,  # to ANZSCTS VPN
    imputation,
    rng_init=12345,
    trans_params = None,
    gbm_params =  None,
    make_key=999,
    row_data=None
    ):
    """
    Args:
        dataset: int, referring to an OpenML dataset ID
        imputation: str, one of "Simple", "Iterative", "Miceforest" method of dealing with missingness
        rng_init: int, initial rng key
        trans_params: dict, hyperparams for the transformer
        gbm_params: dict, hyperparams for the GBM (light gbm)
        make_key: rng_key used for initializing weights of model
    Returns:
        Pandas Dataframe: performance data
        Float: percentage missing (for diagnostic purposes)
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

    X_train_, X_test, y_train_, y_test, cat_bin = get_data(path, imputation, keyinit=key)
    task = 'Supervised Classification'
    classes=2
    rows = np.arange(X_train_.shape[0])

    # turn off verbosity for LGB
    gbm_params['verbosity']=-1
    for i in range(5):
        print("key: {}, k: {}/{}, impute: {}".format(key, i, 5, imputation))
        # resample dataset
        slc = int(len(rows) * 0.8)
        idxs = rng.choice(rows, len(rows))
        idxt = idxs[:slc]
        idxv = idxs[slc:]

        X_train = X_train_[idxt, :]
        y_train = y_train_[idxt]
        X_valid = X_train_[idxt, :]
        y_valid = y_train_[idxt]

        make_model = create_make_model(X_train.shape[1], X_train.shape[0], task, make_key)
        print(trans_params)
        model, batch_size_base2, loss_fun = make_model(
                X_valid=X_valid, y_valid=y_valid, classes=classes,
                **trans_params
        )

        # equalise training classes if categorical
        if task == "Supervised Classification":
            relevant_metrics = ["accuracy", "nll"]
            key = rng.integers(9999)
            ros = RandomOverSampler(random_state=key)
            rws = np.arange(len(X_train)).reshape(-1,1)
            rws, y_train = ros.fit_resample(rws, y_train)
            X_train = X_train[rws.flatten(), :]
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
        gbm_params["learning_rate"] = np.exp(gbm_params["learning_rate"])
        bst = lgb.train(gbm_params, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=50, verbose_eval=100)
        output_gbm = bst.predict(X_test)

        # 1. there is no point dropping if the training set is complete
        # 2. there is no point dropping if when dropped the dataset is empty
        # 3. there is no point dropping if we have imputed the training set
        # 4a. if we are corrupting we want to drop when missing is not none OR
        # 4b. if we are not corrupting we want to drop when missing is none
        if (not empty) and (imputation is None) and False:
            if X_train_drop.shape[0] < trans_params["batch_size"]:
                new_bs = 2
                while new_bs <= X_train_drop.shape[0] - new_bs:
                    new_bs *= 2
            else:
                new_bs = trans_params["batch_size"]
            model_drop, batch_size_base2, loss_fun = make_model(
                X_valid=X_valid, y_valid=y_valid, **train_params)
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
        elif task == "Supervised Classification":
            output = model.predict_proba(X_test)

        # calculate performance metrics
        for rm in relevant_metrics:
            if rm == "accuracy":
                class_o = np.argmax(output, axis=1)
                correct_o = class_o == y_test
                acc = np.sum(correct_o) / y_test.shape[0]

                class_x = np.argmax(output_gbm, axis=1)
                correct_x = class_x == y_test
                acc_gbm = np.sum(correct_x) / y_test.shape[0]
                metrics[("accuracy","full")].append(acc)
                metrics[("accuracy","gbmoost")].append(acc_gbm)
                tqdm.write("strategy:{}, acc full:{}, acc gbm: {}".format(imputation, acc, acc_gbm))
            if rm == "nll":
                nll = (- jnp.log(output + 1e-8) * jax.nn.one_hot(y_test, classes)).sum(axis=-1).mean()
                nll_gbm = (- jnp.log(output_gbm + 1e-8) * jax.nn.one_hot(y_test, classes)).sum(axis=-1).mean()
                metrics[("nll","full")].append(nll)
                metrics[("nll","gbmoost")].append(nll_gbm)
                tqdm.write("strategy:{}, nll full:{}, nll xbg:{}".format(imputation, nll, nll_gbm))
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

    return metrics


if __name__ ==  "__main__":
    parser = argparse.ArgumentParser("train model")
    parser.add_argument("--path")
    parser.add_argument("--imputation", choices=["None", "Drop", "simple", "iterative", "miceforest"], nargs='+')
    parser.add_argument("--epochs", default=10000, type=int)
    parser.add_argument("--load_params", action='store_false') # default is true
    parser.add_argument("--save", action='store_true')
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--inverse", action='store_true')
    parser.add_argument("--gbm_gpu", type=int, default=-1)
    args = parser.parse_args()

    task = 'Supervised Classification'
    classes=2
    rng = np.random.default_rng(1234)
    X_train_, X_test, y_train_, y_test, cat_bin = get_data(args.path, "None")
    rows = np.arange(X_train_.shape[0])
    slc = int(len(rows) * 0.8)
    idxs = rng.choice(rows, len(rows))
    idxt = idxs[:slc]
    idxv = idxs[slc:]

    X_train = X_train_[idxt, :]
    y_train = y_train_[idxt]
    X_valid = X_train_[idxt, :]
    y_valid = y_train_[idxt]

    # BAYESIAN HYPERPARAMETER  SEARCH
    # will search if cannot load params from file
    key = rng.integers(9999)
    ros = RandomOverSampler(random_state=key)
    key = rng.integers(9999)
    if task == "Supervised Classification":
        objective = 'softmax'
        rws = np.arange(len(X_train)).reshape(-1, 1)
        rws, y_train = ros.fit_resample(rws, y_train)
        X_train = X_train[rws.flatten(), :]
    else:
        objective = 'regression'
        resample=False
    ## set up transformer model hp search
    path = "results/anzscts/hyperparams"
    filenames = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    subset = [f for f in filenames if 'anzscts' in f]

    print(X_train.shape, X_valid.shape, X_test.shape)
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
        make_key = rng.integers(9999)
        make_model = create_make_model(X_train.shape[1], X_train.shape[0], task, make_key)
        # find LR range for cyclical training / super convergence
        search_steps = 4e3
        # model, batch_size_base2, loss_fun = make_model(128, 128, 12, X_valid, y_valid, search_steps)
        # model.fit(X_train, y_train)
        # loss_hx = [h["test_current"] for h in model._history]
        # lr_hx = [h["lr"] for h in model._history]
        # lr_max = lr_hx[int(np.argmin(loss_hx) * 0.8)]

        def black_box(
                lr_max=np.log(5e-3),
                reg=6,
                embed_depth=5,
                depth=5,
                batch_size=6,
                b2=0.99
        ):
            model, batch_size_base2, loss_fun = make_model(
                X_valid, y_valid, classes=classes,
                reg=reg, lr_max=lr_max, embed_depth=embed_depth,
                depth=depth, batch_size=batch_size, b2=b2,
                early_stop=True
                )
            model.fit(X_train, y_train)
            # break test into 'batches' to avoid OOM errors
            test_mod = X_test.shape[0] % batch_size_base2 if batch_size_base2 < X_test.shape[0] else 0
            test_rows = np.arange(X_test.shape[0] - test_mod)
            test_batches = np.split(test_rows,
                        np.maximum(1, X_test.shape[0] // batch_size_base2))

            loss_loop = 0
            acc_loop = 0
            pbar1 = tqdm(total=len(test_batches), position=0, leave=False)
            @jax.jit
            def loss_calc(params, x_batch, y_batch, rng):
                out = model.apply_fun(params, x_batch, rng, False)
                loss, _ = loss_fun(params, out, y_batch)
                class_o = np.argmax(jnp.squeeze(out[0]), axis=1)
                correct_o = class_o == y_batch
                acc = np.sum(correct_o) / y_batch.shape[0]
                return loss, acc
            for tbatch in test_batches:
                key_ = jnp.ones((X_test[np.array(tbatch), ...].shape[0], 2))
                loss, acc = loss_calc(model.params, X_test[np.array(tbatch), ...], y_test[np.array(tbatch)], key_)
                loss_loop += loss
                acc_loop += acc
                pbar1.update(1)
            # make nan loss high, and average metrics over test batches
            if np.isnan(loss_loop) or np.isinf(loss_loop):
                loss_loop = 999999
            unique, counts = np.unique(y_test, return_counts=True)
            baseline = np.sum(unique[np.argmax(counts)] == y_test) / y_test.shape[0]
            acc = acc_loop / len(test_batches)
            diff = np.abs(acc - 0.5) - np.abs(baseline - 0.5)
            print(
                loss_loop/len(test_batches),
                acc_loop/len(test_batches),
                diff)
            # return - loss_loop / len(test_batches)
            return diff / (loss_loop / len(test_batches))

        pbounds={
                "lr_max":(np.log(5e-5), np.log(5e-3)),
                "embed_depth":(3, 6),
                "reg":(2,10),
                "depth":(4, 6),
                # "batch_size":(5, 9),
                }

        # bounds_transformer = SequentialDomainReductionTransformer()
        key = rng.integers(9999)
        mutating_optimizer = BayesianOptimization(
            f=black_box,
            pbounds=pbounds,
            verbose=0,
            random_state=int(key),
            # bounds_transformer=bounds_transformer
        )
        mutating_optimizer.probe(params={
            "reg":8, "lr_max":np.log(5e-4), "embed_depth":5, "depth":5
        })
        kappa = 10  # parameter to control exploitation vs exploration. higher = explore
        xi =1e-1
        mutating_optimizer.maximize(init_points=5, n_iter=args.iters, acq="ei", xi=xi, kappa=kappa)
        print(mutating_optimizer.res)
        print(mutating_optimizer.max)
        trans_results = {"max": mutating_optimizer.max, "all":mutating_optimizer.res, "key": make_key}
        
        with open('results/anzscts/hyperparams/{},trans_hyperparams.pickle'.format("anzscts"), 'wb') as handle:
            pickle.dump(trans_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if not loaded_hps_gbm:
        def black_box_gbm(max_depth, learning_rate, max_bin):
            param = {'objective':objective, 'num_class':classes}
            param['max_depth'] = int(max_depth)
            param['num_leaves'] = int(0.8 * (2**max_depth))
            param['learning_rate'] = np.exp(learning_rate)
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
                "learning_rate":(np.log(0.001), np.log(1)),
                "max_bin":(10, 100)
                }
    
        key = rng.integers(9999)
        mutating_optimizer_gbm = BayesianOptimization(
            f=black_box_gbm,
            pbounds=pbounds_gbm,
            verbose=0,
            random_state=int(key),
        )
        kappa = 10  # parameter to control exploitation vs exploration. higher = explore
        xi =1e-1
        mutating_optimizer_gbm.maximize(init_points=5, n_iter=args.iters, acq="ei", xi=xi, kappa=kappa)
        print(mutating_optimizer_gbm.res)
        print(mutating_optimizer_gbm.max)
        best_params_gbm = mutating_optimizer_gbm.max
        best_params_gbm["params"]["objective"]=objective
        best_params_gbm["params"]["num_class"]=classes
        best_params_gbm["params"]['num_leaves'] = int(0.8 * (2**best_params_gbm["params"]["max_depth"]))
        gbm_results = {"max": best_params_gbm, "all":mutating_optimizer_gbm.res} 

        with open('results/anzscts/hyperparams/{},gbm_hyperparams.pickle'.format('anzscts'),'wb') as handle:
            pickle.dump(gbm_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # BOOTSTRAP PERFORMANCE if file not already present
    path = "results/openml"
    result_files = [f for f in listdir(path) if isfile(join(path, f))]
    def result_exists(filename, ds, imp):
        splt = filename[:-7].split(",")
        try:
            return ds == splt[0] and str(imp) == splt[1]
        except:
            return False
    for imputation in args.imputation:
        sub = [f for f in result_files if result_exists(f, 'anzscts', imputation)]
        if len(sub) > 0:
            continue

        if imputation == "None":
            imputation = None
        m1 = run(
            path=args.path,
            imputation=imputation,
            trans_params = trans_results["max"]["params"],
            gbm_params =  gbm_results["max"]["params"],
            make_key=trans_results["key"],
            )
        print('anzscts', imputation)
        print(m1.mean())
        if args.save:
            m1.to_pickle("results/anzscts/{},{}.pickle".format(
            "anzscts", str(imputation)))

