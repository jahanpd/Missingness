import argparse
from jax.interpreters.batching import batch
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
from UAT import UAT, create_early_stopping
from UAT import binary_cross_entropy, cross_entropy, mse
import xgboost as xgb
from tqdm import tqdm

def run(repeats=5, dataset="thoracic", missing=None, imputation=None, train_complete=True, test_complete=True, epochs=10, p=0.15):
    """ 
    repeats: int, number of times to repeat for bootstrapping
    dataset: string, one of "thoracic", "abalone", "banking" etc ...
    missing: str, one of "None", "MCAR", "MAR", "MNAR" to define missingness pattern
    train_complete: bool, whether the training set is to be complete
    strategy: str, one of "Simple", "Iterative", "UAT" method of dealing with missingness
    epochs: int, number of epochs to train model
    """
    metrics = {
        ("accuracy", "full"):[],
        ("accuracy", "drop"):[],
        ("accuracy", "xgboost"):[],
        ("nll", "full"):[],
        ("nll", "drop"):[],
        ("nll", "xgboost"):[],
        ("rmse", "full"):[],
        ("rmse", "drop"):[],
        ("rmse", "xgboost"):[],
    }
    rng = np.random.default_rng(12345)
    for _ in range(repeats):
        key = rng.integers(9999)
        print(key, repeats)
        # import dataset
        if dataset == "spiral":
            # define task
            task = "classification"

            # get data
            X_train, X_valid, X_test, y_train, y_valid, y_test, _, classes = data.spiral(
                2048,
                missing=missing,
                imputation=imputation,  # one of none, simple, iterative, miceforest
                train_complete=train_complete,
                test_complete=test_complete,
                split=0.33,
                rng_key=key,
                p=p,
                cols=4)

            # define model
            model_kwargs_uat = dict(
                features=X_train.shape[1],
                d_model=64,
                embed_hidden_size=64,
                embed_hidden_layers=2,
                embed_activation=jax.nn.relu,
                encoder_layers=3,
                encoder_heads=5,
                enc_activation=jax.nn.relu,
                decoder_layers=3,
                decoder_heads=5,
                dec_activation=jax.nn.relu,
                net_hidden_size=64,
                net_hidden_layers=2,
                net_activation=jax.nn.relu,
                last_layer_size=32,
                out_size=classes,
                W_init = jax.nn.initializers.glorot_uniform(),
                b_init = jax.nn.initializers.normal(0.01),
                )
            
            # define training parameters
            batch_size = 32
            stop_epochs = 50
            
            training_kwargs_uat = dict(
                batch_size=batch_size,
                epochs=epochs,
                lr=5e-5,
                optim="adabelief",
            )
            loss_fun = cross_entropy(2, l2_reg=1e-4, dropout_reg=1e-5)
            relevant_metrics = ["auc", "accuracy", "nll"]
            
            # xgboost params
            param = {'objective':'multi:softprob', 'num_class':classes}
            param['max_depth'] = 10
            param['min_child_weight'] = 3
            param['eta'] = 1

        if dataset == "anneal":
            # define task
            task = "classification"

            # get data
            X_train, X_valid, X_test, y_train, y_valid, y_test, classes = data.anneal(
                imputation=imputation,
                rng_key=key
            )

            # define model
            classes = len(np.concatenate([y_train, y_test]))
            model_kwargs_uat = dict(
                features=X_train.shape[1],
                d_model=128,
                embed_hidden_size=64,
                embed_hidden_layers=2,
                embed_activation=jax.nn.relu,
                encoder_layers=3,
                encoder_heads=5,
                enc_activation=jax.nn.relu,
                decoder_layers=3,
                decoder_heads=5,
                dec_activation=jax.nn.relu,
                net_hidden_size=64,
                net_hidden_layers=2,
                net_activation=jax.nn.relu,
                last_layer_size=32,
                out_size=classes,
                W_init = jax.nn.initializers.glorot_uniform(),
                b_init = jax.nn.initializers.normal(0.01),
                )
            
            # define training parameters
            batch_size = 32
            stop_epochs = 25
            training_kwargs_uat = dict(
                batch_size=32,
                epochs=epochs,
                lr=5e-4,
                optim="adabelief"
            )
            loss_fun = cross_entropy(classes=classes, l2_reg=1e-4, dropout_reg=1e-5)
            relevant_metrics = ["auc", "accuracy", "nll"]
            param = {'objective':'multi:softprob', 'num_class':classes}
            param['max_depth'] = 8
            param['min_child_weight'] = 5

        if dataset == "thoracic":
            # define task
            task = "classification"

            # get data
            X_train, X_valid, X_test, y_train, y_valid, y_test, classes = data.thoracic(
                missing=missing,
                imputation=imputation,  # one of none, simple, iterative, miceforest
                train_complete=train_complete,
                test_complete=test_complete,
                split=0.05,
                rng_key=key,
                p=p
            )
            print(X_train.shape, X_valid.shape, X_test.shape)

            # define model
            model_kwargs_uat = dict(
                features=X_train.shape[1],
                d_model=64,
                embed_hidden_size=64,
                embed_hidden_layers=2,
                embed_activation=jax.nn.relu,
                encoder_layers=3,
                encoder_heads=5,
                enc_activation=jax.nn.relu,
                decoder_layers=3,
                decoder_heads=5,
                dec_activation=jax.nn.relu,
                net_hidden_size=64,
                net_hidden_layers=2,
                net_activation=jax.nn.relu,
                last_layer_size=32,
                out_size=classes,
                W_init = jax.nn.initializers.glorot_uniform(),
                b_init = jax.nn.initializers.normal(0.01),
                )
            
            # define training parameters
            batch_size = 32
            stop_epochs = 80
            training_kwargs_uat = dict(
                batch_size=32,
                epochs=epochs,
                lr=1e-4,
                optim="adabelief"
            )
            loss_fun = cross_entropy(classes=classes, l2_reg=1e-4, dropout_reg=1e-5)
            relevant_metrics = ["auc", "accuracy", "nll"]
            param = {'objective':'multi:softprob', 'num_class':classes}
  
        if dataset == "abalone":
            task = "regression"

            # get data
            X_train, X_valid, X_test, y_train, y_valid, y_test, classes = data.abalone(
                missing=missing,
                imputation=imputation,  # one of none, simple, iterative, miceforest
                train_complete=train_complete,
                test_complete=test_complete,
                split=0.33,
                rng_key=key,
                p=p
            )

            # define model
            model_kwargs_uat = dict(
                features=X_train.shape[1],
                d_model=64,
                embed_hidden_size=64,
                embed_hidden_layers=2,
                embed_activation=jax.nn.relu,
                encoder_layers=3,
                encoder_heads=5,
                enc_activation=jax.nn.relu,
                decoder_layers=3,
                decoder_heads=5,
                dec_activation=jax.nn.relu,
                net_hidden_size=64,
                net_hidden_layers=2,
                net_activation=jax.nn.relu,
                last_layer_size=32,
                out_size=classes,
                W_init = jax.nn.initializers.glorot_uniform(),
                b_init = jax.nn.initializers.normal(0.01),
                )
            
            # define training parameters
            batch_size = 32
            stop_epochs = 90
            training_kwargs_uat = dict(
                batch_size=32,
                epochs=epochs,
                lr=1e-4,
                optim="adabelief"
            )
            loss_fun = mse(l2_reg=1e-4, dropout_reg=1e-5)
            relevant_metrics = ["rmse"]
            param = {'objective':'reg:squarederror'}

        if dataset == "banking":
            # define task
            task = "classification"

            # get data
            X_train, X_valid, X_test, y_train, y_valid, y_test, classes = data.banking(
                imputation=imputation,
                rng_key=key,
                split=0.15
            )
            print(X_train.shape, X_valid.shape, X_test.shape)

            # define model
            model_kwargs_uat = dict(
                features=X_train.shape[1],
                d_model=64,
                embed_hidden_size=64,
                embed_hidden_layers=2,
                embed_activation=jax.nn.relu,
                encoder_layers=3,
                encoder_heads=5,
                enc_activation=jax.nn.relu,
                decoder_layers=3,
                decoder_heads=5,
                dec_activation=jax.nn.relu,
                net_hidden_size=64,
                net_hidden_layers=2,
                net_activation=jax.nn.relu,
                last_layer_size=32,
                out_size=classes,
                W_init = jax.nn.initializers.glorot_uniform(),
                b_init = jax.nn.initializers.normal(0.01),
                )
            
            # define training parameters
            batch_size = 32
            stop_epochs = 90
            training_kwargs_uat = dict(
                batch_size=32,
                epochs=epochs,
                lr=1e-4,
                optim="adabelief"
            )
            loss_fun = cross_entropy(classes=classes, l2_reg=1e-4, dropout_reg=1e-5)
            relevant_metrics = ["auc", "accuracy", "nll"]
            param = {'objective':'multi:softprob', 'num_class':classes}
        
        # equalise training set if categorical
        if task == "classification":
            key = rng.integers(9999)
            ros = RandomOverSampler(random_state=key)
            X_train, y_train = ros.fit_resample(X_train, y_train)

        # sanity check
        if imputation is not None:
            assert np.all(~np.isnan(X_train))

        # get valdation for early stopping and add to training kwargs
        training_kwargs_uat["X_test"] = X_valid
        training_kwargs_uat["y_test"] = y_valid
        steps_per_epoch = X_train.shape[0] // batch_size
        stop_steps_ = steps_per_epoch * stop_epochs
        early_stopping = create_early_stopping(stop_steps_, 20, metric_name="loss", tol=1e-8)
        training_kwargs_uat["early_stopping"] = early_stopping
        # create dropped dataset baseline and implement missingness strategy
        def drop_nans(xarray, yarray):
            row_mask = ~np.any(np.isnan(xarray), axis=1)
            xdrop = xarray[row_mask, :]
            ydrop = yarray[row_mask]
            return xdrop, ydrop
        
        X_train_drop, y_train_drop = drop_nans(X_train, y_train)
        X_test_drop, y_test_drop = drop_nans(X_test, y_test)
        
        if (len(y_train_drop) == 0) or (len(y_test_drop) == 0):
            print("no rows left in dropped dataset...")
            empty=True
        else:
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
        model_drop = model
        if (not train_complete) and (not empty) and (imputation is None) and (missing is not None):
            model_drop = UAT(
                model_kwargs=model_kwargs_uat,
                training_kwargs=training_kwargs_uat,
                loss_fun=loss_fun,
                rng_key=key,
            )
            model_drop.fit(X_train_drop, y_train_drop)
        else:
            empty = True # in order to set dropped metrics to NA

        # XGBoost comparison
        # build XGBoost model
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_valid, label=y_valid)
        dtest = xgb.DMatrix(X_test)
        evallist = [(dvalid, 'eval'), (dtrain, 'train')]
        num_round = epochs
        bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=10)
        output_xgb = bst.predict(dtest)

        
        # assess performance of models on test set and store metrics
        # predict prob will output 
        if task == "regression":
            output = model.predict(X_test)
            if not train_complete and not empty:
                output_drop = model_drop.predict(X_test_drop)
        if task == "classification":
            output = model.predict_proba(X_test)
            if not train_complete and not empty:
                output_drop = model_drop.predict_proba(X_test_drop)

        # calculate performance metrics
        for rm in relevant_metrics:
            if rm == "accuracy":
                class_ = np.argmax(output, axis=1)
                correct = class_ == y_test
                acc = np.sum(correct) / y_test.shape[0]
                
                class_ = np.argmax(output_xgb, axis=1)
                correct = class_ == y_test
                acc_xgb = np.sum(correct) / y_test.shape[0]
                if not train_complete and not empty:
                    class_ = np.argmax(output, axis=1)
                    correct = class_ == y_test
                    acc_drop = np.sum(correct) / y_test_drop.shape[0]
                else:
                    acc_drop = np.nan
                metrics[("accuracy","full")].append(acc)
                metrics[("accuracy","drop")].append(acc_drop)
                metrics[("accuracy","xgboost")].append(acc_xgb)
                tqdm.write("strategy:{}, acc full:{}, acc drop:{}, acc xgb: {}".format(imputation, acc, acc_drop, acc_xgb))
            if rm == "nll":
                nll = (- jnp.log(output) * jax.nn.one_hot(y_test, classes)).sum(axis=-1).mean()
                nll_xgb = (- jnp.log(output_xgb) * jax.nn.one_hot(y_test, classes)).sum(axis=-1).mean()
                if not train_complete and not empty:
                    nll_drop = (output_drop * jax.nn.one_hot(y_test_drop, classes)).sum(axis=-1).mean()
                else:
                    nll_drop = np.nan
                metrics[("nll","full")].append(nll)
                metrics[("nll","drop")].append(nll_drop)
                metrics[("nll","xgboost")].append(nll_xgb)
                tqdm.write("strategy:{}, nll full:{}, nll drop:{}, nll xbg:{}".format(imputation, nll, nll_drop, nll_xgb))
            if rm == "rmse":
                rmse = np.sqrt(np.square(output - y_test).mean())
                rmse_xgb = np.sqrt(np.square(output_xgb - y_test).mean())
                if not train_complete and not empty:
                    rmse_drop = np.sqrt(np.square(output_drop - y_test_drop).mean())
                else:
                    rmse_drop = np.nan
                metrics[("rmse","full")].append(rmse)
                metrics[("rmse","drop")].append(rmse_drop)
                metrics[("rmse","xgboost")].append(rmse_xgb)
    
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
    parser.add_argument("--dataset", choices=["spiral","thoracic", "abalone", "banking", "anneal"], default="spiral", nargs='+')
    parser.add_argument("--missing", choices=["None", "MCAR", "MAR", "MNAR"], default="None", nargs='+')
    parser.add_argument("--imputation", choices=["None", "simple", "iterative", "miceforest"], nargs='+')
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--p", default=0.3, type=float)
    parser.add_argument("--train_complete", action='store_true')
    parser.add_argument("--test_complete", action='store_false')
    parser.add_argument("--save", action='store_true')
    args = parser.parse_args()
    
    for dataset in args.dataset:
        for missing in args.missing:
            if missing == "None":
                missing = None
            for imputation in args.imputation:
                if imputation == "None":
                    imputation = None
                m1 = run(
                    repeats=args.repeats,
                    dataset=dataset,
                    missing=missing,
                    train_complete=args.train_complete,
                    test_complete=args.test_complete,
                    imputation=imputation,
                    epochs=args.epochs,
                    p=args.p
                    )
                print(dataset, missing, imputation)
                print(m1.mean())
                if args.save:
                    m1.to_pickle("results/imputation/UAT_{}_{}_{}_{}_{}_{}.pickle".format(
                    args.repeats, dataset, str(missing), imputation, args.train_complete, args.test_complete))