import argparse
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
import xgboost as xgb
from tqdm import tqdm

devices = jax.local_device_count()

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
    for repeat in range(repeats):
        key = rng.integers(9999)
        print("key: {}, repeat: {}/{}, dataset: {}, missing: {}, impute: {}".format(key, repeat, repeats, dataset, missing, imputation))
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
                split=0.2,
                rng_key=key,
                p=p,
                cols_miss=4)

            # define model
            model_kwargs_uat = dict(
                features=X_train.shape[1],
                d_model=16,
                embed_hidden_size=16,
                embed_hidden_layers=3,
                embed_activation=jax.nn.leaky_relu,
                encoder_layers=3,
                encoder_heads=10,
                enc_activation=jax.nn.leaky_relu,
                decoder_layers=6,
                decoder_heads=20,
                dec_activation=jax.nn.leaky_relu,
                net_hidden_size=16,
                net_hidden_layers=3,
                net_activation=jax.nn.leaky_relu,
                last_layer_size=16,
                out_size=classes,
                W_init = jax.nn.initializers.glorot_uniform(),
                b_init = jax.nn.initializers.normal(0.0001),
                )
            
            # define training parameters
            batch_size = 1024
            stop_epochs = 100
            wait_epochs = 500
            
            training_kwargs_uat = dict(
                batch_size=batch_size,
                epochs=epochs,
                lr=5e-4,
                optim="adam",
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
                d_model=16,
                embed_hidden_size=16,
                embed_hidden_layers=3,
                embed_activation=jax.nn.leaky_relu,
                encoder_layers=3,
                encoder_heads=10,
                enc_activation=jax.nn.leaky_relu,
                decoder_layers=6,
                decoder_heads=20,
                dec_activation=jax.nn.leaky_relu,
                net_hidden_size=16,
                net_hidden_layers=3,
                net_activation=jax.nn.leaky_relu,
                last_layer_size=16,
                out_size=classes,
                W_init = jax.nn.initializers.glorot_uniform(),
                b_init = jax.nn.initializers.normal(0.0001),
                )
            
            # define training parameters
            batch_size = 1024
            stop_epochs = 2500
            wait_epochs = 500
            # lr = exponential_decay(1e-4, X_train.shape[0] // batch_size, 0.999)
            lr = 5e-4
            training_kwargs_uat = dict(
                batch_size=batch_size,
                epochs=epochs,
                lr=lr,
                optim="adam"
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
                split=0.2,
                rng_key=key,
                p=p,
                cols_miss=100
            )
            print(X_train.shape, X_valid.shape, X_test.shape)

            # define model
            model_kwargs_uat = dict(
                features=X_train.shape[1],
                d_model=16,
                embed_hidden_size=16,
                embed_hidden_layers=3,
                embed_activation=jax.nn.leaky_relu,
                encoder_layers=3,
                encoder_heads=10,
                enc_activation=jax.nn.leaky_relu,
                decoder_layers=6,
                decoder_heads=20,
                dec_activation=jax.nn.leaky_relu,
                net_hidden_size=16,
                net_hidden_layers=3,
                net_activation=jax.nn.leaky_relu,
                last_layer_size=16,
                out_size=classes,
                W_init = jax.nn.initializers.glorot_uniform(),
                b_init = jax.nn.initializers.normal(0.0001),
                )
            
            # define training parameters
            batch_size = 2
            # get maximum batch size possible
            while batch_size < X_train.shape[0] - batch_size:
                batch_size *= 2
            stop_epochs = 1000
            wait_epochs = 500

            training_kwargs_uat = dict(
                batch_size=batch_size,
                epochs=epochs,
                lr=1e-3,
                optim="adam"
            )
            loss_fun = cross_entropy(classes=classes, l2_reg=1e-4, dropout_reg=1e-5)
            relevant_metrics = ["auc", "accuracy", "nll"]
            param = {'objective':'multi:softprob', 'num_class':classes}
            param['max_depth'] = 8
            param['min_child_weight'] = 5
  
        if dataset == "abalone":
            task = "regression"

            # get data
            X_train, X_valid, X_test, y_train, y_valid, y_test, classes = data.abalone(
                missing=missing,
                imputation=imputation,  # one of none, simple, iterative, miceforest
                train_complete=train_complete,
                test_complete=test_complete,
                split=0.2,
                rng_key=key,
                p=p,
                cols_miss=100
            )
            print(X_train.shape, X_valid.shape, X_test.shape)

            # define model
            model_kwargs_uat = dict(
                features=X_train.shape[1],
                d_model=16,
                embed_hidden_size=16,
                embed_hidden_layers=3,
                embed_activation=jax.nn.leaky_relu,
                encoder_layers=3,
                encoder_heads=10,
                enc_activation=jax.nn.leaky_relu,
                decoder_layers=6,
                decoder_heads=20,
                dec_activation=jax.nn.leaky_relu,
                net_hidden_size=16,
                net_hidden_layers=3,
                net_activation=jax.nn.leaky_relu,
                last_layer_size=16,
                out_size=classes,
                W_init = jax.nn.initializers.glorot_uniform(),
                b_init = jax.nn.initializers.normal(0.0001),
                )
            
            # define training parameters
            batch_size = 2
            # get maximum batch size possible
            while batch_size < X_train.shape[0] - batch_size:
                batch_size *= 2
            stop_epochs = 1000
            wait_epochs = 500
            training_kwargs_uat = dict(
                batch_size=batch_size,
                epochs=epochs,
                lr=1e-3,
                optim="adam"
            )
            loss_fun = mse(l2_reg=1e-4, dropout_reg=1e-5)
            relevant_metrics = ["rmse"]
            param = {'objective':'reg:squarederror'}
            param['max_depth'] = 8
            param['min_child_weight'] = 5

        if dataset == "banking":
            # define task
            task = "classification"

            # get data
            X_train, X_valid, X_test, y_train, y_valid, y_test, classes = data.banking(
                imputation=imputation,
                rng_key=key,
                split=0.15
            )
            print(X_train.shape, X_valid.shape, X_test.shape, y_train.shape, y_valid.shape, y_test.shape)

            # define model
            model_kwargs_uat = dict(
                features=X_train.shape[1],
                d_model=16,
                embed_hidden_size=16,
                embed_hidden_layers=3,
                embed_activation=jax.nn.leaky_relu,
                encoder_layers=3,
                encoder_heads=10,
                enc_activation=jax.nn.leaky_relu,
                decoder_layers=6,
                decoder_heads=20,
                dec_activation=jax.nn.leaky_relu,
                net_hidden_size=16,
                net_hidden_layers=3,
                net_activation=jax.nn.leaky_relu,
                last_layer_size=16,
                out_size=classes,
                W_init = jax.nn.initializers.glorot_uniform(),
                b_init = jax.nn.initializers.normal(0.0001),
                )
            
            # define training parameters
            batch_size = 2
            # get maximum batch size possible
            while batch_size < X_train.shape[0] - batch_size:
                batch_size *= 2
            
            stop_epochs = 100
            wait_epochs = 500

            training_kwargs_uat = dict(
                batch_size=batch_size,
                epochs=epochs,
                lr=5e-4,
                optim="adam"
            )
            loss_fun = cross_entropy(classes=classes, l2_reg=1e-4, dropout_reg=1e-5)
            relevant_metrics = ["auc", "accuracy", "nll"]
            param = {'objective':'multi:softprob', 'num_class':classes}
            param['max_depth'] = 8
            param['min_child_weight'] = 5
        
        if dataset == "mnist":
            # define task
            task = "classification"

            # get data
            X_train, X_valid, X_test, y_train, y_valid, y_test, classes = data.mnist(
                imputation=imputation,
                rng_key=key,
                split=0.15
            )
            print(X_train.shape, X_valid.shape, X_test.shape, y_train.shape, y_valid.shape, y_test.shape)
            
             # define model
            model_kwargs_uat = dict(
                features=X_train.shape[1],
                d_model=32,
                embed_hidden_size=16,
                embed_hidden_layers=3,
                embed_activation=jax.nn.leaky_relu,
                encoder_layers=5,
                encoder_heads=10,
                enc_activation=jax.nn.leaky_relu,
                decoder_layers=10,
                decoder_heads=20,
                dec_activation=jax.nn.leaky_relu,
                net_hidden_size=16,
                net_hidden_layers=3,
                net_activation=jax.nn.leaky_relu,
                last_layer_size=16,
                out_size=classes,
                W_init = jax.nn.initializers.glorot_uniform(),
                b_init = jax.nn.initializers.normal(0.0001),
                )
            
            # define training parameters
            batch_size = 256
            stop_epochs = 800
            wait_epochs = 500

            training_kwargs_uat = dict(
                batch_size=batch_size,
                epochs=epochs,
                lr=1e-3,
                optim="adam"
            )
            loss_fun = cross_entropy(classes=classes, l2_reg=1e-6, dropout_reg=1e-5)
            relevant_metrics = ["auc", "accuracy", "nll"]
            param = {'objective':'multi:softprob', 'num_class':classes}
            param['max_depth'] = 8
            param['min_child_weight'] = 5

        # equalise training classes if categorical
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
        early_stopping = create_early_stopping(stop_steps_, wait_epochs, metric_name="loss", tol=1e-8)
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
        num_round = 500
        print("training xgboost for {} epochs".format(num_round))
        bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=10, verbose_eval=100)
        output_xgb = bst.predict(dtest)

        if (not train_complete) and (not empty) and (imputation is None) and (missing is not None):
            training_kwargs_uat["X_test"] = X_valid_drop
            training_kwargs_uat["y_test"] = y_valid_drop
            model_drop = UAT(
                model_kwargs=model_kwargs_uat,
                training_kwargs=training_kwargs_uat,
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
                metrics[("rmse","full")].append(rmse)
                metrics[("rmse","drop")].append(rmse_drop)
                metrics[("rmse","xgboost")].append(rmse_xgb)
                metrics[("rmse","xgboost_drop")].append(rmse_xgb_drop)
    
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
    parser.add_argument("--dataset", choices=[
        "spiral","thoracic", "abalone", "banking", "anneal", "mnist"], default="spiral", nargs='+')
    parser.add_argument("--missing", choices=["None", "MCAR", "MAR", "MNAR"], default="None", nargs='+')
    parser.add_argument("--imputation", choices=["None", "Drop", "simple", "iterative", "miceforest"], nargs='+')
    parser.add_argument("--epochs", default=10000, type=int)
    parser.add_argument("--p", default=0.3, type=float)
    parser.add_argument("--train_complete", action='store_true') # default is false
    parser.add_argument("--test_complete", action='store_false') # default is true
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
                    m1.to_pickle("results/imputation/{}_{}_{}_{}_{}_{}.pickle".format(
                    dataset, args.repeats, str(missing), imputation, args.train_complete, args.test_complete))