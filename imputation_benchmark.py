import argparse
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
from UAT import UAT
from UAT import binary_cross_entropy, cross_entropy, mse

def run(repeats=5, dataset="thoracic", missing="None", train_complete=True, strategy="UAT", epochs=10, cols_mis=4, p=0.15):
    """ 
    repeats: int, number of times to repeat for bootstrapping
    dataset: string, one of "thoracic", "abalone", "banking" etc ...
    missing: str, one of "None", "MCAR", "MAR", "MNAR" to define missingness pattern
    train_complete: bool, whether the training set is to be complete
    strategy: str, one of "Simple", "Iterative", "UAT" method of dealing with missingness
    epochs: int, number of epochs to train model
    """
    key = 0
    metrics = {
        ("auc", "full"):[],
        ("auc", "drop"):[],
        ("rmse", "full"):[],
        ("rmse", "drop"):[],
    }
    while key < repeats:
        # import dataset
        if dataset == "thoracic":
            X, y = data.thoracic(missing=missing, rng_key=key, cols_miss=cols_mis, p=p)
            if train_complete:
                X_, y = data.thoracic(missing="None", rng_key=key, cols_miss=cols_mis, p=p)
            loss = binary_cross_entropy
            relevant_metrics = ["auc"]
        if dataset == "abalone":
            X, y, classes = data.abalone(missing=missing, rng_key=key, cols_miss=cols_mis, p=p)
            if train_complete:
                X_, y, classes = data.abalone(missing="None", rng_key=key, cols_miss=cols_mis, p=p)
            loss = mse
            relevant_metrics = ["rmse"]
        if dataset == "banking":
            X, y = data.banking()
            if train_complete:
                X_ = X  # dataset already has at least 1 missing value in approx 1/4 of samples
            loss = binary_cross_entropy
            relevant_metrics = ["auc"]

        # train and test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=key)
        if train_complete:
            X_train, _, _, _ = train_test_split(X_, y, test_size=0.33, random_state=key)
        
        # equalise training set if categorical
        if "auc" in relevant_metrics:
            ros = RandomOverSampler(random_state=key)
            X_train, y_train = ros.fit_resample(X_train, y_train)

        # create dropped dataset baseline and implement missingness strategy
        def drop_nans(xarray, yarray):
            row_mask = ~np.any(np.isnan(xarray), axis=1)
            xdrop = xarray[row_mask, :]
            ydrop = yarray[row_mask]
            return xdrop, ydrop
        
        X_train_drop, y_train_drop = drop_nans(X_train, y_train)
        X_test_drop, y_test_drop = drop_nans(X_test, y_test)
        
        if (len(y_train_drop) == 0) or (len(y_test_drop) == 0):
            continue
        
        if strategy == "Simple":
            imp = SimpleImputer(missing_values=np.nan, strategy='median')
            imp.fit(X_train)
            X_train = imp.transform(X_train)
            X_test = imp.transform(X_test)
        
        if strategy == "Iterative":
            imp = IterativeImputer(max_iter=10, random_state=key)
            imp.fit(X_train)
            X_train = imp.transform(X_train)
            X_test = imp.transform(X_test)
        
        if strategy == "UAT":
            # no need to change dataset for UAT
            pass
        
        # initialize drop model and strategy model and train
        model_kwargs_uat = dict(
                features=X.shape[1],
                d_model=32,
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
                W_init = jax.nn.initializers.glorot_uniform(),
                b_init = jax.nn.initializers.normal(0.01),
                )
        training_kwargs_uat = dict(
            batch_size=32,
            epochs=epochs,
            lr=1e-4,
            optim="adabelief",
            X_test=X_test,
            y_test=y_test
        )
        loss_fun = loss(l2_reg=1e-4, dropout_reg=1e-5)
        model = UAT(
            model_kwargs=model_kwargs_uat,
            training_kwargs=training_kwargs_uat,
            loss_fun=loss_fun,
            rng_key=key,
        )
        model.fit(X_train, y_train)
        model_drop = model
        if not train_complete:
            model_drop = UAT(
                model_kwargs=model_kwargs_uat,
                training_kwargs=training_kwargs_uat,
                loss_fun=loss_fun,
                rng_key=key,
            )
            model_drop.fit(X_train_drop, y_train_drop)
        
        # assess performance of models on test set and store metrics
        # predict prob will output 
        if "rmse" in relevant_metrics:
            output = model.predict(X_test)
            output_drop = model_drop.predict(X_test_drop)
        if "auc" in relevant_metrics:
            output = model.predict_proba(X_test)
            output_drop = model_drop.predict_proba(X_test_drop)

        # calculate AUROC score
        for rm in relevant_metrics:
            if rm == "auc":
                auc = roc_auc_score(y_test, output)
                auc_drop = roc_auc_score(y_test_drop, output_drop)

                metrics[("auc","full")].append(auc)
                metrics[("auc","drop")].append(auc_drop)
            elif rm == "rmse":
                rmse = np.sqrt(np.square(output - y_test).mean())
                rmse_drop = np.sqrt(np.square(output_drop - y_test_drop).mean())
                metrics[("rmse","full")].append(rmse)
                metrics[("rmse","drop")].append(rmse_drop)

        # update random key
        key += 1
    
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
        metrics[m, "delta"] = (metrics[m, "full"].values - metrics[m, "drop"].values) / metrics[m, "drop"].values * 100
    metrics = metrics.sort_index(axis=1)

    return metrics

if __name__ ==  "__main__":
    parser = argparse.ArgumentParser("train model")
    parser.add_argument("--repeats", default=10, type=int)
    parser.add_argument("--dataset", choices=["thoracic", "abalone", "banking"], default="thoracic")
    parser.add_argument("--missing", choices=["None", "MCAR", "MAR", "MNAR"], default="None")
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--p", default=0.3, type=float)
    parser.add_argument("--train_complete", action='store_true')
    parser.add_argument("--save", action='store_true')
    args = parser.parse_args()
    
    m1 = run(
        repeats=args.repeats,
        dataset=args.dataset,
        missing=args.missing,
        train_complete=args.train_complete,
        strategy="UAT",
        epochs=args.epochs,
        p=args.p
        )
    m2 = run(
        repeats=args.repeats,
        dataset=args.dataset,
        missing=args.missing,
        train_complete=args.train_complete,
        strategy="Simple",
        epochs=args.epochs,
        p=args.p
        )
    m3 = run(
        repeats=args.repeats,
        dataset=args.dataset,
        missing=args.missing,
        train_complete=args.train_complete,
        strategy="Iterative",
        epochs=args.epochs,
        p=args.p
        )

    print(m1.mean())
    print(m2.mean())
    print(m3.mean())
    
    if args.save:
        m1.to_pickle("results/imputation/UAT_{}_{}_{}_{}.pickle".format(
            args.repeats, args.dataset, args.missing, args.train_complete))
        m2.to_pickle("results/imputation/Simple_{}_{}_{}_{}.pickle".format(
            args.repeats, args.dataset, args.missing, args.train_complete))
        m3.to_pickle("results/imputation/Iterative_{}_{}_{}_{}.pickle".format(
            args.repeats, args.dataset, args.missing, args.train_complete))