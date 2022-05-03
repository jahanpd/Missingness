import argparse
from jax.experimental.optimizers import l2_norm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import t
import jax
import jax.numpy as jnp
from UAT import UAT, Ensemble, create_early_stopping
from UAT import binary_cross_entropy
import UAT.datasets as data
import os

# jax.config.update("jax_debug_nans", True)
# force parallel processing across cpu cores
# os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=12'

"""
Script for the second spiral dataset experiment exploring the effect of missingness in the latent space.
"""

def run(iterations, missing="None", epochs=10):
    # define models and training params as dicts 
    model_kwargs_uat = dict(
                features=4,
                d_model=32,
                embed_hidden_size=32,
                embed_hidden_layers=2,
                embed_activation=jax.nn.gelu,
                encoder_layers=3,
                encoder_heads=3,
                enc_activation=jax.nn.gelu,
                decoder_layers=3,
                decoder_heads=3,
                dec_activation=jax.nn.gelu,
                net_hidden_size=32,
                net_hidden_layers=2,
                net_activation=jax.nn.gelu,
                last_layer_size=32,
                out_size=1,
                W_init = jax.nn.initializers.he_normal(),
                b_init = jax.nn.initializers.zeros,
                )

    early_stopping = create_early_stopping(1000, 5, metric_name="loss", tol=1e-4)
    training_kwargs_uat = dict(
        batch_size=512,
        epochs=1000,
        lr=1e-3,
        optim="adam",
    )

    loss_fun = binary_cross_entropy(l2_reg=1e-3, dropout_reg=1e-8)
    
    # create data, initialize models, train and record distances bootstraps
    d1_uat, d2_uat = [], []
    drop_probs = []
    rng = np.random.default_rng(42)
    for i in range(iterations):
        key = rng.integers(9999)
        X, X_test, _, y, y_test, _, _, _  = data.spiral(2048, missing=missing, rng_key=key)
        # steps_per_epoch = X_train.shape[0] // training_kwargs_uat["batch_size"]
        stop_steps = 500*3
        early_stopping = create_early_stopping(stop_steps, 50, metric_name="loss", tol=1e-8)
        training_kwargs_uat["X_test"] = X_test
        training_kwargs_uat["y_test"] = y_test
        training_kwargs_uat["early_stopping"] = early_stopping
        key = rng.integers(9999)
        model1 = UAT(
            model_kwargs=model_kwargs_uat,
            training_kwargs=training_kwargs_uat,
            loss_fun=loss_fun,
            rng_key=key,
        )
        model1.fit(X, y)
        # print dropout p
        p_drop = jax.nn.sigmoid(model1.params["logits"])
        print(p_drop)
        drop_probs.append(p_drop)
        d1, d2 = model1.distances(X, y)
        d1_uat.append(d1)
        d2_uat.append(d2)
    
    d1_uat = pd.concat(d1_uat)
    d2_uat = pd.concat(d2_uat)
    drop_probs = np.mean(np.stack(drop_probs, axis=0), axis=0)

    def p_value1(df):
        df["diff"] = df["+noise"] - df["+signal"]
        df_mean = df.groupby(df.index).mean()
        df_std = df.groupby(df.index).std(ddof=1)

        # calculate t-test
        t_stat = df_mean["diff"].values / (df_std["diff"].values / np.sqrt(iterations))
        p = t.cdf(t_stat, df=(2*iterations - 2))
        p = 2 * np.minimum(p, 1-p) # two tailed t-test
        df_mean["p-value"] = p
        return df_mean[["+noise", "+signal", "p-value"]]
    
    def p_value2(df):
        df_mean = df.groupby(df.index).mean()
        df_std = df.groupby(df.index).std(ddof=1)
        # calculate t-test
        t_stat = df_mean["{}"].values / (df_std["{}"].values / np.sqrt(iterations))
        p = t.cdf(t_stat, df=(2*iterations - 2))
        p = 2 * np.minimum(p, 1-p) # two tailed t-test
        df_mean["p-value"] = p
        return df_mean[["{}", "p-value"]]

    
    print(missing)
    d1_uat = p_value1(d1_uat)
    d2_uat = p_value2(d2_uat)

    print(d1_uat)
    print(d2_uat)
    # d1_uat.to_csv("results/latent_space/UAT_Distances1_{}.csv".format(missing))
    # d2_uat.to_csv("results/latent_space/UAT_Distances2_{}.csv".format(missing))
    return d1_uat, d2_uat, drop_probs



if __name__ == "__main__":
    parser = argparse.ArgumentParser("train model")
    parser.add_argument("iterations", default=10, type=int)
    parser.add_argument("--missing", choices=[None, "MNAR"], default=None)
    args = parser.parse_args()

    if args.missing is None:
        missing = [0.0, 0.2, 0.4, 0.6, 0.8, 0.99]
    elif args.missing == "MNAR":
        missing = [(0.0, "MNAR"), (0.2, "MNAR"), (0.4, "MNAR"), (0.6, "MNAR"), (0.8, "MNAR"), (0.99, "MNAR")]
    d1 = {}
    probs={}
    for m in missing:
    # for m in [0.2]:
        d1_uat, d2_uat, drop_probs = run(args.iterations, missing=m)
        d1[str(m)] =  d1_uat["+signal"].values
        probs[str(m)] = drop_probs.flatten()

    print( pd.DataFrame(d1, index=d1_uat.index.values))
    print( pd.DataFrame(probs, index=["x1", "x2", "x3", "signal"]))
    pd.DataFrame(d1, index=d1_uat.index.values).to_csv("results/latent_space/UAT_Distances1_missingness_{}.csv".format(args.missing))
    pd.DataFrame(probs, index=["x1", "x2", "x3", "signal"]).to_csv("results/latent_space/UAT_drop_probs_missingness_{}.csv".format(args.missing))
