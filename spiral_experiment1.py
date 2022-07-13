import argparse
from jax.example_libraries.optimizers import l2_norm
import numpy as np
import pandas as pd
from scipy.stats import t
import jax
import jax.numpy as jnp
from LSAM import LSAM, Ensemble, create_early_stopping
from LSAM import binary_cross_entropy
import LSAM.datasets as data
import os

# jax.config.update("jax_debug_nans", True)
# force parallel processing across cpu cores
# os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=12'

"""
Script for the first spiral dataset experiment exploring the effect of information in the latent space
based on a signal variable and noise variable.
"""

def run(iterations, missing="None", epochs=10):
    # define models and training params as dicts 
    model_kwargs_uat = dict(
                features=4,
                d_model=16,
                embed_hidden_size=16,
                embed_hidden_layers=4,
                embed_activation=jax.nn.gelu,
                encoder_layers=3,
                encoder_heads=3,
                enc_activation=jax.nn.gelu,
                decoder_layers=3,
                decoder_heads=3,
                dec_activation=jax.nn.gelu,
                net_hidden_size=16,
                net_hidden_layers=3,
                net_activation=jax.nn.gelu,
                last_layer_size=32,
                out_size=1,
                W_init = jax.nn.initializers.he_normal(),
                b_init = jax.nn.initializers.zeros,
                )

    early_stopping = create_early_stopping(100, 50, metric_name="loss", tol=1e-8)
    training_kwargs_uat = dict(
        batch_size=512,
        epochs=1000,
        lr=1e-3,
        optim="adam",
        early_stopping=early_stopping,
    )

    model_kwargs_ens = dict(
        features=4,
        net_hidden_size=64,
        z_size=32,
        net_hidden_layers=2
    )
    training_kwargs_ens = dict(
        batch_size=512,
        epochs=1000,
        lr=1e-2,
        optim="adam",
        early_stopping=early_stopping,
    )
    # loss function for ensemble model
    def binary_cross_entropy2(
        l2_reg=1e-3,
        ):
        def loss_fun(params, output, labels):
                logits = output[0]  # shape (batch, features, 1)
                nan_mask = output[2]
                probs = jax.nn.sigmoid(logits)
                @jax.vmap
                def binary_cross_entropy(probs, labels):
                    # probs will be shape (feat, 1)
                    return -(labels * jnp.log(probs + 1e-7) 
                            + (1-labels) * jnp.log(1 - probs + 1e-7)).mean(1)

                scaling = jax.nn.softmax(params["set_order"]).reshape(1, -1)
                bce = (binary_cross_entropy(probs, labels)*nan_mask*scaling).sum(1).mean()
                l2 = l2_norm(params)
                loss = bce + l2_reg*l2

                loss_dict = {
                    "loss":bce,
                    "l2":l2,
                    }

                return loss, loss_dict
        return loss_fun
    loss_fun = binary_cross_entropy(l2_reg=1e-3, dropout_reg=1e-8)
    loss_fun2 = binary_cross_entropy2()
    
    # create data, initialize models, train and record distances bootstraps
    d1_uat, d2_uat = [], []
    d1_ens, d2_ens = [], []
    for i in range(iterations):
        X, X_test, _, y, y_test, _, _, _  = data.spiral(2048, missing=missing, rng_key=i)
        training_kwargs_uat["X_test"] = X_test
        training_kwargs_uat["y_test"] = y_test
        model1 = LSAM(
            model_kwargs=model_kwargs_uat,
            training_kwargs=training_kwargs_uat,
            loss_fun=loss_fun,
            rng_key=i,
        )
        model1.fit(X, y)
        d11, d22 = model1.distances(X, y)
        d1_uat.append(d11)
        d2_uat.append(d22)

        training_kwargs_ens["X_test"] = X_test
        training_kwargs_ens["y_test"] = y_test
        model2 = Ensemble(
            model_kwargs=model_kwargs_ens,
            training_kwargs=training_kwargs_ens,
            loss_fun=loss_fun2,
            rng_key=i,
        )
        model2.fit(X, y)
        d1,d2 = model2.distances(X, y)
        d1_ens.append(d1)
        d2_ens.append(d2)
        print(i, missing)

    d1_uat = pd.concat(d1_uat)
    d2_uat = pd.concat(d2_uat)

    d1_ens = pd.concat(d1_ens)
    d2_ens = pd.concat(d2_ens)

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
    d1_ens = p_value1(d1_ens)
    d2_uat = p_value2(d2_uat)
    d2_ens = p_value2(d2_ens)

    print(d1_uat)
    print(d2_uat)
    print(d1_ens)
    print(d2_ens)
    d1_uat.to_csv("results/latent_space/LSAM_Distances1_{}.csv".format(missing))
    d1_ens.to_csv("results/latent_space/Ensemble_Distances1_{}.csv".format(missing))
    d2_uat.to_csv("results/latent_space/LSAM_Distances2_{}.csv".format(missing))
    d2_ens.to_csv("results/latent_space/Ensemble_Distances2_{}.csv".format(missing))



if __name__ == "__main__":
    parser = argparse.ArgumentParser("train model")
    parser.add_argument("iterations", default=10, type=int)
    args = parser.parse_args()

    run(args.iterations, missing="None")
    # run(args.iterations, missing="MCAR")
    # run(args.iterations, missing="MAR")
    # run(args.iterations, missing="MNAR")
