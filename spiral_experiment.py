import argparse
from jax.experimental.optimizers import l2_norm
import numpy as np
import pandas as pd
from scipy.stats import t
import jax
import jax.numpy as jnp
from UAT import UAT, Ensemble
from UAT import binary_cross_entropy
from UAT.datasets import spiral
import os

# jax.config.update("jax_debug_nans", True)
# force parallel processing across cpu cores
# os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=12'


def run(iterations, missing="None", epochs=10):
    # define models and training params as dicts 
    model_kwargs_uat = dict(
                features=4,
                d_model=32,
                embed_hidden_size=32,
                embed_hidden_layers=2,
                embed_activation=jax.nn.relu,
                encoder_layers=3,
                encoder_heads=3,
                enc_activation=jax.nn.relu,
                decoder_layers=3,
                decoder_heads=3,
                dec_activation=jax.nn.relu,
                net_hidden_size=32,
                net_hidden_layers=2,
                net_activation=jax.nn.relu,
                last_layer_size=32,
                W_init = jax.nn.initializers.glorot_uniform(),
                b_init = jax.nn.initializers.normal(0.01),
                )
    training_kwargs_uat = dict(
        batch_size=32,
        epochs=10,
        lr=1e-3
    )

    model_kwargs_ens = dict(
        features=4,
        net_hidden_size=128,
        z_size=32,
        net_hidden_layers=2
    )
    training_kwargs_ens = dict(
        batch_size=32,
        epochs=10,
        lr=1e-3
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
                    "bce":bce,
                    "l2":l2,
                    }

                return loss, loss_dict
        return loss_fun
    loss_fun = binary_cross_entropy(l2_reg=1e-3, dropout_reg=1e-5)
    loss_fun2 = binary_cross_entropy2()

    # create data, initialize models, train and record distances bootstraps
    d1_uat, d2_uat = [], []
    d1_ens, d2_ens = [], []
    for i in range(iterations):
        X, y, _ = spiral(2048, missing=missing, rng_key=i)
        model1 = UAT(
            model_kwargs=model_kwargs_uat,
            training_kwargs=training_kwargs_uat,
            loss_fun=loss_fun,
            rng_key=i,
        )
        model1.fit(X, y)
        d1, d2 = model1.distances(X, y)
        d1_uat.append(d1)
        d2_uat.append(d2)

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
    
    d1_uat = pd.concat(d1_uat)
    d2_uat = pd.concat(d2_uat)
    d2_uat = d2_uat.groupby(d2_uat.index).mean()
    
    d1_ens = pd.concat(d1_ens)
    d2_ens = pd.concat(d2_ens)
    d2_ens = d2_ens.groupby(d2_ens.index).mean()

    def p_value(df):
        df["diff"] = df["+noise"] - df["+signal"]
        df_mean = df.groupby(df.index).mean()
        df_std = df.groupby(df.index).std(ddof=1)

        # calculate t-test
        t_stat = df_mean["diff"].values / (df_std["diff"].values / np.sqrt(iterations))
        p = t.cdf(t_stat, df=(2*iterations - 2))
        p = 2 * np.minimum(p, 1-p) # two tailed t-test
        df_mean["p-value"] = p
        return df_mean[["+noise", "+signal", "p-value"]]
    
    print(missing)
    d1_uat = p_value(d1_uat)
    d1_ens = p_value(d1_ens)

    print(d1_uat)
    print(d1_ens)
    d1_uat.to_csv("results/UAT_Distances1_{}.csv".format(missing))
    d1_ens.to_csv("results/Ensemble_Distances1_{}.csv".format(missing))
    d2_uat.to_csv("results/UAT_Distances2_{}.csv".format(missing))
    d2_ens.to_csv("results/Ensemble_Distances2_{}.csv".format(missing))



if __name__ == "__main__":
    parser = argparse.ArgumentParser("train model")
    parser.add_argument("iterations", default=10, type=int)
    args = parser.parse_args()

    run(args.iterations, missing="None")
    run(args.iterations, missing="MCAR")
    run(args.iterations, missing="MAR")
    run(args.iterations, missing="MNAR")