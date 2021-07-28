import functools
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental.optimizers import l2_norm
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import UAT.datasets as data
from UAT import UAT, NeuralNet
from UAT import mse_, cross_entropy_

import matplotlib.pyplot as plt

def run(dataset):
    if dataset == "wisconsin":
        bc = datasets.load_breast_cancer()
        X = bc.data # [:, :2]  # we only take the first two features.
        y = bc.target
        outcomes = 2 
        loss_fun_ce = cross_entropy_(l2_reg=1e-10, classes=outcomes)
        t = "classification"
        lr = 1e-4
        p = 0.2

    if dataset == "housing":
        d = datasets.load_boston()
        X = d.data # [:, :2]  # we only take the first two features.
        y = d.target
        outcomes = 1 
        loss_fun_ce = mse_(l2_reg=1e-10)
        t = "regression"
        lr = 1e-4
        p = 0.15

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
    
    if t == "classification":
        ros = RandomOverSampler(random_state=1)
        X_train, y_train = ros.fit_resample(X_train, y_train)

    model_kwargs_nn = dict(
                    in_dim=X.shape[1],
                    hidden_dim=512,
                    out_dim=outcomes,
                    num_hidden=5,
                    activation=jax.nn.relu
                    )
    
    results = {}
    for optim in ["adascore", "adam", "adabelief", "sgd"]:
        training_kwargs = dict(
            batch_size=32,
            epochs=200,
            lr=lr,
            optim=optim,
            X_test=X_test,
            y_test=y_test,
            p=p
        )
        model = NeuralNet(
            model_kwargs=model_kwargs_nn,
            training_kwargs=training_kwargs,
            loss_fun=loss_fun_ce
        )
        model.fit(X, y)
        history = pd.DataFrame(model.history())
        history = history.groupby(['epoch']).agg({"loss": "mean", "test_loss": "mean", "prop": "mean"})
        results[optim] = history
    
    fig, ax = plt.subplots()
    # make a plot
    lns = []
    for optim in ["adam", "adabelief", "sgd", "adascore"]:
        ln = ax.plot(np.arange(len(results[optim])), results[optim]["test_loss"], label = optim)
        lns = lns + ln
        # set x-axis label
        ax.set_xlabel("steps",fontsize=14)
        # set y-axis label
        ax.set_ylabel("Test Loss", fontsize=14)
        # twin object for two different y-axis on the sample plot
    ax2=ax.twinx()
    # make a plot with different y-axis using second axis object
    ln = ax2.plot(np.arange(len(results["adascore"])), results["adascore"]["prop"], label = "adascore weights", linestyle='dotted')
    lns = lns + ln
    labs = [l.get_label() for l in lns]
    ax2.set_ylabel("Proportion Weights On",color="blue",fontsize=14)
    ax.legend(lns, labs)
    plt.show()


# run(dataset="wisconsin")
run(dataset="housing")