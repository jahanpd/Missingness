# %%
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
from UAT import binary_cross_entropy, mse

import matplotlib.pyplot as plt

# X, y = data.banking()

# import some data to play with
bc = datasets.load_breast_cancer()
X = bc.data # [:, :2]  # we only take the first two features.
y = bc.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
ros = RandomOverSampler(random_state=1)
X_train, y_train = ros.fit_resample(X_train, y_train)


model_kwargs_uat = dict(
                features=X.shape[1],
                d_model=64,
                embed_hidden_size=64,
                embed_hidden_layers=2,
                embed_activation=jax.nn.relu,
                encoder_layers=2,
                encoder_heads=3,
                enc_activation=jax.nn.relu,
                decoder_layers=2,
                decoder_heads=3,
                dec_activation=jax.nn.relu,
                net_hidden_size=32,
                net_hidden_layers=2,
                net_activation=jax.nn.relu,
                last_layer_size=8,
                W_init = jax.nn.initializers.glorot_uniform(),
                b_init = jax.nn.initializers.normal(0.01),
                )

model_kwargs_nn = dict(
                in_dim=X.shape[1],
                hidden_dim=512,
                out_dim=2,
                num_hidden=5,
                activation=jax.nn.relu
                )

training_kwargs = dict(
    batch_size=32,
    epochs=200,
    lr=1e-4,
    optim="sgd",
    X_test=X_test,
    y_test=y_test
)
loss_fun_bce = binary_cross_entropy(l2_reg=1e-4, dropout_reg=1e-5)

def cross_entropy(
    classes,
    l2_reg=1e-2
    ):
    def loss_fun(params, output, labels):
            logits = output
            one_hot = jax.nn.one_hot(labels, classes)
            probs = jax.nn.sigmoid(logits)
            @jax.vmap
            def cross_entropy(probs, labels):
                return -(labels * jnp.log(probs + 1e-7) 
                        + (1-labels) * jnp.log(1 - probs + 1e-7)).sum()

            ce = cross_entropy(probs, one_hot).mean()
            l2 = l2_norm(params)
            loss = ce + l2_reg*l2

            loss_dict = {
                "bce":ce,
                "l2":l2,
                }

            return loss, loss_dict
    return loss_fun

loss_fun_ce = cross_entropy(l2_reg=1e-10, classes=2)
# loss_fun = mse(l2_reg=1e-4, dropout_reg=1e-5)
# model = UAT(
#     model_kwargs=model_kwargs_uat,
#     training_kwargs=training_kwargs,
#     loss_fun=loss_fun_bce
# )
model = NeuralNet(
    model_kwargs=model_kwargs_nn,
    training_kwargs=training_kwargs,
    loss_fun=loss_fun_ce
)
model.fit(X, y)

# %%

history = pd.DataFrame(model.history())
print(list(history))
history = history.groupby(['epoch']).agg({"test_loss": "mean", "prop": "mean"})
print(history)
print(list(history))
# %%
# create figure and axis objects with subplots()
fig,ax = plt.subplots()
# make a plot
ax.plot(np.arange(len(history)), history["test_loss"], color="red")
# set x-axis label
ax.set_xlabel("steps",fontsize=14)
# set y-axis label
ax.set_ylabel("test loss",color="red",fontsize=14)
# twin object for two different y-axis on the sample plot
ax2=ax.twinx()
# make a plot with different y-axis using second axis object
ax2.plot(np.arange(len(history)), history["prop"],color="blue")
ax2.set_ylabel("Prop Weights On",color="blue",fontsize=14)
plt.show()
# %%
