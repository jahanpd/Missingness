import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from imblearn.over_sampling import RandomOverSampler
import UAT.datasets as data
from UAT import UAT, create_early_stopping
from UAT import binary_cross_entropy, cross_entropy, mse
import matplotlib.pyplot as plt

# X_train, X_valid, X_test, y_train, y_valid, y_test, _, classes = data.spiral(
#                 2048,
#                 missing=None,
#                 imputation=None,  # one of none, simple, iterative, miceforest
#                 train_complete=True,
#                 test_complete=True,
#                 split=0.2,
#                 rng_key=1)

# X_train, X_valid, X_test, y_train, y_valid, y_test, classes = data.thoracic(
#                 missing=None,
#                 imputation=None,  # one of none, simple, iterative, miceforest
#                 train_complete=True,
#                 test_complete=True,
#                 split=0.2,
#                 rng_key=1,
#             )

X_train, X_valid, X_test, y_train, y_valid, y_test, classes = data.banking(
                imputation=None,
                split=0.2,
                rng_key=1)

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
    out_size=2,
    W_init = jax.nn.initializers.glorot_uniform(),
    b_init = jax.nn.initializers.normal(0.0001),
    )

# define training parameters
batch_size = 1024
stop_epochs = 100
wait_epochs = 500

training_kwargs_uat = dict(
    batch_size=batch_size,
    epochs=1000,
    lr=5e-4,
    optim="adam",
)
loss_fun = cross_entropy(2, l2_reg=1e-4, dropout_reg=1e-5)

# get valdation for early stopping and add to training kwargs
training_kwargs_uat["X_test"] = X_valid
training_kwargs_uat["y_test"] = y_valid
steps_per_epoch = X_train.shape[0] // batch_size
stop_steps_ = steps_per_epoch * stop_epochs
early_stopping = create_early_stopping(stop_steps_, wait_epochs, metric_name="loss", tol=1e-8)
training_kwargs_uat["early_stopping"] = early_stopping

model = UAT(
            model_kwargs=model_kwargs_uat,
            training_kwargs=training_kwargs_uat,
            loss_fun=loss_fun,
            rng_key=1,
        )

model.fit(X_train, y_train)

attn = model.attention(X_test[:10, :])

print(attn.shape)

cont = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
cat = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
names = cont + cat

fig, ax = plt.subplots()
im = ax.imshow(attn[:10, 0, :], cmap = "magma_r" ,interpolation='nearest')
ax.set_xticks(np.arange(len(names)))
ax.set_xticklabels(names)
ax.set_yticks(np.arange(10))
ax.set_yticklabels(["Sample " + str(i) for i in range(1,11)])

plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel("Attention", rotation=-90, va="bottom")

# Turn spines off and create white grid.
ax.spines[:].set_visible(False)
ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
ax.tick_params(which="minor", bottom=False, left=False)

plt.show()