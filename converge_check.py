import argparse
import pickle
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
print(devices)

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
            max_steps=5e3,
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
        batch_size_base2 = 2 ** int(np.round(np.log2(rows/20)))
        # batch_size_base2 = 64
        steps_per_epoch = max(rows // batch_size_base2, 1)
        epochs = max_steps // steps_per_epoch
        while epochs < 100:
            batch_size_base2 *= 2
            steps_per_epoch = max(rows // batch_size_base2, 1)
            epochs = max_steps // steps_per_epoch
            print(epochs)

        freq = 5
        print("lr: {}, d: {}, depth: {}, reg: {}, b2: {}".format(
            np.exp(lr_max), int(embed_depth), int(depth), reg, b2))
        model_kwargs_uat = dict(
                features=features,
                d_model=32,
                embed_hidden_size=32,
                embed_hidden_layers=int(embed_depth),
                embed_activation=jax.nn.gelu,
                encoder_layers=5,
                encoder_heads=4,
                enc_activation=jax.nn.gelu,
                decoder_layers=int(depth),
                decoder_heads=4,
                dec_activation=jax.nn.gelu,
                net_hidden_size=32,
                net_hidden_layers=3,
                net_activation=jax.nn.gelu,
                last_layer_size=32,
                out_size=classes,
                W_init = jax.nn.initializers.he_normal(),
                b_init = jax.nn.initializers.zeros,
                )
        epochs = int(max_steps // steps_per_epoch)
        start_steps = 3*steps_per_epoch # wait at least 5 epochs before early stopping
        stop_steps_ = steps_per_epoch * (epochs // 4) / min(steps_per_epoch, freq)

        # definint learning rate schedule
        m = max_steps // 2
        n_cycles = 3
        decay = piecewise_constant_schedule(
            np.exp(lr_max),
            # 1e-3,
            boundaries_and_scales={
                int(epochs * 0.8 * steps_per_epoch):0.1,
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
                lr=5e-4,
                batch_size=batch_size_base2
                )
            )
        return model, batch_size_base2, loss_fun
    return make_model



data_list = data.get_list(0, key=12, test=lambda x, m: x == m)
data_list["task_type"] = ["Supervised Classification" if x > 0 else "Supervised Regression" for x in data_list.NumberOfClasses]

data_list = data_list.reset_index().sort_values(by=['NumberOfInstances'])

rng = np.random.default_rng(1234)
key = rng.integers(9999)
ros = RandomOverSampler(random_state=key)
class_filter = np.array([x == "Supervised Classification" for x in data_list.task_type])
selection = np.arange(len(data_list))[class_filter]

print(
data_list[
    ['name', 'NumberOfInstances', 'NumberOfFeatures', 'NumberOfClasses', 'NumberOfNumericFeatures', 'NumberOfSymbolicFeatures']
    ].loc[data_list.index[selection]]
)

for row in data_list[['did', 'task_type', 'name', 'NumberOfFeatures']].values[selection,:]:
    print(row[1], row[2], row[3])
    key = rng.integers(9999)
    X, y, classes, cat_bin = data.prepOpenML(row[0], row[1])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=key)
    key = rng.integers(9999)
    X_train, X_test, X_valid, y_train, y_test, y_valid, diagnostics = data.openml_ds(
            X_train,
            y_train,
            X_test,
            y_test,
            row[1],
            cat_bin=cat_bin,
            classes=classes,
            missing=None,
            imputation=None,  # one of none, simple, iterative, miceforest
            train_complete=False,
            test_complete=True,
            split=0.2,
            rng_key=key,
            prop=0.7,
            corrupt=True,
            cols_miss=int(X.shape[1] * 0.8)
        )
    key = rng.integers(9999)
    if row[1] == "Supervised Classification":
        objective = 'softmax'
        X_train, y_train = ros.fit_resample(X_train, y_train)
    else:
        objective = 'regression'
        resample=False
    
    make_key = rng.integers(9999)
    make_model = create_make_model(X_train.shape[1], X_train.shape[0], row[1], make_key)

    def black_box(
            lr_max=np.log(5e-4),
            reg=6,
            embed_depth=5,
            depth=5,
            b2=0.99
    ):
        model, batch_size_base2, loss_fun = make_model(
            X_valid, y_valid, classes=classes,
            reg=reg, lr_max=lr_max, embed_depth=embed_depth,
            depth=depth, b2=b2,
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
        baseline = np.sum(y_test) / y_test.shape[0]
        acc = acc_loop / len(test_batches)
        diff = np.abs(acc - 0.5) - np.abs(baseline - 0.5)
        print(
            loss_loop/len(test_batches),
            acc_loop/len(test_batches),
            diff)
        # return - loss_loop / len(test_batches)
        return diff / (loss_loop / len(test_batches))
    
    black_box()

