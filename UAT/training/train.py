from UAT.aux import flatten_params, unflatten_params
import numpy as np
import jax.numpy as jnp
import jax
from jax import random
from jax.experimental import optimizers
from flax import linen as nn
from flax import optim
from flax import serialization
from tqdm import tqdm
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import\
    roc_auc_score, brier_score_loss, recall_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold



def training_loop_alt(
    X,
    y,
    model_fun,
    params,
    loss_fun,
    metric_fun,
    rng_key=0,
    batch_size=32,
    epochs=100,
    lr=1e-3,
    sampling=None,
    sampling_optim=None,
    sampling_params=None,
    save=None,
    **kwargs
    ):

    rng = random.PRNGKey(rng_key)

    
    def loss(params, x_batch, y_batch, rng):
        out = model_fun(params, x_batch, rng)
        loss, _ = loss_fun(params, out, y_batch)
        return loss
    
    @jax.jit
    def metrics(params, x_batch, y_batch, rng):
        out = model_fun(params, x_batch, rng)
        _, metric_dict = metric_fun(params, out, y_batch)
        return metric_dict

    step_size = lr
    sgd_init, sgd_update, sgd_params = optimizers.sgd(step_size)
    # sgd_init, sgd_update, sgd_params = adabelief(step_size, b1=0.9, b2=0.999, eps=1e-8)
    opt_state = sgd_init(params)

    if sampling is not None:
        # flatten params
        ll_flat, recon = flatten_params(params["last_layer"])
        sampling_params["recon"] = recon
        
        
        def sample_loss(last_layer, params, x_batch, y_batch, rng):
            params["last_layer"] = unflatten_params(last_layer, recon)
            out = model_fun(params, x_batch, rng)
            loss, _ = loss_fun(params, out, y_batch)
            return loss
        hessian_jit = jax.jit(jax.hessian(sample_loss))
        sampling_params["hessian_fn"] = hessian_jit

        sample_init, sample_update, sample_params = sampling_optim(**sampling_params)
        sample_state = sample_init(ll_flat)
        
        # jit functions
        grad_jit = jax.jit(jax.grad(sample_loss))
        update_jit = jax.jit(sample_update, static_argnames="i")
        @jax.jit
        def sample_step(step, params, x_batch, y_batch, rng, sample_state):
            last_layer = params["last_layer"]
            grads = grad_jit(last_layer, params, x_batch, y_batch, rng)
            sample_state = sample_update(step, grads, sample_state, rng=rng)
            return sample_params(sample_state), sample_state

    @jax.jit
    def sgd_step(step, params, x_batch, y_batch, rng, opt_state):
        """ Compute the gradient for a batch and update the parameters """
        grads = jax.grad(loss)(params, x_batch, y_batch, rng)
        opt_state = sgd_update(step, grads, opt_state)
        return sgd_params(opt_state), opt_state
    
    history = []  # store training metrics

    pbar1 = tqdm(total=epochs, leave=False)
    step = 0
    for epoch in range(epochs):
        rng, key = random.split(rng, 2)
        rows = random.permutation(key, X.shape[0])
        batches = np.array_split(rows, 
            X.shape[0] / batch_size)
        pbar2 = tqdm(total=len(batches), leave=True)
        for i, b in enumerate(batches):
            step += 1
            rng, key = random.split(rng, 2)
            batch_x = X[np.array(b),...]
            batch_y = y[np.array(b)]
            params, opt_state = sgd_step(
               step, params, batch_x, batch_y, key, opt_state)

            metrics_dict = metrics(params, batch_x, batch_y, None)

            pbar2.update(1)
            if i % 20 == 0:
                for key in metrics_dict.keys():
                    metrics_dict[key] = np.mean(
                        [l[key] for l in history]
                    )
                pbar2.set_postfix(metrics_dict)
            else:
                pbar2.set_postfix(metrics_dict)

            metrics_dict["epoch"]=epoch
            metrics_dict["step"]=step
            history.append(metrics_dict)

        pbar2.close()
        pbar1.update(1)
    pbar1.close()

    if sampling is not None:
        print("sampling")
        samples_set = []
        pbar1 = tqdm(total=sampling["iterations"], leave=True)
        i = 1
        # set last layer to flattened array for sampling
        MAP = params["last_layer"]
        rng, key = random.split(rng, 2)
        params["last_layer"] = random.normal(key, ll_flat.shape)
        while i < sampling["iterations"]:
            rng, key = random.split(rng, 2)
            rows = random.permutation(key, X.shape[0])
            batches = jnp.array_split(rows, 
                X.shape[0] / batch_size)
            
            for _, b in enumerate(batches):
                rng, key = random.split(rng, 2)
                batch_x = X[np.array(b),...]
                batch_y = y[np.array(b)]

                last_layer, sample_state = sample_step(
                i, params, batch_x, batch_y, key, sample_state)
                params["last_layer"] = last_layer

                if (i > sampling["burn_in"]) and (i % sampling["c"] == 0):
                    samples_set.append(unflatten_params(last_layer, recon)[0])

                i += 1
                
                if i % 20 == 0:
                    params["last_layer"] = unflatten_params(last_layer, recon)
                    metrics_dict = metrics(params, batch_x, batch_y, None)
                    params["last_layer"] = last_layer
                    pbar1.set_postfix(metrics_dict)
                pbar1.update(1)
                if i > sampling["iterations"]:
                    break

        params["last_layer"] = samples_set
        params["last_layer_map"] = [samples_set[0]]
        pbar1.close()

    return params, history, rng


def training_loop(
    X,
    y,
    model_fun,
    params,
    loss_fun,
    metric_fun,
    rng,
    batch_size=32,
    epochs=100,
    lr=1e-3,
    ):
    
    def loss(params, x_batch, y_batch, rng):
        out = model_fun(params, x_batch, rng)
        loss, _ = loss_fun(params, out, y_batch)
        return loss
    
    @jax.jit
    def metrics(params, x_batch, y_batch, rng):
        out = model_fun(params, x_batch, rng)
        _, metric_dict = metric_fun(params, out, y_batch)
        return metric_dict

    step_size = lr
    sgd_init, sgd_update, sgd_params = optimizers.sgd(step_size)
    # sgd_init, sgd_update, sgd_params = adabelief(step_size, b1=0.9, b2=0.999, eps=1e-8)
    opt_state = sgd_init(params)

    @jax.jit
    def sgd_step(step, params, x_batch, y_batch, rng, opt_state):
        """ Compute the gradient for a batch and update the parameters """
        grads = jax.grad(loss)(params, x_batch, y_batch, rng)
        opt_state = sgd_update(step, grads, opt_state)
        return sgd_params(opt_state), opt_state
    
    history = []  # store training metrics

    pbar1 = tqdm(total=epochs, leave=False)
    step = 0
    for epoch in range(epochs):
        rng, key = random.split(rng, 2)
        rows = random.permutation(key, X.shape[0])
        batches = np.array_split(rows, 
            X.shape[0] / batch_size)
        pbar2 = tqdm(total=len(batches), leave=True)
        for i, b in enumerate(batches):
            step += 1
            rng, key = random.split(rng, 2)
            batch_x = X[np.array(b),...]
            batch_y = y[np.array(b)]
            params, opt_state = sgd_step(
               step, params, batch_x, batch_y, key, opt_state)

            metrics_dict = metrics(params, batch_x, batch_y, None)

            pbar2.update(1)
            if i % 20 == 0:
                for key in metrics_dict.keys():
                    metrics_dict[key] = np.mean(
                        [l[key] for l in history]
                    )
                pbar2.set_postfix(metrics_dict)
            else:
                pbar2.set_postfix(metrics_dict)

            metrics_dict["epoch"]=epoch
            metrics_dict["step"]=step
            history.append(metrics_dict)

        pbar2.close()
        pbar1.update(1)
    pbar1.close()

    return params, history, rng



def cross_validate(
    X,
    y,
    model,
    k=5,
    repeats=3,
    sample=10,
    rng_key=0,
    batch=32,
    epochs=100
    ):
    """
    X: numpy array (nsamples, features)
    y: numpy array (nsamples)
    k: int, number of splits for k fold
    sample: int, if sampling posterior distribution select number of samples
    rng_key: int, random generator key
    batch: int, samples per training batch
    epochs: int, number of training epochs
    """

    skf = RepeatedStratifiedKFold(
        n_splits=k,
        n_repeats=repeats,
        random_state=rng_key)

    metrics = []

    i = rng_key
    for train_idx, test_idx in skf.split(X, y):
        i += 1
        x_train = X[train_idx, ...]
        y_train = y[train_idx, ...]
    
        x_test = X[test_idx, ...]
        y_test = y[test_idx, ...]

        params, history, rng = training_loop(
            x_train,
            y_train,
            model,
            rng_key=i,
            batch=batch,
            epochs=epochs,
            save=None
        )

        # test and training set metrics
        rng, key = random.split(rng, 2)

        d, kld, baseline, attn, self_attn, logit = model().apply(
        {"params":params}, x_test, key, dropout=False, sample=sample)

        print(logit.shape)
        
        logit = logit.reshape(-1, sample)  # (batch, samples)
        logit_var = logit.var(1)  # (nsamples, )

        prob_mu = nn.sigmoid(logit.mean(1))
        base_p = nn.sigmoid(baseline.mean())
        epist_score = 1.0 / (logit_var + 1e-7)
        aleat_score = ((prob_mu * y_test.flatten()) +
                ((1 - prob_mu) * (1 - y_test.flatten())))

        labels = (prob_mu > base_p).astype(np.float32)
        correct = (labels == y_test).astype(np.float32)
        
        auc = roc_auc_score(y_true=y_test, y_score=prob_mu)
        aleat = roc_auc_score(y_true=correct, y_score=aleat_score)
        epist = roc_auc_score(y_true=correct, y_score=epist_score)
        acc = np.mean(correct)

        print(i, auc, aleat, epist, acc)
        m = {
            "auc":auc,
            "sens":0,
            "spec":0,
            "acc":acc,
            "brier":0,
            "auc_conf":0,
            "aleat":aleat,
            "epist":epist
        }
        metrics.append(m)
    
    return metrics