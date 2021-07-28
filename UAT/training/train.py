from jax._src.api import jit
from UAT.aux import flatten_params, unflatten_params
from UAT.optimizers.optimizer import adabelief, adascore
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
from scipy.stats import halfnorm
import functools


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
    optim="sgd",
    X_test=None,
    y_test=None,
    p=0.2
    ):
    
    devices = jax.local_device_count()
    print(devices, " device(s)")
    params = jax.tree_map(lambda x: jnp.array([x] * devices), params)
    in_ax1 = jax.tree_map(lambda x: 0, params)

    @jax.jit
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
    if optim == "sgd":
        print("optimizer: sgd, lr: ", step_size)
        optim_init, optim_update, optim_params = optimizers.sgd(step_size)
    
    elif optim == "adascore":
        optim_init, optim_update, optim_params, get_other = adascore(
            step_size, p=p, b1=0.9, b2=0.9, b3=0.999, eps=1e-8)
        thresh = halfnorm.ppf(p)
        print("optimizer: adascore, lr: {}, thresh: {}".format(step_size, thresh))
        jit_other = jax.jit(get_other)

        # this function is for determining proportion of weights 'on'
        def weights_on_fn(opt_state):
            I = jit_other(opt_state)
            I = jax.tree_map(lambda x: jnp.where(x != 0, 1.0, 0.0), I)
            leaves, _ = jax.tree_flatten(I)
            weights_on = np.sum([np.sum(x) for x in leaves])
            total = np.sum([jnp.size(x) for x in leaves])
            return weights_on, total
    
    elif optim == "adam":
        print("optimizer: adam, lr: ", step_size)
        optim_init, optim_update, optim_params = optimizers.adam(
            step_size, b1=0.9, b2=0.99, eps=1e-8)
    
    elif optim == "adabelief":
        print("optimizer: adabelief, lr: ", step_size)
        optim_init, optim_update, optim_params = adabelief(
            step_size, b1=0.9, b2=0.99, eps=1e-8)

    opt_state = optim_init(params)
    in_ax2 = jax.tree_map(lambda x: 0, opt_state)

    @functools.partial(
        jax.pmap,
        axis_name='num_devices',
        in_axes=(None, in_ax1, 0, 0 , None, in_ax2),
        out_axes=(in_ax1, in_ax2),
        static_broadcasted_argnums=0)
    def take_step(step, params, x_batch, y_batch, rng, opt_state):
        """ Compute the gradient for a batch and update the parameters """        
        grads = jax.jit(jax.grad(loss))(params, x_batch, y_batch, rng)
        grads = jax.lax.pmean(grads, axis_name='num_devices')
        opt_state = jax.jit(optim_update)(step, grads, opt_state)
        return optim_params(opt_state), opt_state

    if devices == 1:
        take_step = jax.jit(take_step)
    history = []  # store training metrics

    pbar1 = tqdm(total=epochs, leave=True)
    step = 0
    weights_on, total, prop = 0, 0, 1.0
    for epoch in range(epochs):
        rng, key = random.split(rng, 2)
        mod = X.shape[0] % (batch_size*devices)
        rows = random.permutation(key, X.shape[0] - mod)
        batches = np.split(rows, 
            X.shape[0] // (batch_size*devices))
        pbar2 = tqdm(total=len(batches), leave=False)
        if (X_test is not None) and (y_test is not None):
            params_ = jax.device_get(jax.tree_map(lambda x: x[0], params))
            test_loss = loss(params_, X_test, y_test, None)

        for i, b in enumerate(batches):
                
            step += 1
            rng, key = random.split(rng, 2)
            batch_x = X[np.array(b),...]
            batch_y = y[np.array(b)]
            batch_x = batch_x.reshape((devices, batch_size, -1))
            batch_y = batch_y.reshape((devices, batch_size))


            # parallelized step function
            params, opt_state = take_step(
               step, params, batch_x, batch_y, key, opt_state)

            if step % 20 == 0 and optim == "adascore":
                weights_on, total = weights_on_fn(opt_state)

            # initialize some training metrics
            if step == 1:
                params_ = jax.device_get(jax.tree_map(lambda x: x[0], params))
                metrics_ewa = metrics(params_, batch_x.reshape((batch_size*devices,-1)), batch_y.reshape((batch_size*devices,-1)), None)
                metrics_ewa["step"]=step

            pbar2.update(1)
            if step % 5 == 0:
                params_ = jax.device_get(jax.tree_map(lambda x: x[0], params))
                metrics_dict = metrics(params_, batch_x.reshape((batch_size*devices,-1)), batch_y.reshape((batch_size*devices,-1)), None)
                metrics_dict["step"]=step
                metrics_ewa = jax.tree_map(lambda x, y : 0.1 * y + 0.9 * x, metrics_ewa, metrics_dict)
                metrics_ewa["step"]=step

                # evaluate loss on test set for early stopping
                metrics_ewa_ = metrics_ewa.copy()
                metrics_ewa_["test_loss"] = test_loss
                metrics_ewa_["epoch"] = epoch
                metrics_ewa_["weights_on"] = weights_on
                metrics_ewa_["total_weights"] = total
                metrics_ewa_["prop"] = weights_on / (total + 1e-8)
                # enforce dtype float for dict object while saving
                for k in metrics_ewa_.keys():
                    metrics_ewa_[k] = float(metrics_ewa_[k])
                pbar1.set_postfix(metrics_ewa_)
                history.append(metrics_ewa_)
        pbar2.close()
        pbar1.update(1)

    pbar1.close()
    params = jax.device_get(jax.tree_map(lambda x: x[0], params))
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