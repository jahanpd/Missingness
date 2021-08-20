from UAT.optimizers.optimizer import adabelief, adascore
import numpy as np
import jax.numpy as jnp
import jax
from jax import random
from jax.experimental import optimizers
from tqdm import tqdm
from scipy.stats import halfnorm
import functools


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
    optim="adabelief",
    optim_kwargs=None,
    frequency=5,
    X_test=None, # either a test set or a string - "proportion"
    y_test=None,
    p=0.2,
    anneal=0.9998, # good range is 0.9990 - 0.9999, where lower reaches thresh faster
    early_stopping=None, # or callable
    ):
    
    devices = jax.local_device_count()
    print(devices, " device(s)")
    params = jax.tree_map(lambda x: jnp.array([x] * devices), params)
    in_ax1 = jax.tree_map(lambda x: 0, params)

    @jax.jit
    def loss(params, x_batch, y_batch, rng):
        out = model_fun(params, x_batch, rng, True)
        loss, _ = loss_fun(params, out, y_batch)
        return loss
    
    @jax.jit
    def metrics(params, x_batch, y_batch, rng):
        out = model_fun(params, x_batch, rng, False)
        _, metric_dict = metric_fun(params, out, y_batch)
        return metric_dict

    step_size = lr
    if optim == "sgd":
        print("optimizer: sgd, lr: ", step_size)
        if optim_kwargs is None:
            optim_kwargs = dict(step_size=step_size)
        else:
            optim_kwargs["step_size"] = step_size
        
        optim_init, optim_update, optim_params = optimizers.sgd(**optim_kwargs)
    
    elif optim == "adascore":
        if optim_kwargs is None:
            optim_kwargs = dict(
                step_size=step_size, p=p, b1=0.9, b2=0.9, b3=0.999, eps=1e-8, anneal=anneal
            )
        else:
            optim_kwargs["step_size"] = step_size
            optim_kwargs["p"] = p
            optim_kwargs["anneal"] = anneal

        optim_init, optim_update, optim_params, get_other = adascore(**optim_kwargs)
        thresh = halfnorm.ppf(p)
        print("optimizer: adascore, lr: {}, thresh: {}".format(step_size, thresh))
        jit_other = jax.jit(get_other)

        # this function is for determining proportion of weights 'on'
        def weights_on_fn(opt_state, step):
            m, v, z, thresh_ = jit_other(opt_state)
            z, _ = jax.tree_flatten(z)
            z = np.concatenate([x.ravel() for x in z])
            if type(anneal) == float:
                schedule = lambda x: 100 * anneal**(x) + 1
            else:
                schedule = lambda x : 1.0
            weights_on = np.sum(z >= thresh / schedule(step))
            total = np.size(z)

            out_dict = {
                "weights_on":weights_on,
                "total":total,
                "prop":weights_on / (total + 1e-8),
                "thresh":thresh / schedule(step),
                "z": np.mean(z),
                "z_max":np.max(z),
                "z_min":np.min(z),
            }
            return out_dict
    
    elif optim == "adam":
        print("optimizer: adam, lr: ", step_size)
        if optim_kwargs is None:
            optim_kwargs = dict(
                step_size=step_size, b1=0.9, b2=0.99, eps=1e-8
            )
        else:
            optim_kwargs["step_size"] = step_size

        optim_init, optim_update, optim_params = optimizers.adam(**optim_kwargs)
    
    elif optim == "adabelief":
        print("optimizer: adabelief, lr: ", step_size)
        if optim_kwargs is None:
            optim_kwargs = dict(
                step_size=step_size, b1=0.9, b2=0.99, eps=1e-8
            )
        else:
            optim_kwargs["step_size"] = step_size
        optim_init, optim_update, optim_params = adabelief(**optim_kwargs)

    opt_state = optim_init(params)
    in_ax2 = jax.tree_map(lambda x: 0, opt_state)

    @functools.partial(
        jax.pmap,
        axis_name='num_devices',
        in_axes=(None, in_ax1, 0, 0 , 0, in_ax2),
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

    if optim == "adascore":
        grad_dict = weights_on_fn(opt_state, step+1)
    else:
        grad_dict = {}

    for epoch in range(epochs):
        rng, key = random.split(rng, 2)
        mod = X.shape[0] % (batch_size*devices)
        rows = random.permutation(key, X.shape[0] - mod)
        batches = np.split(rows, 
            X.shape[0] // (batch_size*devices))
        pbar2 = tqdm(total=len(batches), leave=False)
        
        # evaluate test set performance OR early stopping test
        perform_test = ((X_test is not None) and (y_test is not None)) and early_stopping is not None
        if perform_test:
            params_ = jax.device_get(jax.tree_map(lambda x: x[0], params))
            if step == 0:
                params_store = params_.copy()
                metric_store = None
            if X_test == y_test == "proportion":
                test_dict = {"loss":np.nan}
            else:
                key_ = jnp.ones((X_test.shape[0], 2))
                test_dict = metrics(params_, X_test, y_test, key_)
            for k, v in grad_dict.items():
                test_dict[k] = v
            # test for early stopping
            # early stopping is a function that returns a bool, params_store, metric_store
            stop, params_store, metric_store = early_stopping(
                step, params_, params_store, test_dict, metric_store)
            if stop:
                return params_store, history, rng



        for i, b in enumerate(batches):
                
            step += 1
            rng, key = random.split(rng, 2)
            key = random.split(key, devices * batch_size).reshape((devices, batch_size, -1))
            batch_x = X[np.array(b),...]
            batch_y = y[np.array(b)]
            batch_x = batch_x.reshape((devices, batch_size, -1))
            batch_y = batch_y.reshape((devices, batch_size))


            # parallelized step function
            params, opt_state = take_step(
               step, params, batch_x, batch_y, key, opt_state)

            if step % frequency == 0 and optim == "adascore":
                grad_dict = weights_on_fn(opt_state, step)

            # initialize some training metrics
            if step == 1:
                params_ = jax.device_get(jax.tree_map(lambda x: x[0], params))
                metrics_ewa = metrics(params_, batch_x.reshape((batch_size*devices,-1)), batch_y.reshape((batch_size*devices,-1)), None)
                metrics_ewa["step"]=step

            pbar2.update(1)
            if step % frequency == 0:
                params_ = jax.device_get(jax.tree_map(lambda x: x[0], params))
                metrics_dict = metrics(params_, batch_x.reshape((batch_size*devices,-1)), batch_y.reshape((batch_size*devices,-1)), None)
                metrics_dict["step"]=step
                metrics_ewa = jax.tree_map(lambda x, y : 0.1 * y + 0.9 * x, metrics_ewa, metrics_dict)
                metrics_ewa["step"]=step

                metrics_ewa_ = metrics_ewa.copy()
                # evaluate loss on test set for early stopping
                if perform_test:
                    metrics_ewa_["test_loss"] = test_dict["loss"]
                    metrics_ewa_["test_counter"] = metric_store["counter"]
                    metrics_ewa_["epoch"] = epoch
                if optim == "adascore":
                    for key in grad_dict.keys():
                        metrics_ewa_[key] = grad_dict[key]

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
