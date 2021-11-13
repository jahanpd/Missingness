from UAT.optimizers.optimizer import adabound
import optax
import numpy as np
import jax.numpy as jnp
import jax
from jax import random
from jax.experimental import optimizers
from jax.experimental.optimizers import make_schedule
from tqdm import tqdm
from scipy.stats import halfnorm
import functools
import time
from datetime import timedelta
# from UAT.aux import flatten_params, unflatten_params

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
    start_score=100, # epochs when to start 
    early_stopping=None, # or callable
    output_freq=5,
    early_stop=True # if makes it to end of training cycle, return early stop. Also shuts down early stopping for HP optim
    ):
    devices = jax.local_device_count()
    assert batch_size % devices == 0, "batch size must be divisible by number of devices"
    tqdm.write("{} device(s)".format(devices))
    params_flat, params_build = jax.tree_util.tree_flatten(params)
    params_ = params
    params = jax.tree_map(lambda x: jnp.array([x] * devices), params)
    in_ax1 = jax.tree_map(lambda x: 0, params)
    if optim_kwargs is not None:
        if 'k' in optim_kwargs.keys() and 'm' in optim_kwargs.keys():
            print("swa implemented")
            swa = True
        else:
            swa = False
    else:
        swa = False
    
    def loss(params, x_batch, y_batch, rng):
        out = model_fun(params, x_batch, rng, True)
        loss, _ = loss_fun(params, out, y_batch)
        return loss
    
    @jax.jit
    def metrics(params, x_batch, y_batch, rng):
        out = model_fun(params, x_batch, rng, False)
        _, metric_dict = metric_fun(params, out, y_batch)
        return metric_dict

    print_step = make_schedule(lr)
    step_size = lr
    if optim == "sgd":
        tqdm.write("optimizer: sgd, lr: {}, batch_size={}".format(print_step(1), batch_size))
        if optim_kwargs is None:
            optim_kwargs = dict(step_size=step_size)
        else:
            optim_kwargs["step_size"] = step_size
        
        optim_init, optim_update, optim_params = optimizers.sgd(**optim_kwargs)
    
    elif optim == "momentum":
        tqdm.write("optimizer: momentum, lr: {}, batch_size={}".format(print_step(1), batch_size))
        if optim_kwargs is None:
            optim_kwargs = dict(step_size=step_size)
        else:
            optim_kwargs["step_size"] = step_size
        
        optim_init, optim_update, optim_params = optimizers.momentum(**optim_kwargs)

    elif optim == "lamb":
        tqdm.write("optimizer: lamb, lr: {}, batch_size={}".format(print_step(1), batch_size))
        if optim_kwargs is None:
            optim_kwargs = dict(
                learning_rate=step_size, b1=0.9, b2=0.99, eps=1e-8, weight_decay=1e-7
            )
        else:
            optim_kwargs["learning_rate"] = step_size
        optobj = optax.lamb(**optim_kwargs)
        optim_init = optobj.init
        optim_update = optobj.update
    
    elif optim == "fromage":
        tqdm.write("optimizer: fromage, lr: {}, batch_size={}".format(print_step(1), batch_size))
        if optim_kwargs is None:
            optim_kwargs = dict(
                learning_rate=step_size,
            )
        else:
            optim_kwargs["learning_rate"] = step_size
        optobj = optax.fromage(**optim_kwargs)
        optim_init = optobj.init
        optim_update = optobj.update
    
    elif optim == "yogi":
        tqdm.write("optimizer: fromage, lr: {}, batch_size={}".format(print_step(1), batch_size))
        if optim_kwargs is None:
            optim_kwargs = dict(
                learning_rate=step_size, b1=0.9, b2=0.99, eps=1e-8
            )
        else:
            optim_kwargs["learning_rate"] = step_size
        optobj = optax.yogi(**optim_kwargs)
        optim_init = optobj.init
        optim_update = optobj.update

    elif optim == "adam":
        tqdm.write("optimizer: adam, lr: {}, batch_size={}".format(print_step(1), batch_size))
        if optim_kwargs is None:
            optim_kwargs = dict(
                step_size=step_size, b1=0.9, b2=0.99, eps=1e-8
            )
        else:
            optim_kwargs["step_size"] = step_size

        optim_init, optim_update, optim_params = optimizers.adam(**optim_kwargs)
    
    elif optim == "adabound":
        tqdm.write("optimizer: adabound, lr: {}, batch_size={}".format(print_step(1), batch_size))
        if optim_kwargs is None:
            optim_kwargs = dict(
                step_size=step_size, b1=0.9, b2=0.995, eps=1e-8
            )
        else:
            optim_kwargs["step_size"] = step_size
        optim_init, optim_update, optim_params = adabound(**optim_kwargs)

    opt_state = optim_init(params)
    opt_state = jax.tree_map(lambda x: x if len(x.shape) > 0 else x.reshape((1)), opt_state)
    in_ax2 = jax.tree_map(lambda x: 0 if len(x.shape) > 0 else None, opt_state)
    if optim in ["lamb", "fromage", "yogi"]:
        @functools.partial(
            jax.pmap,
            axis_name='num_devices',
            in_axes=(None, in_ax1, 0, 0 , 0, in_ax2),
            out_axes=(in_ax1, in_ax2),
            )
        def take_step(step, params, x_batch, y_batch, rng, opt_state):
            """ Compute the gradient for a batch and update the parameters """
            grads = jax.grad(loss)(params, x_batch, y_batch, rng)
            grads = jax.lax.pmean(grads, axis_name='num_devices')
            updates, opt_state = optim_update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state
    else:
        @functools.partial(
            jax.pmap,
            axis_name='num_devices',
            in_axes=(None, in_ax1, 0, 0 , 0, in_ax2),
            out_axes=(in_ax1, in_ax2),
            )
        def take_step(step, params, x_batch, y_batch, rng, opt_state):
            """ Compute the gradient for a batch and update the parameters """        
            grads = jax.grad(loss)(params, x_batch, y_batch, rng)
            grads = jax.lax.pmean(grads, axis_name='num_devices')
            opt_state = optim_update(step, grads, opt_state)
            return optim_params(opt_state), opt_state
    
    if devices == 1:
        take_step = jax.jit(take_step)
    history = []  # store training metrics

    pbar1 = tqdm(total=epochs, position=0, leave=False)
    step = 0

    # set up test batches to avoid OOM errors on test set computation if test set is large
    rng, key = random.split(rng, 2)
    test_batch_size = 128 
    test_mod = X_test.shape[0] % (test_batch_size) if test_batch_size < X_test.shape[0] else 0
    test_rows = random.permutation(key, X_test.shape[0] - test_mod)
    if test_batch_size < X_test.shape[0]:
        test_batches = np.split(test_rows,
                        X_test.shape[0] // (test_batch_size))
    else:
        test_batches = np.split(test_rows, 1)
    # large_test = True if len(test_batches) > 10 else False
    
    perform_test = ((X_test is not None) and (y_test is not None)) and early_stopping is not None
    if perform_test:
        temp_row = np.minimum(20, X_test.shape[0])
        key_ = jnp.ones((temp_row, 2))
        metric_store_master = metrics(params_, X_test[:temp_row, ...], y_test[:temp_row], key_)
        metric_store_master["loss"] = np.inf
        metric_store_master["counter"] = 0
        test_dict = metric_store_master.copy()
    start_time = time.time()
    params_store = None
    metric_store = None
    swa_params = jax.tree_map(lambda x: jnp.zeros_like(x[0]), params)
    swa_count = 0
    for epoch in range(epochs):
        rng, key = random.split(rng, 2)
        if X.shape[0] > batch_size:
            batch_dev = batch_size // devices
            mod = X.shape[0] % (batch_size)
            rows = random.permutation(key, X.shape[0] - mod)
            batches = np.split(rows, 
                X.shape[0] // (batch_size))
        else:
            mod = X.shape[0] % devices
            batch_size = X.shape[0] - mod
            batch_dev = batch_size // devices
            rows = random.permutation(key, X.shape[0] - mod)
            batches = np.split(rows, 
                1)
        pbar2 = tqdm(total=len(batches), leave=False)

        for i, b in enumerate(batches):
        # evaluate test set performance OR early stopping test
            if perform_test and step % frequency == 0:
                if swa_count > 0:
                    params_ = jax.device_get(jax.tree_map(lambda x: x / swa_count, swa_params))
                else:
                    params_ = jax.device_get(jax.tree_map(lambda x: x[0], params))
                
                if step == 0:
                    params_store = params_.copy()
                    metric_store = metric_store_master.copy()
                else:
                    # calculate list of metrics over 10 batches of the test set
                    test_dict = dict(zip(metric_store.keys(), [0.0]*len(metric_store.keys())))
                    for tbatch in test_batches[:np.minimum(10, len(test_batches))]:
                        key_ = jnp.ones((X_test[np.array(tbatch), ...].shape[0], 2))
                        temp = metrics(params_, X_test[np.array(tbatch), ...], y_test[np.array(tbatch)], key_)
                        for k in temp.keys():
                            test_dict[k] += temp[k]
                    # average metrics over test batches
                    for key, value in test_dict.items():
                        test_dict[key] = value / np.minimum(10, len(test_batches))
                # test for early stopping
                # early stopping is a function that returns a bool, params_store, metric_store
                stop, params_store, metric_store = early_stopping(
                    step, params_, params_store, test_dict, metric_store)
                if stop and early_stop:
                    tqdm.write("Final test loss: {}, epoch: {}".format(
                        metric_store["loss"], metrics_ewa_["epoch"]))
                    return params_store, history, rng
                
            step += 1
            rng, key = random.split(rng, 2)
            key = random.split(key, batch_size).reshape((devices, batch_dev, -1))
            batch_x = X[np.array(b),...]
            batch_y = y[np.array(b)]
            xbs = batch_x.shape
            batch_x = batch_x.reshape( (devices, batch_dev) + xbs[1:] )
            batch_y = batch_y.reshape((devices, batch_dev))


            # parallelized step function
            params, opt_state = take_step(
                int(step), params, batch_x, batch_y, key, opt_state)
            if swa:
                if step > optim_kwargs['m'] and step % optim_kwargs['k'] == 0:
                    swa_params = jax.tree_multimap(lambda x, y: x + y[0], swa_params, params)
                    swa_count += 1

            # initialize some training metrics
            if step <= 1:
                params_ = jax.device_get(jax.tree_map(lambda x: x[0], params))
                metrics_ewa = metrics(params_, batch_x.reshape((batch_size,)+xbs[1:]), batch_y.reshape((batch_size,-1)), None)
                metrics_ewa["step"]=step

            pbar2.update(1)
            if step % output_freq == 0:
                params_ = jax.device_get(jax.tree_map(lambda x: x[0], params))
                metrics_dict = metrics(params_, batch_x.reshape((batch_size,)+xbs[1:]), batch_y.reshape((batch_size,-1)), None)
                metrics_dict["step"]=step
                metrics_ewa = jax.tree_map(lambda x, y : 0.1 * y + 0.9 * x, metrics_ewa, metrics_dict)
                metrics_ewa["step"]=step

                metrics_ewa_ = metrics_ewa.copy()
                # evaluate loss on test set for early stopping
                metrics_ewa_["epoch"] = epoch
                if perform_test:
                    metrics_ewa_["lr"] = print_step(step)
                    # annoying hacky solution
                    metrics_ewa_["test_loss"] = metric_store["loss"] if "loss" in metric_store.keys() else np.nan
                    metrics_ewa_["test_current"] = test_dict["loss"] if "loss" in test_dict.keys() else np.nan
                    metrics_ewa_["test_counter"] = metric_store["counter"] if "counter" in metric_store.keys() else np.nan

                # enforce dtype float for dict object while saving
                for k in metrics_ewa_.keys():
                    metrics_ewa_[k] = float(metrics_ewa_[k])
                pbar1.set_postfix(metrics_ewa_)
                history.append(metrics_ewa_) 
                if np.isnan(metrics_ewa_["loss"]):
                    break
            else:
                metrics_ewa_ = metrics_ewa.copy()
        pbar2.close()
        pbar1.update(1)
        elapsed_time = time.time() - start_time
        if (elapsed_time / 60) > 11.5:
            break
        
    elapsed_time = time.time() - start_time

    if params_store is not None and early_stop:
        params = params_store
    elif swa:
        params = jax.device_get(jax.tree_map(lambda x: x / swa_count, swa_params))
    else:
        params = jax.device_get(jax.tree_map(lambda x: x[0], params))
    tqdm.write("Final test loss: {}, epoch: {}, time: {}".format(
        metric_store["loss"], epoch, timedelta(minutes=elapsed_time)))
    
    pbar1.close()
    return params, history, rng

