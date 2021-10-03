from UAT.optimizers.optimizer import adabelief, adascore
import numpy as np
import jax.numpy as jnp
import jax
from jax import random
from jax.experimental import optimizers
from jax.experimental.optimizers import make_schedule
from tqdm import tqdm
from scipy.stats import halfnorm
import functools
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
    ):
    devices = jax.local_device_count()
    assert batch_size % devices == 0, "batch size must be divisible by number of devices"
    tqdm.write("{} device(s)".format(devices))
    params_flat, params_build = jax.tree_util.tree_flatten(params)
    params_ = params
    params = jax.tree_map(lambda x: jnp.array([x] * devices), params)
    in_ax1 = jax.tree_map(lambda x: 0, params)

    
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
    
    elif optim == "adascore":
        if optim_kwargs is None:
            optim_kwargs = dict(
                step_size=step_size, batch_size=batch_size, N = X.shape[0], start_score=start_score, b1=0.9, b2=0.999, eps=1e-8
            )
        else:
            optim_kwargs["step_size"] = step_size
            optim_kwargs["start_score"] = start_score
            optim_kwargs["batch_size"] = batch_size
            optim_kwargs["N"] = X.shape[0]

        optim_init, optim_update, optim_params, get_other = adascore(**optim_kwargs)
        tqdm.write("optimizer: adascore, lr: {}, start score: {}, batch_size={}".format(print_step(1), start_score, batch_size))
        jit_other = jax.jit(get_other)

        # this function is for determining proportion of weights 'on'
        def weights_on_fn(opt_state, step):
            m, v, vn, p = jit_other(opt_state)
            # p, _ = flatten_params(p)
            # v, _ = flatten_params(v)
            # vn, _ = flatten_params(vn)
            # get ratio of dzdx > dzdy

            out_dict = {
                # "avg_vn":np.mean(vn),
                # "avg_v":np.mean(v),
                # "avg_p":np.mean(p),
            }
            return out_dict
    
    elif optim == "adam":
        tqdm.write("optimizer: adam, lr: {}, batch_size={}".format(print_step(1), batch_size))
        if optim_kwargs is None:
            optim_kwargs = dict(
                step_size=step_size, b1=0.9, b2=0.99, eps=1e-8
            )
        else:
            optim_kwargs["step_size"] = step_size

        optim_init, optim_update, optim_params = optimizers.adam(**optim_kwargs)
    
    elif optim == "adabelief":
        tqdm.write("optimizer: adabelief, lr: {}, batch_size={}".format(print_step(1), batch_size))
        if optim_kwargs is None:
            optim_kwargs = dict(
                step_size=step_size, b1=0.9, b2=0.99, eps=1e-8
            )
        else:
            optim_kwargs["step_size"] = step_size
        optim_init, optim_update, optim_params = adabelief(**optim_kwargs)

    opt_state = optim_init(params)
    in_ax2 = jax.tree_map(lambda x: 0, opt_state)
    
    if optim == "adascore":
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
            g0 = jax.grad(loss)(params, x_batch[:1,...], y_batch[:1], rng)
            g0 = jax.lax.pmean(g0, axis_name='num_devices')
            keys = random.split(rng[0,...], len(params_flat))
            uniform = [random.uniform(key, x.shape) for key, x in zip(keys, params_flat)]
            uniform = jax.tree_util.tree_unflatten(params_build, uniform)
            opt_state = optim_update(step, grads, g0, uniform, opt_state)
            return optim_params(opt_state), opt_state
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

    if optim == "adascore":
        grad_dict = weights_on_fn(opt_state, step+1)
    else:
        grad_dict = {}
    
    # set up test batches to avoid OOM errors on test set computation if test set is large
    rng, key = random.split(rng, 2)
    test_mod = X_test.shape[0] % (batch_size)
    test_rows = random.permutation(key, X_test.shape[0] - test_mod)
    test_batches = np.split(test_rows,
                X_test.shape[0] // (batch_size))
    # large_test = True if len(test_batches) > 10 else False

    perform_test = ((X_test is not None) and (y_test is not None)) and early_stopping is not None
    if perform_test:
        temp_row = np.minimum(20, X_test.shape[0])
        key_ = jnp.ones((temp_row, 2))
        metric_store_master = metrics(params_, X_test[:temp_row, ...], y_test[:temp_row], key_)
        metric_store_master["loss"] = np.inf
        metric_store_master["counter"] = 0
        test_dict = metric_store_master.copy()
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
                params_ = jax.device_get(jax.tree_map(lambda x: x[0], params))
                if step == 0:
                    params_store = params_.copy()
                    metric_store = metric_store_master.copy()
                else:
                    # calculate list of metrics over 10 batches of the test set
                    test_dict = dict(zip(metric_store.keys(), [0.0]*len(metric_store.keys())))
                    for tbatch in test_batches[:np.minimum(10, len(test_batches) - 1)]:
                        key_ = jnp.ones((X_test[np.array(tbatch), ...].shape[0], 2))
                        temp = metrics(params_, X_test[np.array(tbatch), ...], y_test[np.array(tbatch)], key_)
                        for k in temp.keys():
                            test_dict[k] += temp[k]
                    # average metrics over test batches
                    for key, value in test_dict.items():
                        test_dict[key] = value / np.minimum(10, len(test_batches))
                for k, v in grad_dict.items():
                    test_dict[k] = v
                # test for early stopping
                # early stopping is a function that returns a bool, params_store, metric_store
                stop, params_store, metric_store = early_stopping(
                    step, params_, params_store, test_dict, metric_store)
                if stop:
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
               step, params, batch_x, batch_y, key, opt_state)

            if step % output_freq == 0 and optim == "adascore":
                grad_dict = weights_on_fn(opt_state, step)

            # initialize some training metrics
            if step == 1:
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
                if optim == "adascore":
                    metrics_ewa_["avg_vn"] = grad_dict["avg_vn"]
                    metrics_ewa_["avg_v"] = grad_dict["avg_v"]
                    metrics_ewa_["avg_p"] = grad_dict["avg_p"]

                # enforce dtype float for dict object while saving
                for k in metrics_ewa_.keys():
                    metrics_ewa_[k] = float(metrics_ewa_[k])
                if np.isnan(metrics_ewa_["loss"]):
                    break
                pbar1.set_postfix(metrics_ewa_)
                history.append(metrics_ewa_)
        if np.isnan(metrics_ewa_["loss"]):
            break
        pbar2.close()
        pbar1.update(1)
    try:
        tqdm.write("Final test loss: {}, epoch: {}".format(
            metrics_ewa_["test_loss"], metrics_ewa_["epoch"]))
    except:
        tqdm.write("Final loss: {}, epoch: {}".format(
            metrics_ewa_["loss"], metrics_ewa_["epoch"]))
    pbar1.close()
    params = jax.device_get(jax.tree_map(lambda x: x[0], params))
    return params, history, rng

