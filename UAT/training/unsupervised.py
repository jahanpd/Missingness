from UAT.optimizers.optimizer import adabound, adaclipped, adabelief, swat
import optax
from optax._src import transform, combine, base
import numpy as np
import jax.numpy as jnp
import jax
from jax import random
from jax.experimental.optimizers import make_schedule
import functools
import time
from datetime import timedelta
# from UAT.aux import flatten_params, unflatten_params
from typing import Any, Callable, Optional, Union
import sys
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
  from tqdm.notebook import tqdm
else:
  from tqdm import tqdm
ScalarOrSchedule = Union[float, base.Schedule]

def _scale_by_learning_rate(learning_rate: ScalarOrSchedule, flip_sign=True):
  m = -1 if flip_sign else 1
  if callable(learning_rate):
    return transform.scale_by_schedule(lambda count: m * learning_rate(count))
  return transform.scale(m * learning_rate)

def unsupervised_loop(
    X,
    model_fun,
    params,
    rng,
    batch_size=32,
    epochs=100,
    lr=1e-3,
    optim="adabelief",
    optim_kwargs=None,
    early_stop=True, # if makes it to end of training cycle, return early stop. Also shuts down early stopping for HP optim
    ):
    devices = jax.local_device_count()
    assert batch_size % devices == 0, "batch size must be divisible by number of devices"
    tqdm.write("{} device(s)".format(devices))
    params_ = params
    params = jax.tree_map(lambda x: jnp.array([x] * devices), params)
    
    def loss(params, x_batch, rng, samp):
        embed, output = model_fun(params, x_batch, rng, samp)
        # basic mean squared error
        error = jnp.mean(jnp.square(embed - output))
        return error

    print_step = make_schedule(lr)
    step_size = lr

    if optim == "lamb":
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
    
    if optim == "sgd":
        tqdm.write("optimizer: sgb, lr: {}, batch_size={}".format(print_step(1), batch_size))
        if optim_kwargs is None:
            optim_kwargs = dict(
                learning_rate=step_size
            )
        else:
            optim_kwargs["learning_rate"] = step_size
        optobj = optax.sgd(**optim_kwargs)
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
        tqdm.write("optimizer: yogi, lr: {}, batch_size={}".format(print_step(1), batch_size))
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
                learning_rate=step_size, b1=0.9, b2=0.99, eps=1e-8
            )
        else:
            optim_kwargs["learning_rate"] = step_size
        optobj = optax.adamw(**optim_kwargs)
        optim_init = optobj.init
        optim_update = optobj.update

    elif optim == "adabelief":
        tqdm.write("optimizer: adabelief, lr: {}, batch_size={}".format(print_step(1), batch_size))
        if optim_kwargs is None:
            optim_kwargs = dict(
                b1=0.9, b2=0.99, eps=1e-8
            )
            weight_decay = 1e-5
        else:
            weight_decay = optim_kwargs.pop('weight_decay')
        optobj = combine.chain(
            transform.scale_by_belief(**optim_kwargs),
            transform.add_decayed_weights(weight_decay, None),
            _scale_by_learning_rate(step_size)
            )
        optim_init = optobj.init
        optim_update = optobj.update
    else:
        raise AssertionError("{} not implemented".format(optim))

    opt_state = optim_init(params)
    opt_state = jax.tree_map(lambda x: x if len(x.shape) > 0 else x.reshape((1)), opt_state)
    in_ax1 = jax.tree_map(lambda x: 0, params)
    in_ax2 = jax.tree_map(lambda x: 0 if len(x.shape) > 0 else None, opt_state)

    dist = functools.partial(
        jax.pmap,
        axis_name='num_devices',
        in_axes=(None, in_ax1, 0, 0, in_ax2),
        out_axes=(in_ax1, in_ax2, None),
    )

    def _take_step(step, params, x_batch, rng, opt_state):
        """ Compute the gradient for a batch and update the parameters """
        l, grads = jax.value_and_grad(loss)(params, x_batch, rng, True)
        grads = jax.lax.pmean(grads, axis_name='num_devices')
        l = jax.lax.pmean(l, axis_name='num_devices')
        updates, opt_state = optim_update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, l

    take_step = dist(_take_step)

    history = []  # store training metrics

    pbar1 = tqdm(total=epochs, position=0, leave=False)
    step = 0
    # set up test batches to avoid OOM errors on test set computation if test set is large
    start_time = time.time()
    store_loss = np.inf
    early_stop = 0
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
            step += 1
            rng, key = random.split(rng, 2)
            key = random.split(key, batch_size).reshape((devices, batch_dev, -1))
            batch_x = X[np.array(b),...]
            xbs = batch_x.shape
            batch_x = batch_x.reshape( (devices, batch_dev) + xbs[1:] )

            # parallelized step functions
            params, opt_state, loss_ = take_step(step, params, batch_x, key, opt_state)
            
            pbar2.update(1)
            pbar1.set_postfix({"loss": loss_})
            history.append({"loss": loss_})
            if loss_ < store_loss:
                store_loss = loss_
                early_stop = 0
            else:
                early_stop += 1
            
        pbar2.close()
        pbar1.update(1)
        elapsed_time = time.time() - start_time
        if (elapsed_time / 60) > 11.5:
            break
        if early_stop > 100:
            break

    elapsed_time = time.time() - start_time
    params = jax.device_get(jax.tree_map(lambda x: x[0], params))
    
    pbar1.close()
    return params, history, rng

