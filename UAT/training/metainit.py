import optax
from optax._src import transform, combine, base
import numpy as np
import jax.numpy as jnp
import jax
from jax import random
from jax.tree_util import tree_reduce, tree_map, tree_multimap
from jax.experimental import optimizers
from jax.experimental.optimizers import make_schedule
from tqdm import tqdm
from scipy.stats import halfnorm
import functools
import time
from datetime import timedelta


def metainit_loop(
    X,
    y,
    model_fun,
    params,
    loss_fun,
    rng,
    classes,
    batch_size=32,
    epochs=100,
    lr=1e-1,
    ):
    print("Doing metainit")
    devices = jax.local_device_count()
    assert batch_size % devices == 0, "batch size must be divisible by number of devices"
    tqdm.write("{} device(s)".format(devices))
    params = jax.tree_map(lambda x: jnp.array([x] * devices), params)

    def loss(weights, x_batch, y_batch, rng, samp):
        out = model_fun(weights, x_batch, rng, samp)
        loss, _ = loss_fun(weights, out, y_batch)
        return loss

    def gradient_quotient(weights, x_batch, y_batch, rng, eps):
        def _prod(w):
            g = jax.grad(loss)(w, x_batch, y_batch, rng, True)
            temp =  tree_map(lambda x: jnp.sum(x**2 / 2), g)
            t = sum([jnp.sum(x) for x in jax.tree_leaves(temp)])
            return t, dict(grads=g)
        (_, g), prod = jax.value_and_grad(_prod, has_aux=True)(weights)
        grad = g["grads"]
        eps_ = tree_map(lambda x: eps*(2*(x >= 0)-1), grad.copy() )
        out = tree_multimap(
            lambda g, p, e: jnp.sum(jnp.abs(
                (g - p) / (g + (e))
                )),
            grad, prod, eps_
        )
        total = sum([jnp.sum(x) for x in jax.tree_leaves(out)])
        num = sum([jnp.size(x) for x in jax.tree_leaves(weights)])
        return total / num

    in_ax1 = tree_map(lambda x: 0, params)
    memory = tree_map(lambda x: jnp.array([0]*devices), params)
    memory = tree_map(lambda x: jnp.array([0]), params)

    dist = functools.partial(
        jax.pmap,
        axis_name='num_devices',
        in_axes=(None, in_ax1, 0, 0 , 0, in_ax1),
        out_axes=(in_ax1, in_ax1, None),
        # static_broadcasted_argnums=(7)
    )

    def _take_step(step, weights, x_batch, y_batch, rng, memory):
        """ Compute the gradient for a batch and update the parameters """
        gq, grads = jax.value_and_grad(gradient_quotient)(weights, x_batch, y_batch, rng, 1e-5)
        grads = jax.lax.pmean(grads, axis_name='num_devices')
        norm = tree_map(lambda x: jnp.linalg.norm(x), weights)
        sign = jax.tree_multimap(lambda g, w, n: jnp.sign(jnp.sum(g*w)/(n + 1e-8)), grads, weights, norm)
        # sign = jax.tree_multimap(lambda g: jnp.sign(g), grads)
        memory = jax.tree_multimap(lambda m, g: 0.9*m - lr*g, memory, sign)
        new_norm = jax.tree_multimap(lambda m, n: m+n, memory, norm)
        weights = jax.tree_multimap(lambda w, n, nn: w * (n / (nn + 1e-8), weights, norm, new_norm))
        return weights, memory, gq

    take_step = dist(_take_step)
    # take_step = _take_step

    history = []  # store training metrics

    pbar1 = tqdm(total=epochs, position=0, leave=False)
    step = 0

    start_time = time.time()
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

        for i in range(50):
            # evaluate test set performance OR early stopping test
            step += 1
            rng, key = random.split(rng, 2)
            key_ = random.split(key, batch_size).reshape((devices, batch_dev, -1))
            # key_ = random.split(key, batch_size).reshape((batch_dev, -1))

            rng, key = random.split(rng, 2)
            batch_x = random.normal(key, (devices, batch_dev, X.shape[-1]))
            # batch_x = random.normal(key, (batch_dev, X.shape[-1]))
            
            rng, key = random.split(rng, 2)
            batch_y = random.randint(key, (devices, batch_dev), 0, classes)
            # batch_y = random.randint(key, (batch_dev,), 0, classes)

            # parallelized step functions
            params, memory, loss_val = take_step(step, params, batch_x, batch_y, key_, memory)

            # initialize some training metrics
            pbar2.update(1)
            pbar1.set_postfix({"gq": loss_val, "step": step})
            history.append({"gq": loss_val, "step": step})
            if np.isnan(loss_val):
                break
        pbar2.close()
        pbar1.update(1)
        elapsed_time = time.time() - start_time
        if (elapsed_time / 60) > 11.5:
            break

    elapsed_time = time.time() - start_time

    params = jax.device_get(jax.tree_map(lambda x: x[0], params))
    tqdm.write("Final loss: {}, time: {}".format(
        loss_val, timedelta(minutes=elapsed_time)))

    pbar1.close()
    return params, history, rng

