import optax
from optax._src import transform, combine, base
import numpy as np
import jax.numpy as jnp
import jax
from jax import random, jit, jacfwd, jacrev
from jax.tree_util import tree_reduce, tree_map, tree_multimap
from jax.experimental import optimizers
from jax.experimental.optimizers import make_schedule
from jax.flatten_util import ravel_pytree
from tqdm import tqdm
from scipy.stats import halfnorm
import functools
import time
from datetime import timedelta

# forward-over-reverse
def hvp(f, primals, tangents):
  return jax.jvp(grad(f), primals, tangents)[1]

# l1 norm for adam, l2 for sgd
def global_norm(updates):
  """Compute the global norm across a nested structure of tensors."""
  return (
      sum([jnp.sum(jnp.abs(x)) for x in jax.tree_leaves(updates)]))

def global_norm2(updates):
  """Compute the global norm across a nested structure of tensors."""
  return (
      sum([jnp.sum(jnp.square(x))/2 for x in jax.tree_leaves(updates)]))

def gradinit_loop(
    X,
    y,
    model_fun,
    params,
    loss_fun,
    rng,
    batch_size=32,
    steps=1000,
    lr=1e-1,
    version='metainit'  # one of metainit or gradinit
    ):
    print("Doing {} for weight initialization".format(version))
    devices = jax.local_device_count()
    assert batch_size % devices == 0, "batch size must be divisible by number of devices"
    tqdm.write("{} device(s)".format(devices))
    # params = jax.tree_map(lambda x: jnp.array([x] * devices), params)
    shapes = jax.tree_map(lambda x: x.shape, params)
    m = jax.tree_multimap(
        lambda x, s: jnp.ones(1) if len(s) < 3 and s[0] != X.shape[1] else jnp.ones((s[0],) + (1,)*(len(s) - 1) ),
        params, shapes)
    #print(jax.tree_map(lambda x: x.shape, m))
    #print(shapes)
    # some global variables for gradinit
    eta = 5e-4
    gamma = 1e-3
    rho = 1e-3
    alpha = 0.01

    optim_kwargs = dict(
        # learning_rate=rho, b1=0.9, b2=0.99, eps=1e-8
        learning_rate=rho, momentum=0.9
    )
    optobj = optax.sgd(**optim_kwargs)
    optim_init = optobj.init
    optim_update = optobj.update
    if version == 'gradinit':
        opt_state = optim_init(m)
        shape = X.shape[0]
    elif version == 'metainit':
        opt_state = tree_map(lambda x: jnp.array(0.0), params)
        shape = 10 * batch_size
        X = X[:shape, :]
        y = y[:shape]
    steps_per_epoch = shape // batch_size
    epochs = steps // steps_per_epoch

    def loss(params, xb, yb, key):
        out = model_fun(params, xb, key, True)
        loss, _ = loss_fun(params, out, yb)
        return loss

    @jit
    def norm_grad(m, params, xb, yb, key):
        params = jax.tree_multimap(lambda x, y: x*y, m, params)
        _, g1 = jit(jax.value_and_grad(loss))(
            params, xb, yb, key
        )
        return global_norm(g1), g1

    @jit
    def onestep_grad(m, params, g, xb, yb, key):
        params = jax.tree_multimap(lambda x, y: x*y, m, params)
        p_t2 = jax.tree_multimap(
            lambda p, g: p - eta * jnp.sign(g),
            params,
            g
        )
        return loss(p_t2, xb, yb, key)

    def norm_grad2(params, xb, yb, key):
        _, g1 = jax.value_and_grad(loss)(
            params, xb, yb, key
        )
        return global_norm2(g1), g1

    def gradquot(params, xb, yb, rng, eps=1e-5):
        (val, prod), grads = norm_grad2_(
            params, xb, yb, rng
        )
        out = sum([
            jnp.sum(jnp.abs((g - p) / (g + eps * (2 * (e >= 0) - 1)) - 1))
            for g, p, e in zip(
                jax.tree_leaves(grads),
                jax.tree_leaves(prod),
                jax.tree_leaves(jax.lax.stop_gradient(grads))
            )]
        )
        return out

    loss_ = jax.grad(loss)
    norm_grad_ = jax.value_and_grad(norm_grad, has_aux=True)
    norm_grad2_ = jax.value_and_grad(norm_grad2, has_aux=True)
    onestep_grad_ = jax.grad(onestep_grad)
    gradquot_ = jax.value_and_grad(gradquot)

    @jit
    def step_metainit(params, xb, yb, rng, opt_state):
        key = random.split(rng, batch_size)
        gq, grads = gradquot_(params, xb, yb, key)
        norms = tree_map(lambda x: jnp.sqrt(jnp.square(x).sum()), params)
        signs = tree_multimap(
            lambda p, g, n: jnp.sign(jnp.sum(p * g) / (n + 1e-8)),
            params,
            grads,
            norms
        )
        opt_state = tree_multimap(
            lambda m, g: 0.9 * m - lr * g,
            opt_state,
            signs
        )
        params = tree_multimap(
            lambda p, m, n: p * jnp.where(
                jnp.isnan((n + m) / (n + 1e-5)),
                jnp.ones_like(p),
                (n + m) / (n + 1e-5)
                ),
                params,
            opt_state,
            norms
        )
        return params, gq, opt_state

    def step_gradinit(m, params, xb1, yb1, xb2, yb2, rng, opt_state):
        key = random.split(rng, batch_size)
        # g1 is norm wrt m and g2 is loss wrt p
        (val, g2), g1 = norm_grad_(
            m, params, xb1, yb1, key
        )
        if val > gamma:
            updates, opt_state = optim_update(g1, opt_state, m)
            m_new = optax.apply_updates(m, updates)
        else:
            g3 = onestep_grad_(
                m, params, g2, xb2, yb2, key
            )
            updates, opt_state = optim_update(g3, opt_state, m)
            m_new = optax.apply_updates(m, updates)

        # clip m
        m_new = jax.tree_map(lambda x: jnp.clip(x, alpha), m_new)
        return m_new, val, opt_state

    steps = 0
    pbar1 = tqdm(total=epochs, position=0, leave=False)
    history = []
    val_store, val = np.inf, np.inf
    val_unchanged = 0
    params_store = params.copy()
    for epoch in range(epochs):
        rng, key = random.split(rng, 2)
        if X.shape[0] > batch_size:
            mod = X.shape[0] % (batch_size)
            rows = random.permutation(key, X.shape[0] - mod)
            batches = np.split(rows,
                X.shape[0] // (batch_size))
        else:
            mod = X.shape[0] % devices
            batch_size = X.shape[0] - mod
            rows = random.permutation(key, X.shape[0] - mod)
            batches = np.split(rows,
                1)
        pbar2 = tqdm(total=len(batches)-1, leave=False)
        for (b1, b2) in zip(range(0, len(batches)-1), range(1, len(batches))):
            xb1 = X[batches[b1],:]
            yb1 = y[batches[b1]]

            xb2 = X[batches[b2], :]
            yb2 = y[batches[b2]]

            rng, key = random.split(rng, 2)
            xb = random.normal(key, xb1.shape)
            rng, key = random.split(rng, 2)
            yb = random.randint(key, yb1.shape, minval=0, maxval=2)

            # mix batches for stability of stochasticity
            xb2[batch_size // 2 :, :] = xb1[batch_size // 2 :, :]
            yb2[batch_size // 2 :] = yb1[batch_size // 2 :]

            rng, key = random.split(rng, 2)
            val_ = val
            if version == 'gradinit':
                m, val, opt_state = step_gradinit(m, params, xb1, yb1, xb2, yb2, rng, opt_state)
            elif version == 'metainit':
                params, val, opt_state = step_metainit(params, xb, yb, rng, opt_state)

            store = True if val < val_store else False
            val_store = val if store else val_store
            params_store = params.copy() if store else params_store
            if val == val_:
                val_unchanged += 1
            else:
                val_unchanged = 0

            steps += 1
            record = dict(val=val, val_store=val_store, epoch=epoch, count=val_unchanged)
            pbar1.set_postfix(record)
            history.append(record)
            pbar2.update(1)
        pbar1.update(1)
        if val_unchanged > steps_per_epoch*5:
            break

    print("final value is {}".format(val_store))
    params = jax.tree_multimap(lambda x, y: x*y, m, params_store)
    return params, history, rng
