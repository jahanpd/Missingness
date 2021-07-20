import jax
import numpy as np
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten

def flatten_params(params):
    value_flat, value_tree = tree_flatten(params)
    idx = jnp.cumsum(jnp.array([0] + [x.size for x in value_flat]))
    idx_tup = []
    for i,index in enumerate(idx[:-1]):
        idx_tup.append((idx[i], idx[i+1]))
    shapes = [x.shape for x in value_flat]
    recon = (value_tree, shapes, idx_tup)
    flattened = jnp.concatenate([x.ravel() for x in value_flat])
    return flattened, recon

def unflatten_params(flattened_params, recon_tuple):
    value_tree, shapes, idx_tup = recon_tuple
    params_less_flat = []
    for i,idx in enumerate(idx_tup):
        params_less_flat.append(
            flattened_params[idx[0]:idx[1]].reshape(shapes[i])
        )
    params = tree_unflatten(value_tree, params_less_flat)
    return params