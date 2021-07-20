import numpy as np
import jax.numpy as jnp
import jax
from jax import random
from UAT.aux import unflatten_params, flatten_params

# define a function that takes params, loss_fun, X, y, samples and returns last layer uncertainty for weights

def laplace_approximation(
    params,
    apply_fun,
    loss_fun,
    X,
    y,
    rng,
    samples,
    **kwargs
    ):

    last_layer_flat, recon = flatten_params(params["last_layer"])
    def loss(last_layer, params, x_batch, y_batch):
            params["last_layer"] = unflatten_params(last_layer, recon)
            out = apply_fun(params, x_batch, None)
            loss, _ = loss_fun(params, out, y_batch)
            return loss
    
    hessian_loss = jax.jit(jax.hessian(loss))

    hess = hessian_loss(last_layer_flat, params, X, y)
    Sigma = np.linalg.inv(hess)
    sigma = np.linalg.cholesky(Sigma)
    rng, k1 = jax.random.split(rng)
    eps = jax.random.normal(k1, (Sigma.shape[0], samples))
    W_k_samples = last_layer_flat.reshape((-1,1)) + (sigma @ eps)
    W_k = []
    for k in range(samples):
        W_sample = W_k_samples[:, k]
        unflat_param = unflatten_params(W_sample, recon)
        W_k.append(unflat_param[0])
    
    params["last_layer"] = W_k
    params["last_layer_map"] = unflatten_params(last_layer_flat, recon)

    return params, rng