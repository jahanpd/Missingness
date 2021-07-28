import numpy as np
import jax.numpy as jnp
import jax
from jax.experimental.optimizers import l2_norm

def binary_cross_entropy(
    l2_reg=1e-2,
    dropout_reg=1e-4
    ):
    def loss_fun(params, output, labels):
            logits = output[0]
            probs = jnp.mean(jax.nn.sigmoid(jnp.stack(logits, axis=0)), axis=0)
            p_drop = jax.nn.sigmoid(params["logits"])
            @jax.vmap
            def binary_cross_entropy(probs, labels):
                return -(labels * jnp.log(probs + 1e-7) 
                        + (1-labels) * jnp.log(1 - probs + 1e-7))

            bce = binary_cross_entropy(probs, labels).mean()
            norm_params = [params[key] for key in params.keys() if key not in ["logits"]] 
            l2 = l2_norm(norm_params)
            entropy = p_drop * jnp.log(p_drop + 1e-7)
            entropy += (1.0 - p_drop) * jnp.log(1.0 - p_drop + 1e-7)
            entropy = entropy.mean()
            loss = bce + l2_reg*l2 + dropout_reg*entropy

            loss_dict = {
                "bce":bce,
                "l2":l2,
                "ent":entropy
                }

            return loss, loss_dict
    return loss_fun

def cross_entropy(
    classes,
    l2_reg=1e-2,
    dropout_reg=1e-4
    ):
    def loss_fun(params, output, labels):
            logits = output[0]
            one_hot = jax.nn.one_hot(labels, classes)
            probs = jnp.mean(jax.nn.sigmoid(jnp.stack(logits, axis=0)), axis=0)
            p_drop = jax.nn.sigmoid(params["logits"])
            @jax.vmap
            def cross_entropy(probs, labels):
                return -(labels * jnp.log(probs + 1e-7) 
                        + (1-labels) * jnp.log(1 - probs + 1e-7)).sum()

            ce = cross_entropy(probs, one_hot).mean()
            norm_params = [params[key] for key in params.keys() if key not in ["logits"]] 
            l2 = l2_norm(norm_params)
            entropy = p_drop * jnp.log(p_drop + 1e-7)
            entropy += (1.0 - p_drop) * jnp.log(1.0 - p_drop + 1e-7)
            entropy = entropy.mean()
            loss = ce + l2_reg*l2 + dropout_reg*entropy

            loss_dict = {
                "bce":ce,
                "l2":l2,
                "ent":entropy
                }

            return loss, loss_dict
    return loss_fun

def cross_entropy_(
    classes,
    l2_reg=1e-2
    ):
    def loss_fun(params, output, labels):
            logits = output
            one_hot = jax.nn.one_hot(labels, classes)
            probs = jax.nn.sigmoid(logits)
            @jax.vmap
            def cross_entropy(probs, labels):
                return -(labels * jnp.log(probs + 1e-7) 
                        + (1-labels) * jnp.log(1 - probs + 1e-7)).sum()

            ce = cross_entropy(probs, one_hot).mean()
            l2 = l2_norm(params)
            loss = ce + l2_reg*l2

            loss_dict = {
                "loss":ce,
                "l2":l2,
                }

            return loss, loss_dict
    return loss_fun

def mse(
    l2_reg=1e-2,
    dropout_reg=1e-4
    ):
    def loss_fun(params, output, labels):
            est = output[0]
            est = jnp.mean(jnp.stack(est, axis=0), axis=0)
            p_drop = jax.nn.sigmoid(params["logits"])
            @jax.vmap
            def mean_squared_error(est, true):
                return jnp.square(est - true)

            error = mean_squared_error(est, labels).mean()
            norm_params = [params[key] for key in params.keys() if key not in ["logits"]] 
            l2 = l2_norm(norm_params)
            entropy = p_drop * jnp.log(p_drop + 1e-7)
            entropy += (1.0 - p_drop) * jnp.log(1.0 - p_drop + 1e-7)
            entropy = entropy.mean()
            loss = error + l2_reg*l2 + dropout_reg*entropy

            loss_dict = {
                "mse":error,
                "l2":l2,
                "ent":entropy
                }

            return loss, loss_dict
    return loss_fun

def mse_(
    l2_reg=1e-2,
    ):
    def loss_fun(params, output, labels):
            est = output

            @jax.vmap
            def mean_squared_error(est, true):
                return jnp.square(est - true)

            error = mean_squared_error(est, labels).mean()
            l2 = l2_norm(params)
            loss = error + l2_reg*l2

            loss_dict = {
                "loss":error,
                "l2":l2,
                }

            return loss, loss_dict
    return loss_fun