import numpy as np
import jax.numpy as jnp
import jax
from jax.example_libraries.optimizers import l2_norm

def binary_cross_entropy(
    l2_reg=1e-2,
    dropout_reg=1e-4
    ):
    def loss_fun(params, output, labels):
            logits = output[0]
            probs = jax.nn.sigmoid(logits)
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
                "loss":bce,
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
            embed = output[2]
            latent = output[-1]
            one_hot = jax.nn.one_hot(labels, classes)
            probs = jax.nn.softmax(logits)
            # probs = jnp.mean(jax.nn.sigmoid(jnp.stack(logits, axis=0)), axis=0)
            p_drop = jax.nn.sigmoid(params["logits"])
            @jax.vmap
            def cross_entropy(probs, labels):
                return -(labels * jnp.log(probs + 1e-7) 
                        + (1-labels) * jnp.log(1 - probs + 1e-7)).sum()

            ce = cross_entropy(probs, one_hot).mean()
            # norm_params = [params[key] for key in params.keys() if key in ["last_layer"]]
            # l2 = l2_norm(norm_params)
            l2 = jnp.mean(params["x_shift"]**2) + jnp.mean(params["y_shift"]**2)
            entropy = p_drop * jnp.log(p_drop + 1e-7)
            entropy += (1.0 - p_drop) * jnp.log(1.0 - p_drop + 1e-7)
            entropy = entropy.mean()
            loss = ce + l2_reg*l2 + dropout_reg*entropy

            loss_dict = {
                "loss":ce,
                "l2":l2,
                "ent":entropy
                }

            return loss, loss_dict
    return loss_fun

def cross_entropy_conc(
    classes,
    dropout_reg=1e-4,
    l2_reg=1e-4
    ):
    def loss_fun(params, output, labels):
            logits = output[0]
            one_hot = jax.nn.one_hot(labels, classes)
            probs = jax.nn.softmax(logits)
            # probs = jnp.mean(jax.nn.sigmoid(jnp.stack(logits, axis=0)), axis=0)
            @jax.vmap
            def cross_entropy(probs, labels):
                return -(labels * jnp.log(probs + 1e-7) 
                        + (1-labels) * jnp.log(1 - probs + 1e-7)).sum()

            ce = cross_entropy(probs, one_hot).mean()
            p_drop = jax.nn.sigmoid(jnp.concatenate([
                params["fk"]["l1_logit"].flatten(),
                params["fk"]["hidden"]["lgts"].flatten()]))

            entropy = p_drop * jnp.log(p_drop + 1e-7)
            entropy += (1.0 - p_drop) * jnp.log(1.0 - p_drop + 1e-7)
            entropy = entropy.mean()
            loss = ce + dropout_reg*entropy

            loss_dict = {
                "loss":ce,
                "ent":entropy,
                "mse":0
                }

            return loss, loss_dict
    return loss_fun

def dual(
    classes,
    l2_reg=1e-2,
    dropout_reg=1e-4,
    msereg = 1e-5
    ):
    def loss_fun(params, output, labels):
            logits = output[0]  # (b, o)
            zk_f = output[2]  # (b, f, d)
            logits_all = output[3]  # (b, f, o)
            # mask = jax.lax.stop_gradient(output[4])  # (b, f, 1)
            one_hot = jax.nn.one_hot(labels, classes)
            probs = jax.nn.softmax(logits)
            probs_all = jax.nn.softmax(logits_all)
            # probs = jnp.mean(jax.nn.sigmoid(jnp.stack(logits, axis=0)), axis=0)
            p_drop = jax.nn.sigmoid(params["logits"])
            @jax.vmap
            def cross_entropy(probs, labels):
                return -(labels * jnp.log(probs + 1e-7)
                        + (1-labels) * jnp.log(1 - probs + 1e-7)).sum(-1)

            ce = cross_entropy(probs, one_hot).mean()
            ce_all = (cross_entropy(probs_all, one_hot[:, None, :])).mean()
            entropy = p_drop * jnp.log(p_drop + 1e-4)
            entropy += (1.0 - p_drop) * jnp.log(1.0 - p_drop + 1e-4)
            entropy = entropy.mean()
            # mse = jnp.mean(jnp.square(zk_f))
            mse = l2_norm(params["outnet"])
            loss = ce + dropout_reg*entropy + msereg*mse + msereg*ce_all

            loss_dict = {
                "loss":ce,
                "mse":mse,
                "ce_all":ce_all,
                "ent":entropy,
                "pmax": jnp.max(p_drop),
                "pmin": jnp.min(p_drop),
                }

            return loss, loss_dict
    return loss_fun

def dual2(
    classes,
    dropout_reg=1e-4,
    msereg = 1e-5
    ):
    def loss_fun(params, output, labels):
            logits = output[0]  # (b, o)
            z = output[2]  # (b, d)
            one_hot = jax.nn.one_hot(labels, classes)
            probs = jax.nn.softmax(logits)
            p_drop = jax.nn.sigmoid(params["logits"])
            @jax.vmap
            def cross_entropy(probs, labels):
                return -(labels * jnp.log(probs + 1e-7)
                        + (1-labels) * jnp.log(1 - probs + 1e-7)).sum(-1)

            ce = cross_entropy(probs, one_hot).mean()
            entropy = p_drop * jnp.log(p_drop + 1e-4)
            entropy += (1.0 - p_drop) * jnp.log(1.0 - p_drop + 1e-4)
            entropy = entropy.mean()
            reg = l2_norm(params["last_layer"])
            loss = ce + dropout_reg*entropy + msereg*(z.mean())

            loss_dict = {
                "loss":ce,
                "l2":z.mean(),
                "ent":entropy,
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
                "loss":error,
                "l2":l2,
                "ent":entropy
                }

            return loss, loss_dict
    return loss_fun

def brier(
    l2_reg=1e-2,
    dropout_reg=1e-4
    ):
    def loss_fun(params, output, labels):
            logits = output[0]
            probs = jax.nn.softmax(logits)
            p_drop = jax.nn.sigmoid(params["logits"])
            @jax.vmap
            def mean_squared_error(est, true):
                return jnp.square(est - true)

            error = mean_squared_error(probs, labels).mean()
            norm_params = [params[key] for key in params.keys() if key not in ["logits"]] 
            l2 = l2_norm(norm_params)
            entropy = p_drop * jnp.log(p_drop + 1e-7)
            entropy += (1.0 - p_drop) * jnp.log(1.0 - p_drop + 1e-7)
            entropy = entropy.mean()
            loss = error + l2_reg*l2 + dropout_reg*entropy

            loss_dict = {
                "loss":error,
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


def get_metric_fun(
    problem="classification", # vs 'regression'
    classes=2, 
    metrics={} # dict of name and callable eg {"auc": auc}
    ):
    def loss_fun(params, output, labels):
            
            if problem == "classification":
                probs = jax.nn.sigmoid(output)
                one_hot = jax.nn.one_hot(labels, classes)
                @jax.vmap
                def loss_calc(probs, labels):
                    return -(labels * jnp.log(probs + 1e-7) 
                            + (1-labels) * jnp.log(1 - probs + 1e-7)).sum()
                loss_val = loss_calc(probs, one_hot).mean()
            
            elif problem == "regression":
                @jax.vmap
                def loss_calc(est, true):
                    return jnp.square(est - true)
                loss_val = loss_calc(output, labels).mean()
            
            metric_dict = {
                "loss":loss_val,
                }
            
            for metric_name, metric_call in metrics.items():
                metric_dict[metric_name] = metric_call(
                    output, labels
                )

            return loss_val, metric_dict
    return loss_fun
