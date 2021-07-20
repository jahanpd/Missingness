import numpy as np
import jax.numpy as jnp
import jax
from jax import random
from jax.nn import (relu, log_softmax, softmax, softplus, sigmoid, elu,
                    leaky_relu, selu, gelu, normalize)
from jax.nn.initializers import glorot_normal, normal, ones, zeros
from jax.experimental.optimizers import l2_norm
from UAT.models.models import AttentionModel
from UAT.training.train import training_loop
from UAT.uncertainty.laplace import laplace_approximation

class UAT:
    def __init__(
        self,
        model_kwargs,
        training_kwargs,
        loss_fun,
        rng_key=42,
        feat_names=None,
        posterior_params=None,
        ):
        
        init_fun, apply_fun = AttentionModel(
            **model_kwargs
            )
        key = jax.random.PRNGKey(rng_key)
        key, init_key = random.split(key)
        self.key = key
        self.params = init_fun(init_key)
        self.apply_fun = apply_fun
        if (feat_names is not None) and len(feat_names) == model_kwargs["features"]:
            self.feat_names = feat_names
        else:
            self.feat_names = list(range(model_kwargs["features"]))
        self.loss_fun = loss_fun
        self.posterior_params = posterior_params
        self.training_kwargs = training_kwargs

    def fit(self, X, y):

        params, history, rng = training_loop(
            X=X,
            y=y,
            model_fun=self.apply_fun,
            params=self.params,
            loss_fun=self.loss_fun,
            metric_fun=self.loss_fun,
            rng=self.key,
            **self.training_kwargs
            )
        self.params = params
        self.history = history
        self.key = rng

        if self.posterior_params is not None:
            if self.posterior_params["name"] == "laplace":
                params, rng = laplace_approximation(
                    self.params,
                    self.apply_fun,
                    self.loss_fun,
                    X,
                    y,
                    self.key,
                    **self.posterior_params
                )
                self.params = params
                self.key = rng
    
    def predict_proba(self, X):
        out = self.apply_fun(self.params, X)
        logits, attn = out
        probs = jnp.mean(jnp.stack(jax.nn.sigmoid(logits), axis=0), axis=0)
        return probs
    
    def predict(self, X):
        out = self.apply_fun(self.params, X)
        logits, attn = out
        probs = jnp.mean(jnp.stack(jax.nn.sigmoid(logits), axis=0), axis=0)
        return probs, attn


