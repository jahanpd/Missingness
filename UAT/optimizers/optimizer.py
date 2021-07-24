from jax.experimental import optimizers
from jax.experimental.optimizers import *
from typing import Any, Callable, NamedTuple, Tuple, Union

from collections import namedtuple
import functools

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from jax._src.util import partial, safe_zip, safe_map, unzip2
from jax import tree_util
from jax.tree_util import (tree_map, tree_flatten, tree_unflatten,
                           register_pytree_node)
from UAT.aux import flatten_params, unflatten_params

@optimizer
def adabelief(step_size, b1=0.9, b2=0.999, eps=1e-8):
  """Construct optimizer triple for Adam.

  Args:
    step_size: positive scalar, or a callable representing a step size schedule
      that maps the iteration index to positive scalar.
    b1: optional, a positive scalar value for beta_1, the exponential decay rate
      for the first moment estimates (default 0.9).
    b2: optional, a positive scalar value for beta_2, the exponential decay rate
      for the second moment estimates (default 0.999).
    eps: optional, a positive scalar value for epsilon, a small constant for
      numerical stability (default 1e-8).

  Returns:
    An (init_fun, update_fun, get_params) triple.
  """
  step_size = make_schedule(step_size)
  def init(x0):
    m0 = jnp.zeros_like(x0)
    v0 = jnp.zeros_like(x0)
    return x0, m0, v0
  def update(i, g, state):
    x, m, v = state
    m = (1 - b1) * g + b1 * m  # First  moment estimate.
    v = (1 - b2) * jnp.square(g - m) + b2 * v  # Adabelief implementation
    mhat = m / (1 - jnp.asarray(b1, m.dtype) ** (i + 1))  # Bias correction.
    vhat = v / (1 - jnp.asarray(b2, m.dtype) ** (i + 1))
    x = x - step_size(i) * mhat / (jnp.sqrt(vhat) + eps)
    return x, m, v
  def get_params(state):
    x, _, _ = state
    return x
  return init, update, get_params


def SWAG(step_size, K, c, recon, **kwargs):
  """Construct optimizer that continues SGD while sampling to create a posterior.
  https://proceedings.neurips.cc/paper/2019/file/118921efba23fc329e6560b27861f0c2-Paper.pdf

  Args:
    step_size: positive scalar, or a callable representing a step size schedule
      that maps the iteration index to positive scalar.

    K: positive integer, representing low rank matrix columns

    c: positive integer, frequency of sampling

    recon: a reconstruction tuple for unflattening the params

  Returns:
    A tuple of functions with (init, update, get_param, sampling)
  """
  step_size = make_schedule(step_size)
  def init(x0):
    """ x0 must be a flattened parameter array """
    t = 1
    D = []
    x0_squared = jnp.square(x0)
    return x0, x0, x0_squared, t, D, x0
 
  def update(i, g, state, **kwargs):
    x_sum, x, x_square, t, D, x0 = state
    g_flat, _ = flatten_params(g)
    x_new = x - step_size(i) * g_flat
    t = t+1
    if i % c == 0:
      x_sum = x_sum + x_new
      x_square = x_square + jnp.square(x_new)
      x_mean = x_sum / t
      d = x_new - x_mean
      D.append(d)
      if len(D) > K:
        D = D[-K:]
    return x_sum, x_new, x_square, t, D, x0
  
  def get_params(state):
    x_sum, x, x_square, t, D, x0 = state
    return x
  
  def sample(rng, state, s, **kwargs):
    x_sum, x, x_square, t, D, x0 = state
    x_mean = x_sum / t
    diag = (x_square / t) - jnp.square(x_mean)
    D = jnp.stack(D, 1)
    d_shape = x.shape[0]
    k1, k2 = random.split(rng)
    samples =  (x_mean.reshape((1, -1, 1)) + 
            jnp.diag(diag ** 0.5).reshape((1, d_shape, d_shape)) @ 
            random.normal(k1, (s, d_shape, 1)) * 1.0/(jnp.sqrt(2)) + 
            D.reshape((1, d_shape, K)) @ random.normal(k2, (s, K, 1)) *
            (1.0 / jnp.sqrt(2.0 * (K-1.0)))
            )
    samples = [unflatten_params(samples[i, :, 0], recon)[0] for i in range(s)]
    return unflatten_params(x_mean, recon), samples

  return init, update, get_params, sample


def constantSGD(step_size, B, N, c, burn_in, recon, **kwargs):
  """Construct optimizer that continues SGD while sampling according to the principles of constant SGD
  which samples from an approximate posterior in a local minima.
  https://arxiv.org/pdf/1704.04289.pdf

  Args:
    step_size: positive scalar, an optional parameter to scale H if B/N is not small enough
    to compensate for large covariance in the gradient noise approximation

    B: positive integer, batch size

    N: positive integer, sample size of training dataset

    c: positive integer, frequency of sampling

    recon: a reconstruction tuple for unflattening the params

  Returns:
    A tuple of functions with (init, update, get_param, sampling)
  """
  step_size = make_schedule(step_size)
  def init(x0):
    """ x0 must be a flattened parameter array """
    samples = [x0]
    cov = jnp.zeros((x0.shape[0], x0.shape[0]))
    return x0, cov, samples
 
  def update(i, g, state, **kwargs):
    x, cov, samples = state
    # g1 = kwargs["g1"]
    # cov = (1 - (1/(i+1)))*cov + (1/(i+1)) * jnp.dot((g1 - g).reshape((-1,1)), (g1 - g).reshape((1,-1)))
    # H = ((2 * B) / N) * jnp.linalg.inv(cov)
    # x_new = x - jnp.dot(step_size(i)*H,g.reshape((-1, 1))).flatten()
    x_new = x - step_size(i)*g
    if (i % c == 0) and (i > burn_in):
      samples.append(x_new)
    return x_new, cov, samples
  
  def get_params(state):
    x, cov, samples = state
    return x
  
  def sample(rng, state, s, **kwargs):
    x, cov, samples = state
    x_mean = jnp.zeros_like(x)
    samp_out = []
    for samp in samples:
      x_mean = x_mean + samp
      samp_out.append(unflatten_params(samp, recon)[0])
    x_mean = x_mean / len(samples)
    return unflatten_params(x_mean, recon), samp_out

  return init, update, get_params, sample


def laplace(recon, **kwargs):
  """Construct optimizer type set of functions that returns a laplace approximation
     sampling of the posterior distribution.

  Args:
    takes no arguments

  Returns:
    A tuple of functions with (init, update, get_param, sampling)
  """
  hessian_fn = kwargs["hessian_fn"]
  def init(x0):
    return x0
 
  def update(i, g, state, **kwargs):
    return state
  
  def get_params(state):
    x0 = state
    return x0
  
  def sample(rng, state, s, **kwargs):
    x = state
    hess = hessian_fn(x, kwargs["params"], kwargs["X"], kwargs["y"], rng)
    Sigma = np.linalg.inv(hess)
    sigma = np.linalg.cholesky(Sigma)
    rng, k1 = jax.random.split(rng)
    eps = jax.random.normal(k1, (Sigma.shape[0], s))
    W_k_samples = x.reshape((-1,1)) + (sigma @ eps)
    W_k = []
    for k in range(s):
        W_sample = W_k_samples[:, k]
        unflat_param = unflatten_params(W_sample, recon)
        W_k.append(unflat_param[0])
    return unflatten_params(x, recon), W_k

  return init, update, get_params, sample

def optimizer_sampler(opt_maker: Callable[...,
  Tuple[Callable[[Params], State],
        Callable[[Step, Updates, Params], Params],
        Callable[[State], Params]]]) -> Callable[..., Optimizer]:
  """Decorator to make an optimizer defined for arrays generalize to containers.

  With this decorator, you can write init, update, and get_params functions that
  each operate only on single arrays, and convert them to corresponding
  functions that operate on pytrees of parameters. See the optimizers defined in
  optimizers.py for examples.

  Args:
    opt_maker: a function that returns an ``(init_fun, update_fun, get_params)``
      triple of functions that might only work with ndarrays, as per

      .. code-block:: haskell

          init_fun :: ndarray -> OptStatePytree ndarray
          update_fun :: OptStatePytree ndarray -> OptStatePytree ndarray
          get_params :: OptStatePytree ndarray -> ndarray

  Returns:
    An ``(init_fun, update_fun, get_params)`` triple of functions that work on
    arbitrary pytrees, as per

    .. code-block:: haskell

          init_fun :: ParameterPytree ndarray -> OptimizerState
          update_fun :: OptimizerState -> OptimizerState
          get_params :: OptimizerState -> ParameterPytree ndarray

    The OptimizerState pytree type used by the returned functions is isomorphic
    to ``ParameterPytree (OptStatePytree ndarray)``, but may store the state
    instead as e.g. a partially-flattened data structure for performance.
  """

  @functools.wraps(opt_maker)
  def tree_opt_maker(*args, **kwargs):
    init, update, get_params = opt_maker(*args, **kwargs)

    @functools.wraps(init)
    def tree_init(x0_tree):
      x0_flat, tree = tree_flatten(x0_tree)
      initial_states = [init(x0) for x0 in x0_flat]
      states_flat, subtrees = unzip2(map(tree_flatten, initial_states))
      return OptimizerState(states_flat, tree, subtrees)

    @functools.wraps(update)
    def tree_update(i, grad_tree, opt_state):
      states_flat, tree, subtrees = opt_state
      grad_flat, tree2 = tree_flatten(grad_tree)
      if tree2 != tree:
        msg = ("optimizer update function was passed a gradient tree that did "
               "not match the parameter tree structure with which it was "
               "initialized: parameter tree {} and grad tree {}.")
        raise TypeError(msg.format(tree, tree2))
      states = map(tree_unflatten, subtrees, states_flat)
      new_states = map(partial(update, i), grad_flat, states)
      new_states_flat, subtrees2 = unzip2(map(tree_flatten, new_states))
      for subtree, subtree2 in zip(subtrees, subtrees2):
        if subtree2 != subtree:
          msg = ("optimizer update function produced an output structure that "
                 "did not match its input structure: input {} and output {}.")
          raise TypeError(msg.format(subtree, subtree2))
      return OptimizerState(new_states_flat, tree, subtrees)

    @functools.wraps(get_params)
    def tree_get_params(opt_state):
      states_flat, tree, subtrees = opt_state
      states = map(tree_unflatten, subtrees, states_flat)
      params = map(get_params, states)
      return tree_unflatten(tree, params)
    
    @functools.wraps(get_params)
    def tree_get_params(opt_state):
      states_flat, tree, subtrees = opt_state
      states = map(tree_unflatten, subtrees, states_flat)
      params = map(get_params, states)
      return tree_unflatten(tree, params)

    return Optimizer(tree_init, tree_update, tree_get_params)
  return tree_opt_maker

def HSGHMC(step_size, u=1.0, **kwargs):
  """Construct optimizer that continues SGD while sampling to create a posterior.
  https://ojs.aaai.org/index.php/AAAI/article/view/17295

  Args:
    step_size: positive scalar

    u: positive real, inverse mass

    gamma: positive real, viscosity parameter

  Returns:
    A tuple of functions with (init, update, get_param, sampling)
  """
  gamma = -jnp.log(0.9) / step_size
  def init(x0):
    I = jnp.ones(x0.shape[0])
    diag_v = (u * (1.0 - jnp.exp(-2.0 * gamma * step_size))) * I
    diag_x = (u / jnp.square(gamma)) * (
      2.0 * step_size * gamma + 
      4.0 * jnp.exp(-step_size * gamma) - 
      jnp.exp(-2 * gamma * step_size) - 3
    ) * I
    v = jnp.ones_like(x0)
    g = jnp.zeros_like(x0)
    return x0, diag_x, diag_v, v, g, g
 
  def update(i, g, state, **kwargs):
    x, diag_x, diag_v, v, g_k, f = state
    rho = 1.0 / i
    g_new = (rho * g) + (rho * (g_k + g - f))
    k1, k2 = random.split(kwargs["rng"])
    eps_x = diag_x * random.normal(k1, x.shape)
    x = x - ((u / jnp.square(gamma)) * 
            (step_size * gamma + jnp.exp(-gamma * step_size) - 1) *
            g_k + (1.0/gamma) * (1 - jnp.exp(-gamma * step_size)) * v + 
            eps_x
            )
    eps_v = diag_v * random.normal(k2, v.shape)
    v = jnp.exp(-gamma * step_size) * v - (u/gamma)*(1 - jnp.exp(-gamma * step_size)) * g_k + eps_v
    return x, diag_x, diag_v, v, g_new, g
  
  def get_params(state):
    x, diag_x, diag_v, v, g_new, g, = state
    return x

  return init, update, get_params

