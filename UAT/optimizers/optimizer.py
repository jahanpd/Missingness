from jax.experimental import optimizers
from jax.experimental.optimizers import *
from jax.experimental.optimizers import l2_norm
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
from scipy.stats import halfnorm

FFn = Callable[[OptimizerState], Params]

class Optimizer2(NamedTuple):
  init_fn: InitFn
  update_fn: UpdateFn
  params_fn: ParamsFn
  f_fn: FFn

def optimizer2(opt_maker: Callable[...,
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
    init, update, get_params, get_other = opt_maker(*args, **kwargs)

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
    
    @functools.wraps(get_other)
    def tree_get_other(opt_state):
      states_flat, tree, subtrees = opt_state
      states = map(tree_unflatten, subtrees, states_flat)
      out = map(get_other, states)
      return tuple(tree_unflatten(tree, i) for i in zip(*out))

    return Optimizer2(tree_init, tree_update, tree_get_params, tree_get_other)
  return tree_opt_maker

@optimizer2
def adascore(step_size, p=0.2, b1=0.9, b2=0.9, b3=0.999, eps=1e-8, anneal=0.9998):
  # https://statswithr.github.io/book/hypothesis-testing-with-normal-populations.html
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
  thresh = halfnorm.ppf(p)
  if type(anneal) == float:
    schedule = lambda x: 100 * anneal**(x) + 1
  else:
    schedule = lambda x : 1.0
  def init(x0):
    m0 = jnp.zeros_like(x0)
    v0 = jnp.zeros_like(x0)
    z0 = jnp.ones_like(x0) * thresh
    return x0, m0, v0, z0
  def update(i, g, state):
    x, m, v, z  = state
    # include EWVar into calculation
    delta = g - m
    m = m + (1 - b1)*delta # first moment
    v = b2 * (v + (1-b2) * jnp.square(delta)) # second moment
    vhat = v / (1 - jnp.asarray(b1, m.dtype) ** (i + 1))
    mhat = m / (1 - jnp.asarray(b2, m.dtype) ** (i + 1))  # Bias correction.
    inv_std = 1 / (jnp.sqrt(vhat) + eps)
    z_new = jnp.abs(m) * inv_std
    z = (1 - b3) * z_new + b3*z
    zhat = z / (1 - jnp.asarray(b2, m.dtype) ** (i + 1))  # Bias correction.
    I = jnp.where(
      zhat >= thresh/schedule(i),
      step_size(i) * mhat * inv_std,
      0.0)
    x = x - I
    return x, m, v, z
  def get_params(state):
    x, _, _, _ = state
    return x
  def get_other(state):
    x, m, v, z = state
    return m, v, z, thresh
  return init, update, get_params, get_other

@optimizer
def adabelief(step_size, b1=0.9, b2=0.999, eps=1e-8):
  """Construct optimizer triple for Adabelief.

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
    v = (1 - b2) * jnp.square(g-m) + b2 * v  # Second moment estimate.
    mhat = m / (1 - jnp.asarray(b1, m.dtype) ** (i + 1))  # Bias correction.
    vhat = v / (1 - jnp.asarray(b2, m.dtype) ** (i + 1))
    x = x - step_size(i) * mhat / (jnp.sqrt(vhat) + eps)
    return x, m, v
  def get_params(state):
    x, _, _ = state
    return x
  return init, update, get_params