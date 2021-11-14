from collections import namedtuple
import functools
import operator

from six.moves import reduce

import jax.numpy as np
from jax._src.util import partial, safe_zip, safe_map, unzip2
from jax import tree_util
from jax.tree_util import (tree_map, tree_flatten, tree_unflatten,
                           register_pytree_node)

map = safe_map
zip = safe_zip

# code base on:
# https://github.com/touqir14/Adabound-Jax/blob/master/adabound.py


# The implementation here basically works by flattening pytrees. There are two
# levels of pytrees to think about: the pytree of params, which we can think of
# as defining an "outer pytree", and a pytree produced by applying init_fun to
# each leaf of the params pytree, which we can think of as the "inner pytrees".
# Since pytrees can be flattened, that structure is isomorphic to a list of
# lists (with no further nesting).

pack = tuple
OptimizerState = namedtuple("OptimizerState",
                            ["packed_state", "tree_def", "subtree_defs"])
register_pytree_node(
    OptimizerState,
    lambda xs: ((xs.packed_state,), (xs.tree_def, xs.subtree_defs)),
    lambda data, xs: OptimizerState(xs[0], data[0], data[1]))


def optimizer(opt_maker):
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
      packed_state = pack(map(pack, states_flat))
      return OptimizerState(packed_state, tree, subtrees)

    @functools.wraps(update)
    def tree_update(i, grad_tree, opt_state):
      packed_state, tree, subtrees = opt_state
      grad_flat, tree2 = tree_flatten(grad_tree)
      if tree2 != tree:
        msg = ("optimizer update function was passed a gradient tree that did "
               "not match the parameter tree structure with which it was "
               "initialized: parameter tree {} and grad tree {}.")
        raise TypeError(msg.format(tree, tree2))
      states = map(tree_unflatten, subtrees, packed_state)
      new_states = map(partial(update, i), grad_flat, states)
      new_states_flat, subtrees2 = unzip2(map(tree_flatten, new_states))
      for subtree, subtree2 in zip(subtrees, subtrees2):
        if subtree2 != subtree:
          msg = ("optimizer update function produced an output structure that "
                 "did not match its input structure: input {} and output {}.")
          raise TypeError(msg.format(subtree, subtree2))
      new_packed_state = pack(map(pack, new_states_flat))
      return OptimizerState(new_packed_state, tree, subtrees)

    @functools.wraps(get_params)
    def tree_get_params(opt_state):
      packed_state, tree, subtrees = opt_state
      states = map(tree_unflatten, subtrees, packed_state)
      params = map(get_params, states)
      return tree_unflatten(tree, params)

    return tree_init, tree_update, tree_get_params
  return tree_opt_maker


def constant(step_size):
  def schedule(i):
    return step_size
  return schedule

def make_schedule(scalar_or_schedule):
  if callable(scalar_or_schedule):
    return scalar_or_schedule
  elif np.ndim(scalar_or_schedule) == 0:
    return constant(scalar_or_schedule)
  else:
    raise TypeError(type(scalar_or_schedule))

# with stochastic weight averaging
@optimizer
def adabound(
  step_size, b1=0.9, b2=0.99, gamma=0.9999,
  eps=1e-8, weight_decay=1e-8,
  m=None, k=None, offset=1000
  ):
  """Construct optimizer triple for AdaBound.
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
  print(
    step_size(m) * (1 - 1 / (gamma*np.maximum(m-offset, eps) + 1)),
    step_size(m) * (1 + 1 / (gamma*np.maximum(m-offset, eps)))
  )

  def init(x0):
    m0 = np.zeros_like(x0)
    v0 = np.zeros_like(x0)
    return x0, m0, v0,
  
  def update(i, g, state):
    x, m, v = state
    m = (1 - b1) * g + b1 * m  # First  moment estimate.
    v = (1 - b2) * (g ** 2) + b2 * v  # Second moment estimate.
    alpha = step_size(i)

    mhat = m / (1 - b1 ** (i + 1))  # Bias correction.
    vhat = v / (1 - b2 ** (i + 1))
    lower_bound = alpha * (1 - 1 / (gamma*np.maximum(i-offset, eps) + 1))
    upper_bound = alpha * (1 + 1 / (gamma*np.maximum(i-offset, eps)))
    clipped = np.clip(alpha / (np.sqrt(vhat) + eps), lower_bound, upper_bound)
    
    # x = x - mhat * clipped
    x = ((1-weight_decay)*x) - mhat * clipped
    return x, m, v
  
  def get_params(state):
    x, m, v = state
    return x
  
  return init, update, get_params

@optimizer
def adaclipped(
  step_size, lower_bound,
  b1=0.9, b2=0.99, gamma=0.9999,
  eps=1e-8, weight_decay=1e-8,
  m=None, k=None
  ):
  """Construct optimizer triple for AdaBound.
  Args:
    step_size: positive scalar, or a callable representing a step size schedule
      that maps the iteration index to positive scalar.
    lower_bound: positive scalar, or a callable representing a step size schedule
      that maps the iteration index to positive scalar.
    b1: optional, a positive scalar value for beta_1, the exponential decay rate
      for the first moment estimates (default 0.9).
    b2: optional, a positive scalar value for beta_2, the exponential decay rate
      for the second moment estimates (default 0.999).
    eps: optional, a positive scalar value for epsilon, a small constant for
      numerical stability (default 1e-8).
    weight_decay: positive float
    m: positive integer, used for determining step when to start stochastic
      weight averaging
    k: positive integer, used to set frequency of swa

  Returns:
    An (init_fun, update_fun, get_params) triple.
  """

  step_size = make_schedule(step_size)
  lower_bound = make_schedule(lower_bound)

  def init(x0):
    m0 = np.zeros_like(x0)
    v0 = np.zeros_like(x0)
    return x0, m0, v0,
  
  def update(i, g, state):
    x, m, v = state
    m = (1 - b1) * g + b1 * m  # First  moment estimate.
    v = (1 - b2) * ((g-m) ** 2) + b2 * v  # Second moment estimate.
    alpha = step_size(i)
    lb = lower_bound(i)

    mhat = m / (1 - b1 ** (i + 1))  # Bias correction.
    vhat = v / (1 - b2 ** (i + 1))
    clipped = np.clip(alpha / (np.sqrt(vhat) + eps), lb, None)
    
    # x = x - mhat * clipped
    x = ((1-weight_decay)*x) - mhat * clipped
    return x, m, v
  
  def get_params(state):
    x, m, v = state
    return x
  
  return init, update, get_params

@optimizer
def swtich(
  step_size, step_size2,
  b1=0.9, b2=0.99, offset=1000,
  eps=1e-8, weight_decay=1e-8,
  m=None, k=None
  ):
  """Construct optimizer triple for AdaBound.
  Args:
    step_size: positive scalar, or a callable representing a step size schedule
      that maps the iteration index to positive scalar.
    lower_bound: positive scalar, or a callable representing a step size schedule
      that maps the iteration index to positive scalar.
    b1: optional, a positive scalar value for beta_1, the exponential decay rate
      for the first moment estimates (default 0.9).
    b2: optional, a positive scalar value for beta_2, the exponential decay rate
      for the second moment estimates (default 0.999).
    eps: optional, a positive scalar value for epsilon, a small constant for
      numerical stability (default 1e-8).
    weight_decay: positive float
    m: positive integer, used for determining step when to start stochastic
      weight averaging
    k: positive integer, used to set frequency of swa

  Returns:
    An (init_fun, update_fun, get_params) triple.
  """

  step_size = make_schedule(step_size)
  lower_bound = make_schedule(step_size2)

  def init(x0):
    m0 = np.zeros_like(x0)
    v0 = np.zeros_like(x0)
    return x0, m0, v0,
  
  def update(i, g, state):
    x, m, v = state
    m = (1 - b1) * g + b1 * m  # First  moment estimate.
    v = (1 - b2) * ((g-m) ** 2) + b2 * v  # Second moment estimate.

    mhat = m / (1 - b1 ** (i + 1))  # Bias correction.
    vhat = v / (1 - b2 ** (i + 1))
    
    step = np.where(
      i <= offset,
      mhat * (step_size(i) / (np.sqrt(vhat) + eps)),
      g * step_size2(i)
    )
    x = ((1-weight_decay)*x) - step
    return x, m, v
  
  def get_params(state):
    x, m, v = state
    return x
  
  return init, update, get_params

@optimizer
def adabelief(
  step_size,
  b1=0.9, b2=0.99,
  eps=1e-8, weight_decay=1e-8,
  m=None, k=None
  ):
  """Construct optimizer triple for AdaBound.
  Args:
    step_size: positive scalar, or a callable representing a step size schedule
      that maps the iteration index to positive scalar.
    lower_bound: positive scalar, or a callable representing a step size schedule
      that maps the iteration index to positive scalar.
    b1: optional, a positive scalar value for beta_1, the exponential decay rate
      for the first moment estimates (default 0.9).
    b2: optional, a positive scalar value for beta_2, the exponential decay rate
      for the second moment estimates (default 0.999).
    eps: optional, a positive scalar value for epsilon, a small constant for
      numerical stability (default 1e-8).
    weight_decay: positive float
    m: positive integer, used for determining step when to start stochastic
      weight averaging
    k: positive integer, used to set frequency of swa

  Returns:
    An (init_fun, update_fun, get_params) triple.
  """

  step_size = make_schedule(step_size)

  def init(x0):
    m0 = np.zeros_like(x0)
    v0 = np.zeros_like(x0)
    return x0, m0, v0,
  
  def update(i, g, state):
    x, m, v = state
    m = (1 - b1) * g + b1 * m  # First  moment estimate.
    v = (1 - b2) * ((g-m) ** 2) + b2 * v  # Second moment estimate.
    alpha = step_size(i)

    mhat = m / (1 - b1 ** (i + 1))  # Bias correction.
    vhat = v / (1 - b2 ** (i + 1))
    step = alpha / (np.sqrt(vhat) + eps)
    
    # x = x - mhat * clipped
    x = ((1-weight_decay)*x) - mhat * step
    return x, m, v
  
  def get_params(state):
    x, m, v = state
    return x
  
  return init, update, get_params