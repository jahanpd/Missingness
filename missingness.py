import argparse
from jax.experimental.optimizers import l2_norm
from jax.interpreters.batching import batch
import numpy as np
import jax
from data import spiral_missing, thoracic, abalone
from matplotlib import pyplot as plt 
import matplotlib.cm as cm 
from UAT import UAT
from UAT import binary_cross_entropy


from UAT.models.models import EnsembleModel

init_fun, vapply = EnsembleModel(2)
params = init_fun(jax.random.PRNGKey(0))
out = vapply(params, np.random.rand(10,2))
print(out[0].shape)
print(out[1].shape)