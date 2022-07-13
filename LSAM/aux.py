import jax
import numpy as np
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten
from imblearn.over_sampling import RandomOverSampler

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


class oversampled_Kfold():
    def __init__(self, n_splits, n_repeats=1, key=42, resample=False):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.rng = np.random.default_rng(4321)
        self.ros = RandomOverSampler(random_state=key)
        self.resample = resample

    def get_n_splits(self, X=1, y=1, groups=None):
        return self.n_splits*self.n_repeats

    def split(self, X, y, groups=None):
        rows = np.arange(len(X))
        self.rng.shuffle(rows)
        splits = np.array_split(rows, self.n_splits)
        train, test = [], []
        for repeat in range(self.n_repeats):
            for idx in range(len(splits)):
                trainingIdx = np.concatenate(splits[:idx] + splits[idx+1:])
                if self.resample:
                    Xidx_r, y_r = self.ros.fit_resample(trainingIdx.reshape((-1,1)), y[trainingIdx.astype(int)])
                else:
                    Xidx_r, y_r = trainingIdx, trainingIdx
                train.append(Xidx_r.flatten())
                test.append(splits[idx])
        return list(zip(train, test))