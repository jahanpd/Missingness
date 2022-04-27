from UAT.models.layers import Embed, BijectLayer, Bijector
from UAT.models.models import AttentionModel_MAP
import jax.numpy as jnp
import jax
from jax import random
from jax._src.config import config

# config.update('jax_disable_jit', True)

features = 10
ndim = 17
n = 32
key = random.PRNGKey(323)

def generateData(key):
    key = random.PRNGKey(323)
    key, subkey = random.split(key)
    return random.normal(subkey, (n, features)), key


embed_init, embed_apply, embed_rapply = Embed(features, ndim)
embedParams = embed_init(key)

bl_init, bl_apply, bl_rapply = BijectLayer(features, ndim, 10)
blParams = bl_init(key)

bij_init, bij_apply, bij_rapply = Bijector(features, ndim, 10)
bijParams = bij_init(key)


embedApply = jax.vmap(embed_apply, in_axes=(None, 0, None), out_axes=0)
embedRapply = jax.vmap(embed_rapply, in_axes=(None, 0, None), out_axes=0)
blApply = jax.vmap(bl_apply, in_axes=(None, 0, None, None), out_axes=0)
blRapply = jax.vmap(bl_rapply, in_axes=(None, 0, None, None), out_axes=0)
bijApply = jax.vmap(bij_apply, in_axes=(None, 0, None), out_axes=0)
bijRapply = jax.vmap(bij_rapply, in_axes=(None, 0, None), out_axes=0)

def test(key):
    data, rng = generateData(key)
    x = data[..., None]
    # test embed
    y = embedApply(embedParams, data[..., None], False)
    xp = embedRapply(embedParams, y, False)
    diff = jnp.sum((xp - x)**2)
    assert diff < 1e-8
    print("embed passed")


    y1 = blApply(blParams, y, 1, False)
    yp = blRapply(blParams, y1, 1, False)
    diff = jnp.sum((yp - y)**2)
    assert diff < 1e-8
    print("bl odd passed")

    y1 = blApply(blParams, y, 2, False)
    yp = blRapply(blParams, y1, 2, False)
    diff = jnp.sum((yp - y)**2)
    assert diff < 1e-8
    print("bl even passed")

    y1 = bijApply(bijParams, y, True)
    diff = jnp.sum((y1 - y)**2)
    yp = bijRapply(bijParams, y1, True)
    diff = jnp.sum((yp - y)**2)
    assert diff < 1e-8
    print("bijector passed")
