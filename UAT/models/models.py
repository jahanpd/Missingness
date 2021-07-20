import numpy as np
import jax.numpy as jnp
import jax
from jax import random
from jax.nn import (relu, log_softmax, softmax, softplus, sigmoid, elu,
                    leaky_relu, selu, gelu, normalize)
from jax.nn.initializers import glorot_normal, normal, ones, zeros
from .layers import (DenseGeneral, Dense, NeuralNet, NeuralNetGeneral, 
                     AttentionBlock, AttentionLayer)
from itertools import combinations

def EnsembleModel(
        features,
        net_hidden_size=32,
        z_size=8,
        net_hidden_layers=5,
        W_init=glorot_normal(),
        b_init = zeros,
        activation=relu,
        reg={"reg":1e-5}
    ):
    f_init_funs = []
    f_apply_funs = []
    
    # create a list of indexes combinations
    cols = []
    for f in range(1,features):
        combs = list(combinations(list(range(features)), f))
        # tuples to list
        cols += [list(l) for l in combs]
    K = len(cols)

    for c in cols:
        f_init_fun, f_apply_fun = NeuralNet(
        len(c), net_hidden_size, z_size, net_hidden_layers, W_init=W_init, b_init=b_init, activation=activation)
        f_init_funs.append(f_init_fun)
        f_apply_funs.append(f_apply_fun)

    g_init_fun, g_apply_fun = NeuralNet(
        z_size, net_hidden_size, 1, net_hidden_layers, W_init=W_init, b_init=b_init, activation=activation)

    def init_fun(rng):
        params = {}
        f_params = []
        for i,f_init in enumerate(f_init_funs):
            rng, key = random.split(rng)
            f_params.append(f_init(key))
        params["f"] = f_params

        rng, key = random.split(rng)
        
        params["g"] = g_init_fun(key)

        rng, key = random.split(rng)
        params["null_set"] = normal()(key, (z_size,))


        return params
    
    def apply_fun(params, X):
        """ Takes a list of datasets U (derived from X) of length 2^D - 1 
            where D is the number of variables in original dataset X """

        # prep null set
        latent_space = []
        for i, (ds, f_) in enumerate(zip(cols, f_apply_funs)):
            U_k = X[np.array(ds)]
            z = f_(params["f"][i], U_k)
            latent_space.append(z)
        latent_space.append(jnp.ones_like(latent_space[0]) * params["null_set"])

        def g_fun(carry, x):
            logits = g_apply_fun(carry,x)
            return carry, logits
        latent_space = jnp.stack(latent_space, axis=0)
        # scan to avoid another for loop during jit compilation
        _, logits = jax.lax.scan(g_fun, params["g"], xs=latent_space)
        return logits, latent_space


    vapply = jax.vmap(apply_fun, in_axes=(None, 0), out_axes=(0, 0))

    return init_fun, vapply

def AttentionModel(
        features,
        d_model=32,
        embed_hidden_size=64,
        embed_hidden_layers=5,
        embed_activation=relu,
        encoder_layers=10,
        encoder_heads=5,
        enc_activation=relu,
        decoder_layers=10,
        decoder_heads=5,
        dec_activation=relu,
        net_hidden_size=64,
        net_hidden_layers=5,
        net_activation=relu,
        last_layer_size=64,
        out_size=1,
        W_init = glorot_normal(),
        b_init = zeros,
        temp = 0.1,
        eps = 1e-7,
        reg = {"reg":1e-5, "drop":1e-5}
    ):
    # temp = 1 / (features - 1)
    init_net1, net1 = NeuralNetGeneral(
        features, 1, embed_hidden_size, d_model, embed_hidden_layers, W_init=W_init, b_init=b_init, activation=embed_activation)
    init_enc, enc = AttentionBlock(
        encoder_layers, d_model, encoder_heads, W_init=W_init, b_init=b_init, activation=enc_activation)
    init_dec, dec = AttentionBlock(
        decoder_layers, d_model, decoder_heads, W_init=W_init, b_init=b_init, activation=dec_activation)
    init_net2, net2 = NeuralNet(
        d_model, net_hidden_size, last_layer_size, net_hidden_layers, W_init=W_init, b_init=b_init, activation=net_activation)
    init_ll, last_layer = Dense(last_layer_size, out_size, bias=True, W_init=W_init, b_init=b_init)

    def init_fun(rng):

        params = {}        
        rng, key = random.split(rng)
        params["net1"] = init_net1(key)
        rng, key = random.split(rng)
        params["enc"] = init_enc(key)
        rng, key = random.split(rng)
        params["dec"] = init_dec(key)
        rng, key = random.split(rng)
        params["net2"] = init_net2(key)
        rng, key = random.split(rng)
        params["last_layer"] = [init_ll(key)]
        rng, key = random.split(rng)
        params["y"] = W_init(key, (1, d_model))
        rng, key = random.split(rng)
        params["x_shift"] = W_init(key, (features, d_model))
        rng, key = random.split(rng)
        params["logits"] = jnp.zeros((1, features))

        return params

    def apply_fun(params, inputs, rng):
        # mask missing variables through forward pass
        nan_mask = jnp.where(jnp.isnan(inputs), 0.0, 1.0).reshape((1,-1))
        # replace nans in data to -1 to preserve output, will still be masked
        inputs = jnp.nan_to_num(inputs, nan=-1.0)
        # use concrete dropout to further drop features
        if rng is not None:
            probs = jax.nn.sigmoid(params["logits"])
            rng, unif_rng = random.split(rng)
            unif_noise = random.uniform(unif_rng, (1, features))
            drop_prob = (jnp.log(probs + eps) 
                            - jnp.log(1.0 - probs + eps)
                            + jnp.log(unif_noise + eps)
                            - jnp.log(1.0 - unif_noise + eps)
                            )
            drop_prob = jax.nn.sigmoid(drop_prob / temp)
            random_arr = 1.0 - drop_prob  # (1 if keeping 0 if removing)
            mask = nan_mask * random_arr
        else:
            mask = nan_mask

        # apply embedding to input of (features)
        x = inputs[..., None]
        z1 = net1(params["net1"], x) + params["x_shift"]
        enc_output, sattn = enc(params["enc"], z1, mask=mask, enc_output=None)
        z2, attn = dec(params["dec"], params["y"], enc_output=enc_output, mask=mask)
        h = net2(params["net2"], z2)
        logits = []
        for layer in params["last_layer"]:
            logits.append(last_layer(layer, h))

        # process attention
        # sattn = sattn.mean(0).mean(0)  # (feat, feat)
        attn = attn.mean(0).mean(0)  # (out, feat)
        # attn_out = attn @ sattn  # (out, feat)
        return logits, attn
    
    vapply = jax.vmap(apply_fun, in_axes=(None, 0, None), out_axes=(0, 0) )

    return init_fun, vapply

