import numpy as np
import jax.numpy as jnp
import jax
from jax import random
from jax.nn import (relu, log_softmax, softmax, softplus, sigmoid, elu,
                    leaky_relu, selu, gelu, normalize)
from jax.nn.initializers import glorot_uniform, glorot_normal, normal, ones, zeros
from .layers import (DenseGeneral, Dense, ConcDropNeuralNet, NeuralNet, NeuralNetGeneral, 
                     AttentionBlock, AttentionLayer, LayerNorm, Embed, Bijector,
                     AttentionLayer2, AttentionBlock2)
from itertools import combinations


def EnsembleModel(
        features,
        net_hidden_size=32,
        z_size=8,
        net_hidden_layers=5,
        W_init=glorot_normal(),
        b_init = zeros,
        activation=relu,
    ):
    f_init_funs = []
    f_apply_funs = []
    
    # create a list of indexes combinations
    cols = []
    for f in range(1,features+1):
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

        params["set_order"] = ones(key, (len(cols)+1,) )


        return params
    
    def apply_fun(params, X, placeholder1, placeholder2, placeholder3):
        """ Takes a list of datasets U (derived from X) of length 2^D - 1 
            where D is the number of variables in original dataset X """ 

        # prep null set
        latent_space = []
        nan_mask = []
        for i, (ds, f_) in enumerate(zip(cols, f_apply_funs)):
            U_k = X[np.array(ds)]
            nan_mask.append(jnp.isnan(U_k.sum()))
            U_k = jnp.nan_to_num(U_k, nan=0.0)
            z = f_(params["f"][i], U_k)
            latent_space.append(z)
        latent_space.append(jnp.ones_like(latent_space[0]) * params["null_set"])
        nan_mask.append(False) # to account for empty set
        nan_mask = jnp.where(jnp.array(nan_mask), 0.0, 1.0)

        def g_fun(carry, x):
            logits = g_apply_fun(carry,x)
            return carry, logits
        latent_space = jnp.stack(latent_space, axis=0)
        # scan to avoid another for loop during jit compilation
        _, logits = jax.lax.scan(g_fun, params["g"], xs=latent_space)
        return logits, latent_space, nan_mask


    vapply = jax.vmap(apply_fun, in_axes=(None, 0, None, None, None), out_axes=(0, 0, 0))

    return init_fun, vapply, cols


# convenience function for processing attention
def process_attn(sattn, attn):
    # process attention
    # the average salf attention over layers and heads
    sattn = sattn.mean(0).mean(0)  # (layers, heads, feat, feat) -> (feat, feat)
    # renormalize
    sattn = sattn / sattn.sum(axis=1, keepdims=True)
    attn = attn.mean(0).mean(0)  # (layers, heads, out, feat) -> (out, feat)
    # renormalize
    attn = attn / attn.sum(axis=1, keepdims=True)
    attn_out = attn @ sattn  # (out, feat)
    return  attn_out


def MaskedNeuralNet(
        features,
        d_model=32,
        embed_hidden_size=64,
        embed_hidden_layers=5,
        embed_activation=relu,
        out_size=1,
        W_init = glorot_uniform(),
        b_init = zeros,
        temp = 0.1,
        eps = 1e-7,
        **kwargs
    ):
    init_fk, fk = ConcDropNeuralNet(
        features, embed_hidden_size, out_size, embed_hidden_layers, W_init=W_init, b_init=b_init, activation=embed_activation
    )
    def init_fun(rng):

        params = {}
        rng, key = random.split(rng)
        params["fk"] = init_fk(key)

        return params

    def apply_fun(params, inputs, rng, dropout, train):
        # mask missing variables through forward pass
        rng, key = random.split(rng)
        rand_noise = random.normal(key, inputs.shape)
        # replace nans in data to with random noise if training
        if train:
            inputs = jnp.where(jnp.isnan(inputs), rand_noise, inputs)
            nan_mask = jnp.ones(inputs.shape).reshape((-1,1))
        else:
            nan_mask = jnp.where(jnp.isnan(inputs), 0.0, 1.0).reshape((-1,1))
            inputs = jnp.nan_to_num(inputs, nan=-1.0)
        # use concrete dropout to further drop features (f,)
        if dropout:
            probs = jax.nn.sigmoid(params["fk"]["l1_logit"])
            rng, unif_rng = random.split(rng)
            unif_noise = random.uniform(unif_rng, params["fk"]["l1_logit"].shape)
            drop_prob = (jnp.log(probs + eps)
                            - jnp.log(1.0 - probs + eps)
                            + jnp.log(unif_noise + eps)
                            - jnp.log(1.0 - unif_noise + eps)
                            )
            drop_prob = jax.nn.sigmoid(drop_prob / temp)
            # this line is 1.0 - drop_prob if you want to select more important variables
            # during training
            random_arr = 1. - drop_prob  # (1 if keeping 0 if removing)
            mask = nan_mask * random_arr
        else:
            mask = nan_mask

        rng, key = random.split(rng)
        logits, hiddens = fk(params["fk"], inputs, mask1=mask, rng=key)
        z = hiddens[-1, ...]
        return jnp.squeeze(logits), z, z, z

    vapply = jax.vmap(apply_fun, in_axes=(None, 0, 0, None, None), out_axes=(0, 0, 0, 0) )

    return init_fun, vapply

def AttentionModel2(
        features,
        d_model=32,
        embed_hidden_layers=2,
        decoder_layers=10,
        decoder_heads=5,
        dec_activation=relu,
        net_hidden_size=64,
        net_hidden_layers=5,
        net_activation=relu,
        out_size=1,
        W_init = glorot_uniform(),
        b_init = zeros,
        temp = 0.1,
        eps = 1e-7,
        **kwargs
    ):
    # temp = 1 / (features - 1)
    init_embed, embedf, embedr = Embed(
        features, d_model, W_init=jax.nn.initializers.ones, b_init=zeros
    )
    init_biject, bijectf, bijectr = Bijector(
        1, d_model, embed_hidden_layers, W_init=W_init, b_init=b_init, tactivation=jax.nn.softplus, sactivation=jax.nn.tanh
    )
    init_attnblk, attnblock = AttentionBlock2(
        decoder_layers, d_model, decoder_heads, W_init=W_init, b_init=b_init, activation=dec_activation
    )
    init_out, outnet = NeuralNet(
        d_model, net_hidden_size, out_size, net_hidden_layers, W_init=W_init, b_init=b_init, activation=net_activation)

    def init_fun(rng):

        params = {}
        rng, key = random.split(rng)
        params["embed"] = init_embed(key)

        rng, key = random.split(rng)
        params["bijector"] = init_biject(key)

        rng, key = random.split(rng)
        params["bijector2"] = init_biject(key)

        rng, key = random.split(rng)
        params["bijector3"] = init_biject(key)

        rng, key = random.split(rng)
        params["attnblk"] = init_attnblk(key)

        rng, key = random.split(rng)
        params["outnet"] = init_out(key)

        rng, key = random.split(rng)
        params["logits"] = jnp.ones((1, features)) * -1.0

        rng, key = random.split(rng)
        params["feat"] = W_init(key, (features, d_model))

        return params

    def apply_fun(params, inputs, rng, dropout, train):
        # mask missing variables through forward pass
        rng, key = random.split(rng)
        # replace nans in data with a placeholder
        if train:
            # nan_mask = jnp.ones_like(inputs).reshape((1, -1))
            nan_mask = jnp.where(jnp.isnan(inputs), 0.0, 1.0).reshape((1, -1))
        else:
            nan_mask = jnp.where(jnp.isnan(inputs), 0.0, 1.0).reshape((1, -1))
        inputs = jnp.nan_to_num(inputs, nan=0.0)
        # use concrete dropout to further drop features (f,)
        if dropout:
            probs = jax.nn.sigmoid(params["logits"])
            rng, unif_rng = random.split(rng)
            unif_noise = random.uniform(unif_rng, params["logits"].shape)
            drop_prob = (jnp.log(probs + eps)
                            - jnp.log(1.0 - probs + eps)
                            + jnp.log(unif_noise + eps)
                            - jnp.log(1.0 - unif_noise + eps)
                            )
            drop_prob = jax.nn.sigmoid(drop_prob / temp)
            # this line is 1.0 - drop_prob if you want to select more important variables
            # during training
            random_arr = 1. - drop_prob  # (1 if keeping 0 if removing)
            mask = nan_mask * random_arr
        else:
            mask = nan_mask

        # apply embedding to input of (features)
        x = inputs[..., None]
        z0_f = embedf(params["embed"], x) # (f, ndim)
        zk_f = bijectf(params["bijector"], z0_f + params["feat"]) #  (f, ndim)
        rand_noise = random.normal(key, zk_f.shape)
        zk_f = jnp.where(jnp.transpose(mask, (1,0)) == 1, zk_f, rand_noise)
        all_logits = outnet(params["outnet"], zk_f)  # (f, o)
        zkm, attn = attnblock(
            params["attnblk"],
            zk_f,
            zk_f,
            mask=mask)
        logits = outnet(params["outnet"], zkm)
        return jnp.squeeze(logits), attn, zk_f, all_logits, zkm

    vapply = jax.vmap(apply_fun, in_axes=(None, 0, 0, None, None), out_axes=(0, 0, 0, 0, 0) )

    return init_fun, vapply

def MixtureModel(
        features,
        d_model=32,
        embed_hidden_layers=2,
        embed_activation=relu,
        dec_activation=relu,
        net_hidden_size=64,
        net_hidden_layers=5,
        net_activation=relu,
        out_size=1,
        W_init = glorot_uniform(),
        b_init = zeros,
        temp = 0.1,
        eps = 1e-7,
        **kwargs
    ):
    # temp = 1 / (features - 1)
    #init_embed, embedf, embedr = Embed(
    #    features, d_model, W_init=jax.nn.initializers.ones, b_init=zeros
    #)
    init_net1, net1 = NeuralNetGeneral(
        features, 1, d_model, d_model, embed_hidden_layers, W_init=W_init, b_init=b_init, activation=embed_activation
    )
    init_biject, bijectf, bijectr = Bijector(
        features, d_model, embed_hidden_layers, W_init=W_init, b_init=b_init, tactivation=jax.nn.softplus, sactivation=jax.nn.tanh
    )
    init_attn, attnf = NeuralNet(
        d_model, d_model, 1, net_hidden_layers, W_init=W_init, b_init=b_init, activation=dec_activation
    )
    init_out, outnet = NeuralNet(
        d_model, net_hidden_size, out_size, net_hidden_layers, W_init=W_init, b_init=b_init, activation=net_activation)

    def init_fun(rng):

        params = {}
        rng, key = random.split(rng)
        params["embed"] = init_net1(key)

        rng, key = random.split(rng)
        params["attn"] = init_attn(key)

        rng, key = random.split(rng)
        params["outnet"] = init_out(key)

        rng, key = random.split(rng)
        params["logits"] = jnp.ones((1, features)) * 0.

        rng, key = random.split(rng)
        params["feat"] = W_init(key, (features, d_model))

        return params

    def pattn(raw, mask, sa = False):
        # scaled softplus attention
        if sa:
            diag = 1.0 - jnp.diag(jnp.ones(features))
            mask = diag * mask
        scaled_attention = (jax.nn.softplus(raw) / jnp.sqrt(d_model)) * mask
        attention_weights = scaled_attention / (scaled_attention.sum(-1, keepdims=True) + 1e-4)
        return attention_weights

    def apply_fun(params, inputs, rng, dropout, train):
        # mask missing variables through forward pass
        rng, key = random.split(rng)
        rand_noise = random.normal(key, inputs.shape)
        # replace nans in data to with random noise if training
        if train:
            inputs = jnp.where(jnp.isnan(inputs), rand_noise, inputs)
            nan_mask = jnp.ones(inputs.shape).reshape((1,-1))
        else:
            nan_mask = jnp.where(jnp.isnan(inputs), 0.0, 1.0).reshape((1, -1))
            inputs = jnp.nan_to_num(inputs, nan=-1.0)
        # use concrete dropout to further drop features (f,)
        if dropout:
            probs = jax.nn.sigmoid(params["logits"])
            rng, unif_rng = random.split(rng)
            unif_noise = random.uniform(unif_rng, params["logits"].shape)
            drop_prob = (jnp.log(probs + eps)
                            - jnp.log(1.0 - probs + eps)
                            + jnp.log(unif_noise + eps)
                            - jnp.log(1.0 - unif_noise + eps)
                            )
            drop_prob = jax.nn.sigmoid(drop_prob / temp)
            # this line is 1.0 - drop_prob if you want to select more important variables
            # during training
            random_arr = 1. - drop_prob  # (1 if keeping 0 if removing)
            mask = nan_mask * random_arr
        else:
            mask = nan_mask

        # apply embedding to input of (features)
        x = inputs[..., None]
        zk_f = net1(params["embed"], x) #  (f, ndim)
        zk_f = (zk_f * jnp.transpose(mask, (1, 0)))
        attn = pattn(
            jnp.transpose(
                attnf(params["attn"], zk_f + params["feat"]), (1, 0)  # (1, f)
            ),
            mask,
            False
        )
        zkm = jnp.matmul(attn, zk_f)  # (1, dims)
        fkm = jnp.matmul(attn, params["feat"]) # (1, dims)
        all_logits = outnet(params["outnet"], zk_f + params["feat"])  # (f, dims)
        logits = outnet(params["outnet"], zkm + fkm)
        return jnp.squeeze(logits), attn, zk_f, all_logits, mask, zkm

    vapply = jax.vmap(apply_fun, in_axes=(None, 0, 0, None, None), out_axes=(0, 0, 0, 0, 0, 0) )

    return init_fun, vapply

def CentroidCluster(
        features,
        d_model=32,
        embed_hidden_layers=2,
        net_hidden_size=64,
        net_hidden_layers=5,
        net_activation=relu,
        out_size=1,
        W_init = glorot_uniform(),
        b_init = zeros,
        temp = 0.1,
        eps = 1e-7,
        **kwargs
    ):
    # temp = 1 / (features - 1)
    init_embed, embedf, embedr = Embed(
        features, d_model, W_init=jax.nn.initializers.ones, b_init=zeros
    )
    init_biject, bijectf, bijectr = Bijector(
        features, d_model, embed_hidden_layers, W_init=W_init, b_init=b_init, tactivation=jax.nn.softplus, sactivation=jax.nn.tanh
    )
    init_ff, ff = NeuralNet(
        d_model, net_hidden_size, d_model, net_hidden_layers, W_init=W_init, b_init=b_init, activation=net_activation
    )
    init_gf, gf = NeuralNet(
        d_model, net_hidden_size, out_size, net_hidden_layers, W_init=W_init, b_init=b_init, activation=net_activation)

    def init_fun(rng):

        params = {}
        rng, key = random.split(rng)
        params["embed"] = init_embed(key)

        rng, key = random.split(rng)
        params["bijector"] = init_biject(key)

        rng, key = random.split(rng)
        params["bijector2"] = init_biject(key)

        rng, key = random.split(rng)
        params["ff"] = init_ff(key)

        rng, key = random.split(rng)
        params["gf"] = init_gf(key)

        rng, key = random.split(rng)
        params["logits"] = jnp.zeros((features, 1))

        return params

    def apply_fun(params, inputs, rng, dropout):
        # mask missing variables through forward pass
        nan_mask = jnp.where(jnp.isnan(inputs), 0.0, 1.0).reshape((-1,1))
        # replace nans in data to -1 to preserve output, will still be masked
        inputs = jnp.nan_to_num(inputs, nan=-1.0)
        # use concrete dropout to further drop features (1 ,f)
        if dropout:
            probs = jax.nn.sigmoid(params["logits"])
            rng, unif_rng = random.split(rng)
            unif_noise = random.uniform(unif_rng, (features, 1))
            drop_prob = (jnp.log(probs + eps)
                            - jnp.log(1.0 - probs + eps)
                            + jnp.log(unif_noise + eps)
                            - jnp.log(1.0 - unif_noise + eps)
                            )
            drop_prob = jax.nn.sigmoid(drop_prob / temp)
            # this line is 1.0 - drop_prob if you want to select more important variables
            # during training
            random_arr = 1. - drop_prob  # (1 if keeping 0 if removing)
            mask = nan_mask * random_arr
        else:
            mask = nan_mask

        # apply embedding to input of (features)
        x = inputs[..., None]
        z0_f = embedf(params["embed"], x) # (f, ndim)
        zk_f = bijectf(params["bijector"], z0_f) #  (f, ndim)
        zk_f = bijectf(params["bijector2"], zk_f) #  (f, ndim)
        centroid = (zk_f * mask).mean(0, keepdims=False)  # (ndim,)

        z = ff(params["ff"], jax.lax.stop_gradient(centroid))
        logits = gf(params["gf"], z)
        return jnp.squeeze(logits), 0.0, zk_f, z

    vapply = jax.vmap(apply_fun, in_axes=(None, 0, 0, None), out_axes=(0, 0, 0, 0) )

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
        W_init = glorot_uniform(),
        b_init = zeros,
        temp = 0.1,
        eps = 1e-7,
        unsupervised_pretraining=False,
        **kwargs
    ):
    # temp = 1 / (features - 1)
    init_embed, embedf, embedr = Embed(
        features, d_model, W_init=W_init, b_init=b_init
    )
    init_biject, bijectf, bijectr = Bijector(
        features, d_model, embed_hidden_layers, W_init=W_init, b_init=b_init, tactivation=jax.nn.softplus, sactivation=jax.nn.tanh
    )
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
        params["embed"] = init_embed(key)
        rng, key = random.split(rng)
        params["bijector"] = init_biject(key)
        rng, key = random.split(rng)
        params["bijector2"] = init_biject(key)
        rng, key = random.split(rng)
        params["bijector3"] = init_biject(key)
        rng, key = random.split(rng)
        params["bijector4"] = init_biject(key)
        rng, key = random.split(rng)
        params["enc"] = init_enc(key)
        rng, key = random.split(rng)
        params["dec"] = init_dec(key)
        rng, key = random.split(rng)
        params["net2"] = init_net2(key)
        rng, key = random.split(rng)
        params["last_layer"] = init_ll(key)
        rng, key = random.split(rng)
        params["y"] = W_init(key, (1, d_model))
        rng, key = random.split(rng)
        params["x_shift"] = W_init(key, (features, d_model))
        rng, key = random.split(rng)
        params["logits"] = jnp.zeros((1, features))

        return params

    def apply_fun(params, inputs, rng, dropout):
        # mask missing variables through forward pass
        nan_mask = jnp.where(jnp.isnan(inputs), 0.0, 1.0).reshape((1,-1))
        # replace nans in data to -1 to preserve output, will still be masked
        inputs = jnp.nan_to_num(inputs, nan=-1.0)
        # use concrete dropout to further drop features
        if dropout:
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
        z0_f = embedf(params["embed"], x) # (f, ndim)
        zk_f = bijectf(params["bijector"], z0_f) #  (f, ndim)
        zk_f = bijectf(params["bijector2"], z0_f + zk_f) #  (f, ndim)
        zk_f = bijectf(params["bijector3"], z0_f + zk_f) #  (f, ndim)
        zk_f = bijectf(params["bijector4"], z0_f + zk_f) #  (f, ndim)
        z1 = zk_f + params["x_shift"]
        enc_output, sattn = enc(params["enc"], z1, mask=mask, enc_output=None)
        z2, attn = dec(params["dec"], params["y"], enc_output=enc_output, mask=mask)
        h = net2(params["net2"], z2)
        logits = last_layer(params["last_layer"], h)
        attn = process_attn(sattn, attn)
        return jnp.squeeze(logits), attn, z0_f, z2

    vapply = jax.vmap(apply_fun, in_axes=(None, 0, 0, None), out_axes=(0, 0, 0, 0) )

    return init_fun, vapply

def AttentionModel_MAP(
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
        outcomes=1,
        W_init = glorot_uniform(),
        b_init = zeros,
        temp = 0.1,
        eps = 1e-7,
        unsupervised_pretraining=False,
        noise_std=0.1,
        **kwargs
    ):
    # temp = 1 / (features - 1)
    init_net1, net1 = NeuralNetGeneral(
        features, 1, embed_hidden_size, d_model, embed_hidden_layers, W_init=W_init, b_init=b_init, activation=embed_activation)
    init_enc, enc = AttentionBlock(
        encoder_layers, d_model, encoder_heads, depth=1, W_init=W_init, b_init=b_init, activation=enc_activation)
    init_dec, dec = AttentionBlock(
        decoder_layers, d_model, decoder_heads, depth=1, W_init=W_init, b_init=b_init, activation=dec_activation)
    init_net2, net2 = NeuralNet(
        d_model, net_hidden_size, d_model, net_hidden_layers, W_init=W_init, b_init=b_init, activation=net_activation)
    init_ll, last_layer = Dense(d_model, out_size, bias=True, W_init=W_init, b_init=b_init)
    init_ln, layernorm = LayerNorm()

    def init_fun(rng):

        params = {}
        rng, key = random.split(rng)
        params["net11"] = init_net1(key)
        rng, key = random.split(rng)
        params["enc"] = init_enc(key)
        rng, key = random.split(rng)
        params["dec"] = init_dec(key)
        rng, key = random.split(rng)
        params["net2"] = init_net2(key)
        rng, key = random.split(rng)
        params["last_layer"] = init_ll(key)
        rng, key = random.split(rng)
        # params["x_shift"] = W_init(key, (features, d_model))
        # params["x_shift"] = jnp.zeros((outcomes, d_model))
        params["x_shift"] = random.normal(key, (outcomes, d_model))
        rng, key = random.split(rng)
        params["y"] = W_init(key, (outcomes, d_model))
        rng, key = random.split(rng)
        # params["y_shift"] = W_init(key, (outcomes, d_model))
        # params["y_shift"] = jnp.zeros((outcomes, d_model))
        params["y_shift"] = random.normal(key, (outcomes, d_model))
        rng, key = random.split(rng)
        params["logits"] = jnp.zeros((1, features))
        rng, key = random.split(rng)
        params["ln1"] = init_ln(key, d_model)
        rng, key = random.split(rng)
        params["ln2"] = init_ln(key, d_model)


        return params

    def apply_fun(params, inputs, rng, dropout, train):
        # mask missing variables through forward pass
        nan_mask = jnp.where(jnp.isnan(inputs), 0.0, 1.0).reshape((1,-1))
        # replace nans in data to -1 to preserve output, will still be masked
        inputs = jnp.nan_to_num(inputs, nan=-1.0)
        # use concrete dropout to further drop features (f,)
        if dropout:
            probs = jax.nn.sigmoid(params["logits"])
            rng, unif_rng = random.split(rng)
            unif_noise = random.uniform(unif_rng, params["logits"].shape)
            drop_prob = (jnp.log(probs + eps)
                            - jnp.log(1.0 - probs + eps)
                            + jnp.log(unif_noise + eps)
                            - jnp.log(1.0 - unif_noise + eps)
                            )
            drop_prob = jax.nn.sigmoid(drop_prob / temp)
            # this line is 1.0 - drop_prob if you want to select more important variables
            # during training
            random_arr = 1. - drop_prob  # (1 if keeping 0 if removing)
            mask = nan_mask * random_arr
        else:
            mask = nan_mask

        # apply embedding to input of (features)
        x = inputs[..., None]
        zk_f = layernorm(params["ln1"], net1(params["net11"], x)) + params["x_shift"] #  (f, ndim)
        # zk_f = net1(params["net11"], x) + params["x_shift"] #  (f, ndim)
        rng, key = random.split(rng)
        # if train:
        #     rng, key = random.split(rng)
        #     zk_f = zk_f + (random.normal(key, zk_f.shape)*noise_std)
        enc_output, sattn = enc(params["enc"], zk_f, mask=mask, enc_output=None)
        if unsupervised_pretraining:
            z2, attn = dec(params["dec"], params["x_shift"], enc_output=enc_output, mask=mask)
            z2 = net2(params["net2"], z2)
            mu = jnp.mean(zk_f, axis=-1)
            var = jnp.var(zk_f, axis=-1)
            kld = -0.5 * (1.0 + jnp.log(var) - var - (mu - 1)**2)
            z1 = jax.lax.stop_gradient(
                jnp.transpose(zk_f, (1, 0)) * nan_mask) # break gradient for ground truth, zero out missing variables
            z2 = jnp.transpose(z2, (1, 0)) * nan_mask # zero out missing values
            return z1, z2, kld
        else:
            # z2 (1, ndim)
            z2, attn = dec(params["dec"], params["y"], enc_output=enc_output, mask=mask)
            z2 =layernorm(params["ln2"], z2) + params["y_shift"] #  (f, ndim)
            # if train:
            #     rng, key = random.split(rng)
            #     z2 = z2 + (random.normal(key, z2.shape)*noise_std)
            h = net2(params["net2"], z2)
            logits = last_layer(params["last_layer"], h)
            attn = process_attn(sattn, attn)
            return jnp.squeeze(logits), attn, zk_f, z2
    if unsupervised_pretraining:
        vapply = jax.vmap(apply_fun, in_axes=(None, 0, 0, None, None), out_axes=(0, 0, 0) )
    else:
        vapply = jax.vmap(apply_fun, in_axes=(None, 0, 0, None, None), out_axes=(0, 0, 0, 0) )

    return init_fun, vapply
