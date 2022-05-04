import numpy as np
import jax.numpy as jnp
import jax
from jax import random
from jax.nn import (relu, log_softmax, softmax, softplus, sigmoid, elu,
                    leaky_relu, selu, gelu, normalize)
from jax.nn.initializers import glorot_uniform, glorot_normal, normal, ones, zeros
from .layers import (DenseGeneral, Dense, NeuralNet, NeuralNetGeneral, 
                     AttentionBlock, AttentionLayer, LayerNorm, Embed, Bijector)
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
    
    def apply_fun(params, X, placeholder1, placeholder2):
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


    vapply = jax.vmap(apply_fun, in_axes=(None, 0, None, None), out_axes=(0, 0, 0))

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

def AttentionModel_MAP(
        features,
        d_model=32,
        embed_hidden_size=64,
        embed_hidden_layers=5,
        embed_activation=relu,
        encoder_layers=4,
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
        unsupervised_pretraining=False
    ):
    # temp = 1 / (features - 1)
    init_embed, embedf, embedr = Embed(
        features, d_model, W_init=W_init, b_init=b_init
    )
    init_biject, bijectf, bijectr = Bijector(
        features, d_model, encoder_layers, W_init=W_init, b_init=b_init, tactivation=jax.nn.softplus, sactivation=jax.nn.tanh
    )
    init_sattn, sattention = NeuralNetGeneral(
        features, d_model, embed_hidden_size, features, embed_hidden_layers, W_init=W_init, b_init=b_init, activation=embed_activation
    )
    init_oattn, oattention = NeuralNetGeneral(
        features, d_model * 2, embed_hidden_size, 1, embed_hidden_layers, W_init=W_init, b_init=b_init, activation=embed_activation
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
        params["sattention"] = init_sattn(key)

        rng, key = random.split(rng)
        params["oattention"] = init_oattn(key)

        rng, key = random.split(rng)
        params["outnet"] = init_out(key)

        rng, key = random.split(rng)
        params["logits"] = jnp.zeros((1, features))

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

    def apply_fun(params, inputs, rng, dropout):
        # mask missing variables through forward pass
        nan_mask = jnp.where(jnp.isnan(inputs), 0.0, 1.0).reshape((1,-1))
        # replace nans in data to -1 to preserve output, will still be masked
        inputs = jnp.nan_to_num(inputs, nan=-1.0)
        # use concrete dropout to further drop features (1 ,f)
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
            # this line is 1.0 - drop_prob if you want to select more important variables
            # during training
            random_arr = drop_prob  # (1 if keeping 0 if removing)
            mask = nan_mask * random_arr
        else:
            mask = nan_mask

        # apply embedding to input of (features)
        x = inputs[..., None]
        z0_f = embedf(params["embed"], x) # (f, ndim)
        zk_f = bijectf(params["bijector"], z0_f) #  (f, ndim)
        if unsupervised_pretraining:
            sattn = pattn(sattention(params["sattention"], zk_f), mask, True)  # (f, f)
            zrk_f = jnp.matmul(sattn, zk_f)  # (f, ndim)
            zr0_f = bijectr(params["bijector"], zrk_f)
            return z0_f, zr0_f
        else:
            sattn = pattn(sattention(params["sattention"], zk_f), mask, True)  # (f, f)
            zrk_f = jnp.matmul(sattn, zk_f)  # (f, ndim)
            # enckey is necessary to carry information about included variables
            enckey = jnp.matmul(sattn, params["feat"])  # (f, ndim)
            zr0_f = bijectr(params["bijector"], zrk_f)

            oattn = pattn(
                jnp.transpose(
                    oattention(
                        params["oattention"],
                        jnp.concatenate([zk_f,enckey], -1)),
                    (1, 0)
                ), # (1, f)
                mask, False) # (1, f)
            zk = jnp.matmul(oattn, zk_f) # (1, ndim)
            logits = outnet(params["outnet"], zk)
            return jnp.squeeze(logits), oattn, z0_f, zk_f, zk
    if unsupervised_pretraining:
        vapply = jax.vmap(apply_fun, in_axes=(None, 0, 0, None), out_axes=(0, 0) )
    else:
        vapply = jax.vmap(apply_fun, in_axes=(None, 0, 0, None), out_axes=(0, 0, 0, 0, 0) )

    return init_fun, vapply


def AttentionModel_MAP_QK(
        features,
        d_model=32,
        embed_hidden_size=64,
        embed_hidden_layers=5,
        embed_activation=relu,
        encoder_layers=4,
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
        unsupervised_pretraining=False
    ):
    # temp = 1 / (features - 1)
    init_embed, embedf, embedr = Embed(
        features, d_model, W_init=W_init, b_init=b_init
    )
    init_biject, bijectf, bijectr = Bijector(
        features, d_model, encoder_layers, W_init=W_init, b_init=b_init, tactivation=jax.nn.softplus, sactivation=jax.nn.tanh
    )
    init_qs, qs = DenseGeneral(
        1, d_model, d_model, W_init=W_init, b_init=b_init
    )
    init_ks, ks = DenseGeneral(
        1, d_model, d_model, W_init=W_init, b_init=b_init
    )
    init_oattn, oattention = NeuralNetGeneral(
        1, d_model, embed_hidden_size, 1, W_init=W_init, b_init=b_init, activation=embed_activation
    )
    #init_sattn, sattention = NeuralNetGeneral(
    #    features, d_model, embed_hidden_size, features, embed_hidden_layers, W_init=W_init, b_init=b_init, activation=embed_activation
    #)
    #init_qa, qa = NeuralNetGeneral(
    #    features, d_model, embed_hidden_size, 1, embed_hidden_layers, W_init=W_init, b_init=b_init, activation=embed_activation
    #)
    init_out, outnet = NeuralNet(
        d_model, net_hidden_size, out_size, net_hidden_layers, W_init=W_init, b_init=b_init, activation=net_activation)

    def init_fun(rng):

        params = {}
        rng, key = random.split(rng)
        params["embed"] = init_embed(key)

        rng, key = random.split(rng)
        params["bijector"] = init_biject(key)

        #rng, key = random.split(rng)
        #params["sattention"] = init_sattn(key)

        rng, key = random.split(rng)
        params["xshift"] = W_init(key, (features, d_model))

        rng, key = random.split(rng)
        params["qs"] = init_qs(key)

        rng, key = random.split(rng)
        params["ks"] = init_ks(key)

        rng, key = random.split(rng)
        params["oattention"] = init_oattn(key)

        rng, key = random.split(rng)
        params["outnet"] = init_out(key)

        rng, key = random.split(rng)
        params["logits"] = jnp.zeros((1, features))

        return params

    def pattn(raw, mask, sa = False):
        # scaled softplus attention
        if sa:
            diag = 1.0 - jnp.diag(jnp.ones(features))
            mask = diag * mask
        scaled_attention = (jax.nn.softplus(raw) / jnp.sqrt(d_model)) * mask
        attention_weights = scaled_attention / (scaled_attention.sum(-1, keepdims=True) + 1e-4)
        return attention_weights

    def apply_fun(params, inputs, rng, dropout):
        # mask missing variables through forward pass
        nan_mask = jnp.where(jnp.isnan(inputs), 0.0, 1.0).reshape((1,-1))
        # replace nans in data to -1 to preserve output, will still be masked
        inputs = jnp.nan_to_num(inputs, nan=-1.0)
        # use concrete dropout to further drop features (1 ,f)
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
            # this line is 1.0 - drop_prob if you want to select more important variables
            # during training
            random_arr = drop_prob  # (1 if keeping 0 if removing)
            mask = nan_mask * random_arr
        else:
            mask = nan_mask

        # apply embedding to input of (features)
        x = inputs[..., None]
        z0_f = embedf(params["embed"], x) # (f, ndim)
        zk_f = bijectf(params["bijector"], z0_f) #  (f, ndim)
        if unsupervised_pretraining:
            q = qs(params["qs"], zk_f + params["xshift"])  # (f, ndim)
            k = ks(params["ks"], zk_f + params["xshift"])
            sattn = pattn(jnp.matmul(q, jnp.transpose(k, (1, 0))), mask, True)  # (f, f)
            zrk_f = jnp.matmul(sattn, zk_f)  # (f, ndim)
            zr0_f = bijectr(params["bijector"], zrk_f)
            return z0_f, zr0_f
        else:
            q = qs(params["qs"], zk_f + params["xshift"])  # (f, ndim)
            k = ks(params["ks"], zk_f + params["xshift"])
            sattn = pattn(jnp.matmul(q, jnp.transpose(k, (1, 0))), mask, True)  # (f, f)            
            zrk_f = jnp.matmul(sattn, zk_f)  # (f, ndim)
            zkf_zrkf = zk_f + zrk_f
            zr0_f = bijectr(params["bijector"], zrk_f)

            oattn = pattn(
                jnp.transpose(
                    oattention(params["oattention"], zkf_zrkf + params["xshift"]),
                    (1, 0)
                ), # (1, f)
                mask, False) # (1, f)
            zk = jnp.matmul(oattn, zk_f) # (1, ndim)
            logits = outnet(params["outnet"], zk)
            return jnp.squeeze(logits), oattn, z0_f, zr0_f, zk
    if unsupervised_pretraining:
        vapply = jax.vmap(apply_fun, in_axes=(None, 0, 0, None), out_axes=(0, 0) )
    else:
        vapply = jax.vmap(apply_fun, in_axes=(None, 0, 0, None), out_axes=(0, 0, 0, 0, 0) )

    return init_fun, vapply


def AttentionModel_MAP_OG(
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
        unsupervised_pretraining=False
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
        z1 = net1(params["net1"], x) + params["x_shift"]
        enc_output, sattn = enc(params["enc"], z1, mask=mask, enc_output=None)
        if unsupervised_pretraining:
            z2, attn = dec(params["dec"], params["x_shift"], enc_output=enc_output, mask=mask)
            mu = jnp.mean(z1, axis=-1)
            var = jnp.var(z1, axis=-1)
            kld = -0.5 * (1.0 + jnp.log(var) - var - (mu - 1)**2)
            z1 = jax.lax.stop_gradient(
                jnp.transpose(z1, (1, 0)) * nan_mask) # break gradient for ground truth, zero out missing variables
            z2 = jnp.transpose(z2, (1, 0)) * nan_mask # zero out missing values
            return z1, z2, kld
        else:
            z2, attn = dec(params["dec"], params["y"], enc_output=enc_output, mask=mask)
            h = net2(params["net2"], z2)
            logits = last_layer(params["last_layer"], h)
            attn = process_attn(sattn, attn)
            return jnp.squeeze(logits), attn, z2
    if unsupervised_pretraining:
        vapply = jax.vmap(apply_fun, in_axes=(None, 0, 0, None), out_axes=(0, 0, 0) )
    else:
        vapply = jax.vmap(apply_fun, in_axes=(None, 0, 0, None), out_axes=(0, 0, 0) )

    return init_fun, vapply

