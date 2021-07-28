import numpy as np
import jax.numpy as jnp
import jax
from jax import random
from jax.nn import (relu, log_softmax, softmax, softplus, sigmoid, elu,
                    leaky_relu, selu, gelu, normalize)
from jax.nn.initializers import glorot_normal, normal, ones, zeros

def DenseGeneral(
    features,
    in_dim,
    out_dim,
    W_init = glorot_normal(),
    b_init = normal(),
    ):
    def init_fun(
        rng,
        ):
        k1, k2 = random.split(rng, 2)
        W = W_init(k1, (features, in_dim, out_dim))
        b = b_init(k2, (features, out_dim,))
        return (W, b)
    def apply_fun(params, x, scan=False):
        if scan:
            W, b = params["hw"], params["hb"]
        else:
            W, b = params
        return jnp.einsum('...ij,...ijk->...ik', x, W) + b
    return init_fun, apply_fun

def Dense(
    in_dim,
    out_dim,
    bias = True,
    W_init = glorot_normal(),
    b_init = normal(),
    ):
    def init_fun(
        rng
        ):
        k1, k2 = random.split(rng, 2)
        out_shape = (out_dim)
        W = W_init(k1, (in_dim, out_dim))
        if bias:
            b = b_init(k2, (out_dim,))
            return (W, b)
        else:
            return (W)
    def apply_fun(params, x, scan = False):
        if bias:
            if scan:
                W, b = params["hw"], params["hb"]
            else:
                W, b = params
            return jnp.einsum('...i,...ij->...j', x, W) + b
        else:
            if scan:
                W = params["hw"]
            else:
                W = params
            return jnp.einsum('...i,...ij->...j', x, W)
    return init_fun, apply_fun


def NeuralNetGeneral(
    features,
    in_dim,
    hidden_dim,
    out_dim,
    num_hidden = 1,
    W_init = glorot_normal(),
    b_init = normal(),
    activation=sigmoid
    ):
    init_l1, layer_1 = DenseGeneral(
        features, in_dim, hidden_dim, W_init=W_init, b_init=b_init)
    init_hidden, layer_h = DenseGeneral(
        features, hidden_dim, hidden_dim, W_init=W_init, b_init=b_init)
    init_out, layer_out = DenseGeneral(
        features, hidden_dim, out_dim, W_init=W_init, b_init=b_init)
    
    def init_fun(rng):
        params = {}
        rng, key = random.split(rng)
        params["l1"] = init_l1(key)
        
        hw, hb = [], []
        for _ in range(num_hidden):
            rng, key = random.split(rng)
            W, b = init_hidden(key)
            hw.append(W)
            hb.append(b)
        
        params["hidden"] = {
            "hw":jnp.stack(hw, axis=0),
            "hb":jnp.stack(hb, axis=0)
        }

        rng, key = random.split(rng)
        params["out"] = init_out(key)

        return params  
  
    def apply_fun(params, inputs):
        h = activation(layer_1(params["l1"], inputs))

        def body(carry, x):
            # note x is actually the params
            temp = activation(layer_h(x, carry, scan=True))
            return temp, None
        
        h, _ = jax.lax.scan(body, h, params["hidden"])

        out = layer_out(params["out"], h)
        return out
    
    return init_fun, apply_fun

def NeuralNet(
    in_dim,
    hidden_dim,
    out_dim,
    num_hidden = 1,
    W_init = glorot_normal(),
    b_init = normal(),
    activation=relu
    ):
    init_l1, layer_1 = Dense(
        in_dim, hidden_dim, W_init=W_init, b_init=b_init)
    init_hidden, layer_h = Dense(
        hidden_dim, hidden_dim, W_init=W_init, b_init=b_init)
    init_out, layer_out = Dense(
        hidden_dim, out_dim, W_init=W_init, b_init=b_init)
    
    def init_fun(rng):
        params = {}
        rng, key = random.split(rng)
        params["l1"] = init_l1(key)
        
        hw, hb = [], []
        for _ in range(num_hidden):
            rng, key = random.split(rng)
            W, b = init_hidden(key)
            hw.append(W)
            hb.append(b)
        
        params["hidden"] = {
            "hw":jnp.stack(hw, axis=0),
            "hb":jnp.stack(hb, axis=0)
        }

        rng, key = random.split(rng)
        params["out"] = init_out(key)

        return params
    
    def apply_fun(params, inputs, rng=None):
        h = activation(layer_1(params["l1"], inputs))

        def body(carry, x):
            # note x is actually the params
            temp = activation(layer_h(x, carry, scan=True))
            return temp, None
        
        h, _ = jax.lax.scan(body, h, params["hidden"])

        out = layer_out(params["out"], h)
        return out
    
    return init_fun, apply_fun


def AttentionLayer(
    heads,
    dims,
    dff = None,
    W_init = glorot_normal(),
    b_init = normal(),
    activation = softplus
    ):
    if dff is None:
        dff = dims * 4
    init_q, q_map = Dense(dims, dims*heads, W_init=W_init, b_init=b_init)
    init_k, k_map = Dense(dims, dims*heads, W_init=W_init, b_init=b_init)
    init_v, v_map = Dense(dims, dims*heads, W_init=W_init, b_init=b_init)
    init_out, out = Dense(dims*heads, dims, W_init=W_init, b_init=b_init)
    # feeedforward network
    init_l1, l1 = Dense(dims, dff, W_init=W_init, b_init=b_init)
    init_l2, l2 = Dense(dff, dims, W_init=W_init, b_init=b_init)

    def init_fun(rng):
        params = {}
        rng, key = random.split(rng)
        params["q"] = init_q(key)
        rng, key = random.split(rng)
        params["k"] = init_k(key)
        rng, key = random.split(rng)
        params["v"] = init_v(key)
        rng, key = random.split(rng)
        params["out"] = init_out(key)
        rng, key = random.split(rng)
        params["l1"] = init_l1(key)
        rng, key = random.split(rng)
        params["l2"] = init_l2(key)
        return params
    
    def apply_fun(params, q, k, v, mask):
        # note params input is from the AttentionBlock init_params construction, not the above

        q_ = jnp.transpose(q_map((params["qw"],params["qb"]), q).reshape((-1, heads, dims)), (1,0,2))
        k_ = jnp.transpose(k_map((params["kw"],params["kb"]), k).reshape((-1, heads, dims)), (1,2,0))
        v_ = jnp.transpose(v_map((params["vw"],params["vb"]), v).reshape((-1, heads, dims)), (1,0,2))

        # scaled dot product attention
        qk = jnp.matmul(q_,k_)
        scaled_attention_logits = jax.nn.softplus(qk / jnp.sqrt(dims)) * mask
        attention_weights = scaled_attention_logits / (scaled_attention_logits.sum(-1, keepdims=True) + 1e-6)
        scaled_attention = jnp.transpose(jnp.matmul(attention_weights, v_), (1,0,2)).reshape((-1, dims*heads))
        x = out((params["outw"],params["outb"]), scaled_attention)

        # residual connection
        x = x + q
        # feedforward network
        x = activation(l1((params["l1w"],params["l1b"]), x))
        x = l2((params["l2w"],params["l2b"]), x)

        return x, attention_weights
    
    return init_fun, apply_fun

def AttentionBlock(
        num_layers,
        dims,
        heads,
        dff=None,
        W_init = glorot_normal(),
        b_init = normal(),
        activation = softplus
    ):
    
    init_attn, apply_attn = AttentionLayer(
        heads, dims, dff=dff, W_init=W_init, b_init=b_init, activation=activation
    )

    def init_fun(rng):
        
        qw, qb = [], []
        kw, kb = [], []
        vw, vb = [], []
        outw, outb = [], []
        l1w, l1b = [], []
        l2w, l2b = [], []

        for _ in range(num_layers):
            rng, layer_rng = random.split(rng)
            params = init_attn(layer_rng)
            qw.append(params["q"][0])
            qb.append(params["q"][1])
            kw.append(params["k"][0])
            kb.append(params["k"][1])
            vw.append(params["v"][0])
            vb.append(params["v"][1])
            outw.append(params["out"][0])
            outb.append(params["out"][1])
            l1w.append(params["l1"][0])
            l1b.append(params["l1"][1])
            l2w.append(params["l2"][0])
            l2b.append(params["l2"][1])

        EncoderParams = {
            "qw":jnp.stack(qw, axis=0),
            "qb":jnp.stack(qb, axis=0),
            "kw":jnp.stack(kw, axis=0),
            "kb":jnp.stack(kb, axis=0),
            "vw":jnp.stack(vw, axis=0),
            "vb":jnp.stack(vb, axis=0),
            "outw":jnp.stack(outw, axis=0),
            "outb":jnp.stack(outb, axis=0),
            "l1w":jnp.stack(l1w, axis=0),
            "l1b":jnp.stack(l1b, axis=0),
            "l2w":jnp.stack(l2w, axis=0),
            "l2b":jnp.stack(l2b, axis=0),
        }
        return EncoderParams

    def apply_fun(params, q, mask, enc_output=None):
        
        def body(carry, x):
            # note x is actually the params
            if enc_output is not None:
                attn_out, sattn = apply_attn(x, carry, enc_output, enc_output, mask)
            else:
                attn_out, sattn = apply_attn(x, carry, carry, carry, mask)
            return attn_out, sattn
        
        out, sattn = jax.lax.scan(body, q, params)
        return out, sattn

    return init_fun, apply_fun
