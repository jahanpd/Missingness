import numpy as np
import jax.numpy as jnp
import jax
from jax import random
from jax.nn import (relu, log_softmax, softmax, softplus, sigmoid, elu,
                    leaky_relu, selu, gelu, normalize)
from jax.nn.initializers import glorot_uniform, normal, ones, zeros

def Embed(
    features,
    ndim,
    W_init = glorot_uniform(),
    b_init = zeros,
    ):
    def init_fun(
        rng,
        ):
        k1, k2 = random.split(rng, 2)
        W = W_init(k1, (features, ndim)) * 1e-3
        b = b_init(k2, (ndim,))
        return (W, b)

    def apply_fun(params, x, scan=False):
        W, b = params
        return (x * W) + b

    def rapply_fun(params, x, scan=False):
        W, b = params
        return jnp.divide((x - b), W + 1e-8)

    return init_fun, apply_fun, rapply_fun

def BijectLayer(
    features,
    ndim,
    layers,
    W_init = glorot_uniform(),
    b_init = zeros,
    tactivation = jax.nn.softplus,
    sactivation = jax.nn.tanh,
    ):
    idx = ndim // 2
    def init_fun(
        rng,
        ):
        k1, k2, k3, k4 = random.split(rng, 4)
        sW = W_init(k1, (features, idx, ndim - idx))
        sb = b_init(k2, (features, ndim - idx))
        tW = W_init(k3, (features, idx, ndim - idx))
        tb = b_init(k4, (features, ndim - idx))
        return (sW, sb, tW, tb)

    def apply_fun(params, x, lay, scan=False):
        if scan:
            sW, sb = params["sW"], params["sb"]
            tW, tb = params["tW"], params["tb"]
        else:
            sW, sb, tW, tb = params

        x1 = x[:, :idx]
        x1a = x[:, idx:]
        x2 = x[:, -idx:]
        x2a = x[:, :-idx]

        trans = jnp.where(
            lay % 2 == 0,
            tactivation(
                jnp.einsum('...ij,...ijk->...ik', x1, sW) + sb
            ),
            tactivation(
                jnp.einsum('...ij,...ijk->...ik', x2, sW) + sb
            )
        )
        scale = jnp.where(
            lay % 2 == 0,
            sactivation(
                jnp.einsum('...ij,...ijk->...ik', x1, tW) + tb
            ),
            sactivation(
                jnp.einsum('...ij,...ijk->...ik', x2, tW) + tb
            )
        )
        h = jnp.where(
            lay % 2 == 0,
            (x1a + trans) * scale,
            (x2a + trans) + scale
        )
        return jnp.where(
            lay % 2 == 0,
            jnp.concatenate([x1, h], -1),
            jnp.concatenate([h, x2], -1),
        )

    def rapply_fun(params, x, lay, scan=False):
        if scan:
            sW, sb = params["sW"], params["sb"]
            tW, tb = params["tW"], params["tb"]
        else:
            sW, sb, tW, tb = params

        x1 = x[:, :idx]
        x1a = x[:, idx:]

        x2 = x[:, -idx:]
        x2a = x[:, :-idx]

        trans = jnp.where(
            lay % 2 == 0,
            tactivation(
                jnp.einsum('...ij,...ijk->...ik', x1, sW) + sb
            ),
            tactivation(
                jnp.einsum('...ij,...ijk->...ik', x2, sW) + sb
            )
        )
        scale = jnp.where(
            lay % 2 == 0,
            sactivation(
                jnp.einsum('...ij,...ijk->...ik', x1, tW) + tb
            ),
            sactivation(
                jnp.einsum('...ij,...ijk->...ik', x2, tW) + tb
            )
        )
        h = jnp.where(
            lay % 2 == 0,
            (x1a / (1e-8 + scale)) - trans,
            (x2a / (1e-8 + scale)) - trans
        )
        return jnp.where(
            lay % 2 == 0,
            jnp.concatenate([x1, h], -1),
            jnp.concatenate([h, x2], -1),
        )

    return init_fun, apply_fun, rapply_fun

def Bijector(
    features,
    ndim,
    layers,
    W_init = glorot_uniform(),
    b_init = zeros,
    tactivation = jax.nn.softplus,
    sactivation = jax.nn.tanh,
    ):

    init_bij, bijf, bijr = BijectLayer(
            features, ndim, layers, W_init, b_init, tactivation, sactivation)

    def init_fun(
        rng,
    ):

        sW, sb, tW, tb = [], [], [], []
        for l in range(layers):
            rng, key = random.split(rng)
            sW_, sb_, tW_, tb_ = init_bij(key)
            sW.append(sW_)
            sb.append(sb_)
            tW.append(tW_)
            tb.append(tb_)

        params = {
            "sW":jnp.stack(sW, axis=0),
            "sb":jnp.stack(sb, axis=0),
            "tW":jnp.stack(tW, axis=0),
            "tb":jnp.stack(tb, axis=0),
        }

        return params

    def apply_fun(params, x, scan=False):

        def body(carry, x):
            state, lay = carry
            # note x is actually the params
            temp = bijf(x, state, lay, scan=True)
            lay = lay + 1
            return (temp, lay), None

        out, _ = jax.lax.scan(body, (x, 0), params)
        return out[0]

    def rapply_fun(params, x, scan=False):

        def body(carry, x):
            state, lay = carry
            # note x is actually the params
            temp = bijr(x, state, lay, scan=True)
            lay = lay - 1
            return (temp, lay), None

        out, _ = jax.lax.scan(body, (x, layers-1), params, reverse=True)
        return out[0]

    return init_fun, apply_fun, rapply_fun

def DenseGeneral(
    features,
    in_dim,
    out_dim,
    W_init = glorot_uniform(),
    b_init = zeros,
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
    W_init = glorot_uniform(),
    b_init = zeros,
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
    def apply_fun(params, x, scan = False, mask = 1.0):
        if bias:
            if scan:
                W, b = params["hw"], params["hb"]
            else:
                W, b = params
            return jnp.einsum('...i,...ij->...j', x, W*mask) + b
        else:
            if scan:
                W = params["hw"]
            else:
                W = params
            return jnp.einsum('...i,...ij->...j', x, W*mask)
    return init_fun, apply_fun

def DenseRoll(
    features,
    in_dim,
    out_dim,
    bias = True,
    W_init = glorot_uniform(),
    b_init = zeros,
    ):
    def init_fun(
        rng
        ):
        k1, k2, k3, k4 = random.split(rng, 4)
        out_shape = (out_dim)
        W1 = W_init(k1, (in_dim, out_dim))
        W2 = W_init(k2, (features, out_dim))
        if bias:
            b1 = b_init(k3, (out_dim,))
            b2 = b_init(k4, (out_dim,))
            return (W1, W2, b1, b2)
        else:
            return (W1, W2)
    def apply_fun(params, x, res, scan = False, mask = 1.0):
        if bias:
            if scan:
                W1, W2, b1, b2 = params["hw1"], params["hb1"], params["hw2"], params["hb2"]
            else:
                W1, W2, b1, b2 = params
            return (jnp.einsum('...i,...ij->...j', x, W1) + b1 +
                    jnp.einsum('...i,...ij->...j', res*mask, W2) + b2)
        else:
            if scan:
                W1, W2 = params["hw1"], params["hw2"]
            else:
                W1, W2 = params
            return (jnp.einsum('...i,...ij->...j', x, W1) +
                    jnp.einsum('...i,...ij->...j', res*mask, W2))
    return init_fun, apply_fun

def ResNet(
    features,
    hidden_dim,
    out_dim,
    num_hidden = 1,
    W_init = glorot_uniform(),
    b_init = zeros,
    activation=relu
    ):
    init_l1, layer_1 = Dense(
        features, hidden_dim, W_init=W_init, b_init=b_init)
    init_hidden, layer_h = DenseRoll(
        features, hidden_dim, hidden_dim, W_init=W_init, b_init=b_init)
    init_out, layer_out = Dense(
        hidden_dim, out_dim, W_init=W_init, b_init=b_init)

    def init_fun(rng):
        params = {}
        rng, key = random.split(rng)
        params["l1"] = init_l1(key)

        hw1, hw2, hb1, hb2 = [], [], [], []
        for _ in range(num_hidden):
            rng, key = random.split(rng)
            W1, W2, b1, b2 = init_hidden(key)
            hw1.append(W1)
            hw2.append(W2)
            hb1.append(b1)
            hb2.append(b2)
        
        params["hidden"] = {
            "hw1":jnp.stack(hw1, axis=0),
            "hw2":jnp.stack(hw2, axis=0),
            "hb1":jnp.stack(hb1, axis=0),
            "hb2":jnp.stack(hb2, axis=0)
        }

        rng, key = random.split(rng)
        params["out"] = init_out(key)

        return params
    
    def apply_fun(params, inputs, mask, rng=None):
        h = activation(layer_1(params["l1"], inputs, mask=mask))

        def body(carry, x):
            # note x is actually the params
            temp = activation(layer_h(x, carry, inputs, mask=mask, scan=True))
            return temp, None
        
        h, _ = jax.lax.scan(body, h, params["hidden"])

        out = layer_out(params["out"], h)
        return out
    
    return init_fun, apply_fun

def NeuralNetGeneral(
    features,
    in_dim,
    hidden_dim,
    out_dim,
    num_hidden = 1,
    W_init = glorot_uniform(),
    b_init = zeros,
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
    W_init = glorot_uniform(),
    b_init = zeros,
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
            rng, key = random.split(rng, 2)
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

def ConcDropNeuralNet(
    in_dim,
    hidden_dim,
    out_dim,
    num_hidden = 1,
    W_init = glorot_uniform(),
    b_init = zeros,
    activation=relu,
    temp = 0.1,
    eps = 1e-7
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
        params["l1_logit"] = jnp.zeros((in_dim,hidden_dim))

        hw, hb, lgts = [], [], []
        for _ in range(num_hidden):
            rng, key = random.split(rng, 2)
            W, b = init_hidden(key)
            hw.append(W)
            hb.append(b)
            lgts.append(jnp.zeros((hidden_dim,hidden_dim)))

        params["hidden"] = {
            "hw":jnp.stack(hw, axis=0),
            "hb":jnp.stack(hb, axis=0),
            "lgts":jnp.stack(lgts, axis=0)
        }

        rng, key = random.split(rng)
        params["out"] = init_out(key)

        return params

    def masksamp(lgts, rng):
        probs = jax.nn.sigmoid(lgts)
        unif_noise = random.uniform(rng, lgts.shape)
        drop_prob = (jnp.log(probs + eps)
                        - jnp.log(1.0 - probs + eps)
                        + jnp.log(unif_noise + eps)
                        - jnp.log(1.0 - unif_noise + eps)
                        )
        drop_prob = jax.nn.sigmoid(drop_prob / temp)
        # this line is 1.0 - drop_prob if you want to select more important variables
        # during training
        random_arr = 1. - drop_prob  # (1 if keeping 0 if removing)
        return random_arr

    def apply_fun(params, inputs, mask1, rng):

        h = activation(layer_1(params["l1"], inputs, mask=mask1))

        def body(carry, x):
            # note x is actually the params
            mask = masksamp(x["lgts"], x["rng"])
            temp = activation(layer_h(x, carry, mask=mask, scan=True))
            return temp, None

        rng = random.split(rng, num_hidden)
        params["hidden"]["rng"] = rng

        h, _ = jax.lax.scan(body, h, params["hidden"])

        out = layer_out(params["out"], h)
        return out

    return init_fun, apply_fun

def LayerNorm(epsilon=1e-5, center=True, scale=True, beta_init=zeros, gamma_init=ones):
    def init_fun(rng, elements): 
        k1, k2 = random.split(rng, 2)
        beta = beta_init(k1, (elements,))
        gamma = gamma_init(k2, (elements,))
        return (beta, gamma)
    def apply_fun(params, x):
        beta, gamma = params
        mean = jnp.mean(x, axis=-1, keepdims=True)
        mean2 = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        var = mean2 - jnp.square(mean)
        mul = gamma * jax.lax.rsqrt(var + epsilon)
        return (x - mean) * mul + beta
    return init_fun, apply_fun

def AttentionLayer(
    heads,
    dims,
    dff = None,
    W_init = glorot_uniform(),
    b_init = zeros,
    activation = softplus
    ):
    if dff is None:
        dff = dims * 4
    init_q, q_map = Dense(dims, dims*heads, W_init=W_init, b_init=b_init)
    init_k, k_map = Dense(dims, dims*heads, W_init=W_init, b_init=b_init)
    init_v, v_map = Dense(dims, dims*heads, W_init=W_init, b_init=b_init)
    init_out, out = Dense(dims*heads, dims, W_init=W_init, b_init=b_init)
    # feedforward network
    init_l1, l1 = Dense(dims, dff, W_init=W_init, b_init=b_init)
    init_l2, l2 = Dense(dff, dims, W_init=W_init, b_init=b_init)
    # layer norm
    init_ln1, layernorm1 = LayerNorm()
    init_ln2, layernorm2 = LayerNorm()

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
        rng, key = random.split(rng)
        params["ln1"] = init_ln1(key, dims)
        rng, key = random.split(rng)
        params["ln2"] = init_ln2(key, dims)
        return params
    
    def apply_fun(params, q, k, v, mask, vmap=True):
        # note params input is from the AttentionBlock init_params construction, not the above
        # q (features, embedding)
        # q = layernorm1((params["ln1b_block"], params["ln1g_block"]), q)
        q_ = jnp.transpose(q_map((params["qw_block"],params["qb_block"]), q).reshape((-1, heads, dims)), (1,0,2))
        k_ = jnp.transpose(k_map((params["kw_block"],params["kb_block"]), k).reshape((-1, heads, dims)), (1,2,0))
        v_ = jnp.transpose(v_map((params["vw_block"],params["vb_block"]), v).reshape((-1, heads, dims)), (1,0,2))

        # scaled dot product attention
        qk = jnp.matmul(q_,k_)
        scaled_attention_logits = jax.nn.softplus(qk / jnp.sqrt(dims)) * mask
        attention_weights = scaled_attention_logits / (scaled_attention_logits.sum(-1, keepdims=True) + 1e-6)
        # attention_weights = jax.nn.softmax(qk + ((mask - 1.0)*1e8 ))
        scaled_attention = jnp.transpose(jnp.matmul(attention_weights, v_), (1,0,2)).reshape((-1, dims*heads))
        x = out((params["outw_block"],params["outb_block"]), scaled_attention)

        # residual connection
        x = x + q
        x = layernorm2((params["ln2b_block"], params["ln2g_block"]), x)
        # feedforward network
        x = activation(l1((params["l1w_block"],params["l1b_block"]), x))
        x = l2((params["l2w_block"],params["l2b_block"]), x)

        return x, attention_weights
    
    return init_fun, apply_fun

def AttentionBlock(
        num_layers,
        dims,
        heads,
        dff=None,
        W_init = glorot_uniform(),
        b_init = zeros,
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
        ln1g, ln1b = [], []
        ln2g, ln2b = [], []

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
            ln1b.append(params["ln1"][0])
            ln1g.append(params["ln1"][1])
            ln2b.append(params["ln2"][0])
            ln2g.append(params["ln2"][1])

        BlockParams = {
            "qw_block":jnp.stack(qw, axis=0),
            "qb_block":jnp.stack(qb, axis=0),
            "kw_block":jnp.stack(kw, axis=0),
            "kb_block":jnp.stack(kb, axis=0),
            "vw_block":jnp.stack(vw, axis=0),
            "vb_block":jnp.stack(vb, axis=0),
            "outw_block":jnp.stack(outw, axis=0),
            "outb_block":jnp.stack(outb, axis=0),
            "l1w_block":jnp.stack(l1w, axis=0),
            "l1b_block":jnp.stack(l1b, axis=0),
            "l2w_block":jnp.stack(l2w, axis=0),
            "l2b_block":jnp.stack(l2b, axis=0),
            "ln1b_block":jnp.stack(ln1b, axis=0),
            "ln1g_block":jnp.stack(ln1g, axis=0),
            "ln2b_block":jnp.stack(ln2b, axis=0),
            "ln2g_block":jnp.stack(ln2g, axis=0),
        }
        return BlockParams

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

def AttentionLayer2(
    heads,
    dims,
    dff = None,
    W_init = glorot_uniform(),
    b_init = zeros,
    activation = softplus
    ):
    if dff is None:
        dff = dims * 4
    init_q, q_map = Dense(dims, dims*heads, W_init=W_init, b_init=b_init)
    init_k, k_map = Dense(dims, dims*heads, W_init=W_init, b_init=b_init)
    init_v, v_map = Dense(dims, dims*heads, W_init=W_init, b_init=b_init)
    init_out, out = Dense(dims*heads, dims, W_init=W_init, b_init=b_init)
    # feedforward network
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

    def apply_fun(params, q, v, state, mask, vmap=True):
        # note params input is from the AttentionBlock init_params construction, not the above
        # q (features, embedding)
        # q = layernorm1((params["ln1b_block"], params["ln1g_block"]), q)
        q_ = jnp.transpose(q_map((params["qw_block"],params["qb_block"]), q).reshape((-1, heads, dims)), (1,0,2))
        k_ = jnp.transpose(k_map((params["kw_block"],params["kb_block"]), q).reshape((-1, heads, dims)), (1,2,0))
        v_ = jnp.transpose(v_map((params["vw_block"],params["vb_block"]), v).reshape((-1, heads, dims)), (1,0,2))

        # scaled dot product attention
        qk = jnp.matmul(q_,k_)  # (h, f, f)
        qk = jnp.transpose(jnp.sum(qk * mask, 2, keepdims=True), (0,2,1))  # (h, 1, f)
        scaled_attention_logits = jax.nn.softplus(qk / jnp.sqrt(dims)) * mask
        attention_weights = scaled_attention_logits / (scaled_attention_logits.sum(-1, keepdims=True) + 1e-6)
        # attention_weights = jax.nn.softmax(qk + ((mask - 1.0)*1e8 ))
        scaled_attention = jnp.transpose(
            jnp.matmul(attention_weights, v_), (1,0,2)).reshape((-1, dims*heads))  # (1, heads*dims)
        x = out((params["outw_block"],params["outb_block"]), scaled_attention)  # (1, dims)

        # residual connection
        x = x + state
        # feedforward network
        x = activation(l1((params["l1w_block"],params["l1b_block"]), x))
        x = l2((params["l2w_block"],params["l2b_block"]), x)

        return x, attention_weights

    return init_fun, apply_fun

def AttentionBlock2(
        num_layers,
        dims,
        heads,
        dff=None,
        W_init = glorot_uniform(),
        b_init = zeros,
        activation = softplus
    ):

    init_attn, apply_attn = AttentionLayer2(
        heads, dims, dff=dff, W_init=W_init, b_init=b_init, activation=activation
    )

    def init_fun(rng):

        qw, qb = [], []
        kw, kb = [], []
        vw, vb = [], []
        outw, outb = [], []
        l1w, l1b = [], []
        l2w, l2b = [], []
        ln1g, ln1b = [], []
        ln2g, ln2b = [], []

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

        BlockParams = {
            "qw_block":jnp.stack(qw, axis=0),
            "qb_block":jnp.stack(qb, axis=0),
            "kw_block":jnp.stack(kw, axis=0),
            "kb_block":jnp.stack(kb, axis=0),
            "vw_block":jnp.stack(vw, axis=0),
            "vb_block":jnp.stack(vb, axis=0),
            "outw_block":jnp.stack(outw, axis=0),
            "outb_block":jnp.stack(outb, axis=0),
            "l1w_block":jnp.stack(l1w, axis=0),
            "l1b_block":jnp.stack(l1b, axis=0),
            "l2w_block":jnp.stack(l2w, axis=0),
            "l2b_block":jnp.stack(l2b, axis=0),
        }
        return BlockParams

    def apply_fun(params, q, v, mask):

        def body(carry, x):
            # note x is actually the params
            attn_out, sattn = apply_attn(x, q, v, carry, mask)
            return attn_out, sattn

        out, sattn = jax.lax.scan(body, jnp.zeros((1,dims)), params)
        return out, sattn

    return init_fun, apply_fun
