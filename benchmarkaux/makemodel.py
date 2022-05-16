from UAT import UAT, create_early_stopping
from UAT import binary_cross_entropy, cross_entropy, mse, brier, dual
from UAT.aux import oversampled_Kfold
from UAT.training.lr_schedule import attention_lr, linear_increase
from optax import linear_onecycle_schedule, join_schedules, piecewise_constant_schedule, linear_schedule
import jax
import jax.numpy as jnp
import numpy as np

def create_make_model(features, rows, task, key):
    """
        Create a function to make a transformer based model with args in closure.
        Args:
            features: int, number of features
            rows: int, number of rows in training dataset
            task: str, one of 'Supervised Classification' or 'Supervised Regression'
            key: int, an rng key
        Returns:
            Callable to create model
    """
    def make_model(
            X_valid,
            y_valid,
            classes,
            d_model=32,
            max_steps=1e4,
            lr_max=None,
            batch_size=32,
            depth=4,  #bij
            nndepth=2,
            nnwidth=4,
            early_stop=True,
            reg=1e-5,
            msereg=1e-5,
            dropreg=1e-5,
            start_es=0.5
        ):
        """
        Args:
            X_valid: ndarray, for early stopping
            y_valid: ndarray,
            classes: int, number of possible classes in outcome,
            batch_size: int, number of samples in each batch,
            max_steps: int, total number of iterations to train for
            lr_max: float, maximum learning rate
            embed_depth: int, depth of the embedding neural networks,
            depth: int, depth of the decoder in the transformer,
            early_stop: bool, whether to do early stopping,
            b2: float, interval (0, 1) hyperparameter for adam/adabelief,
            reg: int, exponent in regularization (1e-reg)
        Returns:
            model: Object
            batch_size_base2: int
            loss_fun: Callable
        """
        # use a batch size to get around 10-20 iterations per epoch
        # this means you cycle over the datasets a similar number of times
        # regardless of dataset size.
        # batch_size_base2 = min(2 ** int(np.round(np.log2(rows/20))), batch_size)
        batch_size_base2 = batch_size
        steps_per_epoch = max(rows // batch_size_base2, 1)
        epochs = max_steps // steps_per_epoch
        nnwidth = d_model
        decay = piecewise_constant_schedule(
            lr_max,
            # 1e-3,
            boundaries_and_scales={
                int(0.5 * epochs * steps_per_epoch):0.1,
                int(0.8 * epochs * steps_per_epoch):0.1,
            })

        while epochs < 150:
            if batch_size_base2 > 500:
                break
            batch_size_base2 *= 2
            steps_per_epoch = max(rows // batch_size_base2, 1)
            epochs = max_steps // steps_per_epoch
            print(epochs)

        freq = 5
        print("lr: {}, depth: {}, d_model: {}, width: {}".format(
            lr_max, int(depth), int(d_model), nnwidth))
        model_kwargs_uat = dict(
                features=features,
                d_model=d_model,
                embed_hidden_size=int(d_model),
                embed_hidden_layers=int(2),  # bij
                embed_activation=jax.nn.gelu,
                encoder_layers=int(depth),  # attn
                encoder_heads=5,
                enc_activation=jax.nn.gelu,
                decoder_layers=int(depth),  # attn
                decoder_heads=5,
                dec_activation=jax.nn.gelu,
                net_hidden_size=int(d_model),  # output
                net_hidden_layers=int(nndepth),  # output or mixture attn
                net_activation=jax.nn.gelu,
                last_layer_size=d_model // 2,
                out_size=classes,
                W_init = jax.nn.initializers.he_normal(),
                b_init = jax.nn.initializers.zeros,
                )
        epochs = int(max_steps // steps_per_epoch)
        start_steps = int(0.8*epochs*steps_per_epoch) # wait at least 80 epochs before early stopping
        stop_steps_ = steps_per_epoch * (epochs // 8) / min(steps_per_epoch, freq)

        optim_kwargs=dict(
            b1=0.9, b2=0.99,
            eps=1e-9,
            weight_decay=reg,
        )
        early_stopping = create_early_stopping(start_steps, stop_steps_, metric_name="loss", tol=1e-8)
        training_kwargs_uat = dict(
                    optim="adam",
                    frequency=min(steps_per_epoch, freq),
                    batch_size=batch_size_base2,
                    lr=decay,
                    #lr=1e-4,
                    epochs=epochs,
                    early_stopping=early_stopping,
                    optim_kwargs=optim_kwargs,
                    early_stop=early_stop,
                    steps_til_samp=-1
                )
        if task == "Supervised Classification":
            # loss_fun = cross_entropy(classes, l2_reg=0, dropout_reg=1e-5)
            loss_fun = dual(classes, dropout_reg=dropreg, msereg=msereg)
            # loss_fun = brier(l2_reg=0.0, dropout_reg=1e-7)
        elif task == "Supervised Regression":
            loss_fun = mse(l2_reg=0.0, dropout_reg=5e-1)
        training_kwargs_uat["X_test"] = X_valid
        training_kwargs_uat["y_test"] = y_valid
        model = UAT(
            model_kwargs=model_kwargs_uat,
            training_kwargs=training_kwargs_uat,
            loss_fun=loss_fun,
            rng_key=key,
            classes=classes,
            )
        return model, batch_size_base2, loss_fun
    return make_model
