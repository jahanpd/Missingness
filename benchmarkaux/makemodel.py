from LSAM import LSAM, create_early_stopping
from LSAM import binary_cross_entropy, cross_entropy, mse, brier, dual, dual2, cross_entropy_conc
from optax import linear_onecycle_schedule, join_schedules, piecewise_constant_schedule, linear_schedule
import jax
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
            embedding_size=32,
            embedding_layers=2,
            encoder_heads=5,
            encoder_layers=5,
            decoder_heads=5,
            decoder_layers=5,
            net_size=32,
            net_layers=2,
            max_steps=1e-4,
            learning_rate=1e-3,
            early_stop=0.5,
            batch_size=32,
            noise_std=0.1,
            dropreg=1e-5,
            msereg=1e-5,
            weight_decay=1e-5,
            l2=1e-5,
            optimizer="adam"
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
        batch_sizes = [64, 128]
        distances = [(np.abs(max_steps - ((rows / b) * 100)), b) for b in batch_sizes]
        batch_size_base2 = min(distances, key=lambda x: x[0])[1]
        # batch_size_base2 = batch_size
        steps_per_epoch = max(rows // batch_size_base2, 1)
        epochs = max(50, max_steps // steps_per_epoch)
        decay = piecewise_constant_schedule(
            learning_rate,
            boundaries_and_scales={
                int(0.5 * epochs * steps_per_epoch):1.0,
                int(0.8 * epochs * steps_per_epoch):1.0,
            })
        
        freq = 5
        model_kwargs_uat = dict(
                features=features,
                d_model=d_model,
                embed_hidden_size=embedding_size,
                embed_hidden_layers=embedding_layers,
                embed_activation=jax.nn.selu,
                encoder_layers=encoder_layers,
                encoder_heads=encoder_heads,
                enc_activation=jax.nn.gelu,
                decoder_layers=decoder_layers,
                decoder_heads=decoder_heads,
                dec_activation=jax.nn.gelu,
                net_hidden_size=net_size,
                net_hidden_layers=net_layers,
                net_activation=jax.nn.selu,
                last_layer_size=d_model,
                out_size=classes,
                W_init = jax.nn.initializers.he_normal(),
                b_init = jax.nn.initializers.zeros,
                noise_std = noise_std
                )
        if steps_per_epoch > max_steps:
            stop_steps_ = steps_per_epoch // 6
            start_steps = int(early_stop*steps_per_epoch) # wait at least X epochs before early stopping
        else:
            stop_steps_ = steps_per_epoch * (epochs // 6) / min(steps_per_epoch, freq)
            start_steps = int(early_stop*epochs*steps_per_epoch) # wait at least X epochs before early stopping

        optim_kwargs=dict(
            b1=0.9, b2=0.999,
            eps=1e-9,
            weight_decay=weight_decay,
        )
        early_stopping = create_early_stopping(start_steps, stop_steps_, metric_name="loss", tol=1e-8)
        early_stop_bool = True if early_stop >= 0 else False 

        training_kwargs_uat = dict(
                    optim=optimizer,
                    frequency=min(steps_per_epoch, freq),
                    batch_size=batch_size_base2,
                    lr=decay,
                    epochs=int(epochs),
                    early_stopping=early_stopping,
                    optim_kwargs=optim_kwargs,
                    early_stop=early_stop_bool,
                    steps_til_samp=-1
                )
        if task == "Supervised Classification":
            loss_fun = cross_entropy(classes, l2_reg=l2, dropout_reg=dropreg)
        elif task == "Supervised Regression":
            loss_fun = mse(l2_reg=0.0, dropout_reg=dropreg)
        training_kwargs_uat["X_test"] = X_valid
        training_kwargs_uat["y_test"] = y_valid
        model = LSAM(
            model_kwargs=model_kwargs_uat,
            training_kwargs=training_kwargs_uat,
            loss_fun=loss_fun,
            rng_key=key,
            classes=classes,
            unsupervised_pretraining=None
            #unsupervised_pretraining=dict(
            #    lr=1e-4,
            #    batch_size=batch_size_base2,
            #    cut_off=100
            #    )
            )
        return model, batch_size_base2, loss_fun
    return make_model
