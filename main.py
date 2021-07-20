import argparse
from jax.experimental.optimizers import l2_norm
from jax.interpreters.batching import batch
import numpy as np
import jax
from data import spiral, thoracic, abalone
from matplotlib import pyplot as plt 
import matplotlib.cm as cm 
from UAT import UAT
from UAT import binary_cross_entropy

jax.config.update("jax_debug_nans", True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("train model")
    parser.add_argument("dataset", choices=["spiral", "thoracic", "abalone"])
    parser.add_argument("posterior", choices=["laplace", "HSGHMC"])
    args = parser.parse_args()

    # get data
    if args.dataset == "spiral":
        X, y, plot_data = spiral(2048)

        # set up model and parameters
        reg = {"reg":1e-2, "drop":1e-3}
        loss_fun = binary_cross_entropy(l2_reg=1e-2, dropout_reg=1e-3)

        model_kwargs = dict(
            features=2,
            d_model=8,
            embed_hidden_size=32,
            embed_hidden_layers=4,
            embed_activation=jax.nn.relu,
            encoder_layers=1,
            encoder_heads=1,
            enc_activation=jax.nn.relu,
            decoder_layers=5,
            decoder_heads=10,
            dec_activation=jax.nn.relu,
            net_hidden_size=64,
            net_hidden_layers=2,
            net_activation=jax.nn.relu,
            last_layer_size=128,
            out_size=1,
            W_init = jax.nn.initializers.glorot_uniform(),
            b_init = jax.nn.initializers.normal(0.01),
            temp = 0.1,
            reg=reg)
        
        training_kwargs = dict(
            batch_size=32,
            epochs=150,
            lr=1e-2,
        )
        posterior_params = dict(
            name="laplace",
            samples=50
        )
        model = UAT(
            model_kwargs=model_kwargs,
            training_kwargs=training_kwargs,
            loss_fun=loss_fun,
            posterior_params=posterior_params
        )

        model.fit(X, y)
    
    if args.dataset == "thoracic":
        X, y = thoracic()
        # X, X_test, y, y_test = train_test_split(
        #         X, y, test_size=0.33, random_state=42)
        print(X.shape, y.shape)
        
        # set up model and parameters
        reg = {
            "score":1,
            "dist":1,
            "ent":1,
            "nlogpz":1
            }
        embed_nodes = [2,64,64,3]
        mlp_nodes = [3,64,64,3]
        attn_nodes = [3,32,32,1]
        _, apply_fun_nodrop, _, _ = VMN_JAX(
            X.shape[-1], embed_nodes, mlp_nodes, dropout=False, reg=reg)
        init_fun, apply_fun, loss_fn, metric_fn = VMN_JAX(
            X.shape[-1], embed_nodes, mlp_nodes, dropout=True, reg=reg)
        key = jax.random.PRNGKey(1000)
        params = init_fun(key)

        save_path = 'saved_models/thoracic.bin'
        
        batch_size = 32
        epochs = 100
        lr = 1e-2

    if args.dataset == "abalone":
        X, y = abalone()
        print(X.shape, y.shape)
        reg = {
            "score":1,
            "dist":1,
            "ent":1,
            "nlogpz":1
            }
        embed_nodes = [2,64,64,3]
        mlp_nodes = [3,64,64,3]
        attn_nodes = [3,32,32,1]
        _, apply_fun_nodrop, _, _ = VMN_JAX(
            X.shape[-1], embed_nodes, mlp_nodes, dropout=False, reg=reg)
        init_fun, apply_fun, loss_fn, metric_fn = VMN_JAX(
            X.shape[-1], embed_nodes, mlp_nodes, dropout=True, reg=reg)
        key = jax.random.PRNGKey(1000)
        params = init_fun(key)

        batch_size = 128
        epochs = 300
        lr = 1e-2

    if args.dataset == "spiral":
        
        apply_fun_jit = jax.jit(model.apply_fun)
        
        params = model.params
        W_k = params["last_layer"]
        W_MAP = params["last_layer_map"]
        params["last_layer"] = W_MAP
        K_inference = len(W_k)

        samps = 100
        rang = 50
        xs = np.linspace(-rang, rang, samps)
        ys = np.linspace(-rang, rang, samps)    
        xx1 = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)

        rng = jax.random.PRNGKey(100)
        out1 = apply_fun_jit(params, xx1, None)

        xx2 = np.stack([xs, xs]).T
        out2 = apply_fun_jit(params, xx2, None)

        xx3 = np.stack([
            np.hstack([xs, np.array([np.nan]*samps)]),
            np.hstack([np.array([np.nan]*samps), xs])
            ]).T
        out3 = apply_fun_jit(params, xx3, None)

        X, y, plot_data = spiral(50)
        (x_a, x_b) = plot_data
        out4 = apply_fun_jit(params, X, None)
        labels = y == 1

        # sample from posterior across all input space
        logits_sample_full = []
        logits_sample_partial = []
        logits_sample_mapping = []
        print("sampling posterior")
        for k in range(K_inference):
            print(f'sample {k}')
            params["last_layer"] = [W_k[k]]
            logit_s1, attn_s1 = apply_fun_jit(params, xx1, None)
            logit_s2, attn_s2 = apply_fun_jit(params, xx3, None)
            logit_s3, attn_s3 = apply_fun_jit(params, X, None)
            logit_s1 = np.mean(np.stack(logit_s1, axis=0), axis=0)
            logit_s2 = np.mean(np.stack(logit_s2, axis=0), axis=0)
            logit_s3 = np.mean(np.stack(logit_s3, axis=0), axis=0)
            logits_sample_full.append(logit_s1)
            logits_sample_partial.append(logit_s2)
            logits_sample_mapping.append(logit_s3)
        
        probs_sample_full = jax.nn.sigmoid(np.stack(logits_sample_full, axis=0))
        probs_sample_partial = jax.nn.sigmoid(np.stack(logits_sample_partial, axis=0))
        logits_sample_mapping = np.stack(logits_sample_mapping, axis=0)
        probs_mu_full = probs_sample_full.mean(0)
        probs_mu_partial = probs_sample_partial.mean(0)
        logits_mu_mapping = logits_sample_mapping.mean(0)
        confidence_bayes_full = np.abs(probs_mu_full - 0.50).reshape((samps,samps))
        confidence_bayes_partial = np.abs(probs_mu_partial - 0.50).flatten()

        logit1, attn1 = out1
        probs_map = np.mean(np.stack(jax.nn.sigmoid(logit1), axis=0), axis=0)
        logit2, attn2 = out2
        logit2 = np.mean(np.stack(jax.nn.sigmoid(logit2), axis=0), axis=0)
        logit3, attn3 = out3
        probs_map_part = np.mean(np.stack(jax.nn.sigmoid(logit3), axis=0), axis=0)
        logit4, attn4 = out4
        logit4 = np.mean(np.stack(logit4, axis=0), axis=0)

        confidence = np.abs(probs_map - 0.50).reshape((samps,samps))
        confidence_partial = np.abs(probs_map_part - 0.50).flatten()

        # find attn for x1 and x2
        print(jax.nn.sigmoid(params["logits"]))
        print(attn1.shape)
        x1_attn = (attn1[:,0,0]).reshape((samps, samps))
        x2_attn = (attn1[:,0,1]).reshape((samps, samps))
        print(x2_attn.shape)
        # dist = np.abs((embed1[:, 0, :] - embed1[:, 1, :])**2).sum(-1).reshape((samps, samps))
        # [0.1, 0.9] @ [[0.4,0.6], [0.2,0.8]]
        fig = plt.figure(figsize=plt.figaspect(0.5))
        gs = fig.add_gridspec(2, 4)

        ax1 = fig.add_subplot(gs[0, 0])
        font_size=10
        ax1.set_title('Spiral Dataset', fontsize=font_size)
        im = ax1.scatter(x_a[:,0],x_a[:,1], s=1, label="0")
        im = ax1.scatter(x_b[:,0],x_b[:,1], s=1, label="1")
        ax1.legend()

        ax2 = fig.add_subplot(gs[1, 0])
        ax2.set_title('Mapping', fontsize=font_size)
        im = ax2.scatter(logit4[~labels], np.zeros((~labels).sum()), s=1, label="0 MAP")
        im = ax2.scatter(logit4[labels], np.zeros((labels).sum()), s=1, label="1 MAP")
        im = ax2.scatter(logits_mu_mapping[~labels], np.ones((~labels).sum()), s=1, label="0 Bayes")
        im = ax2.scatter(logits_mu_mapping[labels], np.ones((labels).sum()), s=1, label="1 Bayes")
        ax2.legend()

        ax3 = fig.add_subplot(gs[0, 1])
        ax3.set_title('Attention on X1', fontsize=font_size)
        im = ax3.imshow(x1_attn, extent=[-rang, rang, -rang, rang], aspect='auto')

        ax4 = fig.add_subplot(gs[1, 1])
        ax4.set_title('Attention on X2', fontsize=font_size)
        im = ax4.imshow(x2_attn, extent=[-rang, rang, -rang, rang], aspect='auto')

        ax5 = fig.add_subplot(gs[0, 2])
        ax5.set_title('Confidence Bayes', fontsize=font_size)
        im = ax5.imshow(confidence_bayes_full, extent=[-rang, rang, -rang, rang], aspect='auto', vmin=0.0, vmax=0.5)

        ax6 = fig.add_subplot(gs[1, 2])
        ax6.set_title('Confidence MAP', fontsize=font_size)
        im = ax6.imshow(confidence, extent=[-rang, rang, -rang, rang], aspect='auto', vmin=0.0, vmax=0.5)

        ax7 = fig.add_subplot(gs[0, 3])
        ax7.set_title('Confidence X1', fontsize=font_size)
        im = ax7.plot(xs, confidence_bayes_partial[:samps], label = "Bayes")
        im = ax7.plot(xs, confidence_partial[:samps], label = "MAP")
        ax7.legend()

        ax8 = fig.add_subplot(gs[1, 3])
        ax8.set_title('Confidence X2', fontsize=font_size)
        im = ax8.plot(xs, confidence_bayes_partial[samps:], label = "Bayes")
        im = ax8.plot(xs, confidence_partial[samps:], label = "MAP")
        ax8.legend()

        # ax5 = fig.add_subplot(gs[0:, 2:], projection='3d')
        # ax5.set_title('Trivariate means for each group', fontsize=font_size)
        # # im = ax5.scatter(z_c2[~labels,0],z_c2[~labels,1], z_c2[~labels,2], s=0.4, label="0")
        # # im = ax5.scatter(z_c2[labels,0],z_c2[labels,1],z_c2[labels,2], s=0.4, label="1")
        # im = ax5.scatter(xs,embed2[:,0,0], embed2[:,0,0], embed2[:,0,1], s=0.7, alpha=0.6, label="x1")
        # im = ax5.scatter(xs,embed2[:,0,0], embed2[:,1,0], embed2[:,1,1], s=0.7, alpha=0.6, label="x2")
        # im = ax5.scatter(z_mu2[~labels,1,0],z_mu2[~labels,1,1], z_mu2[~labels,1,2], s=0.4, alpha=0.6, label="x2_0")
        # im = ax5.scatter(z_mu2[labels,1,0],z_mu2[labels,1,1],z_mu2[labels,1,2], s=0.4, alpha=0.6, label="x2_1")
        # ax5.legend()
        
        plt.tight_layout()
        plt.show()

    if args.dataset == "thoracic":
        
        # create dummy data to plot latent space
        fvc = np.concatenate([np.linspace(1, 7, 100),np.linspace(1, 7, 100)])
        age = np.concatenate([np.linspace(1, 100, 100),np.linspace(1, 100, 100)])
        bern = np.concatenate([np.zeros(100),np.ones(100)])
        data = np.ones((200,16)) * np.nan
        data[:, 1] = fvc
        data[:, -1] = age
        data[:, 8] = bern  # weakness
        data[:, -6] = bern  # t2dm 
        data[:, -5] = bern # ami
        data[:, -3] = bern # pvd

        rng = jax.random.PRNGKey(100)
        out = apply_fun_nodrop(params, data, rng)

        scores, score_match_loss, z, z_c = out
        z_mu = z
        print(score_match_loss.mean())

        bern_0 = z_mu[:100,...]
        bern_1 = z_mu[100:,...]
        dist_0_w = np.abs((bern_0[:, 1, :] - bern_0[:, 8, :])**2).sum(-1).flatten()
        dist_1_w = np.abs((bern_1[:, 1, :] - bern_1[:, 8, :])**2).sum(-1).flatten()
        
        dist_0_s = np.abs((bern_0[:, 1, :] - bern_0[:, -3, :])**2).sum(-1).flatten()
        dist_1_s = np.abs((bern_1[:, 1, :] - bern_1[:, -3, :])**2).sum(-1).flatten()

        dist_0_aw = np.abs((bern_0[:, -1, :] - bern_0[:, 8, :])**2).sum(-1).flatten()
        dist_1_aw = np.abs((bern_1[:, -1, :] - bern_1[:, 8, :])**2).sum(-1).flatten()
        
        dist_0_as = np.abs((bern_0[:, -1, :] - bern_0[:, -3, :])**2).sum(-1).flatten()
        dist_1_as = np.abs((bern_1[:, -1, :] - bern_1[:, -3, :])**2).sum(-1).flatten()

        fig, axes = plt.subplots(4,1, sharex=False, sharey=False)
        font_size = 10
        axes[0].set_title('FVC vs Distance (Preop Weakness)', fontsize=font_size)
        im = axes[0].plot(fvc[:100], dist_0_w, label='weak +')
        im = axes[0].plot(fvc[:100], dist_1_w, label='weak -')
        axes[0].legend()

        axes[1].set_title('FVC vs Distance (Smoking)', fontsize=font_size)
        im = axes[1].plot(fvc[:100], dist_0_s, label='smok +')
        im = axes[1].plot(fvc[:100], dist_1_s, label='smok -')
        axes[1].legend()

        axes[2].set_title('Age vs Distance (Preop Weakness)', fontsize=font_size)
        im = axes[2].plot(age[:100], dist_0_aw, label='weak +')
        im = axes[2].plot(age[:100], dist_1_aw, label='weak -')
        axes[2].legend()
        axes[3].set_title('Age vs Distance (Smoking)', fontsize=font_size)
        im = axes[3].plot(age[:100], dist_0_as, label='smok +')
        im = axes[3].plot(age[:100], dist_1_as, label='smok -')
        axes[3].legend()
        plt.tight_layout()
        plt.show()

    if args.dataset == "abalone":
        length = np.linspace(0, 1, 100)
        diam = np.linspace(0, 1, 100)    
        X1, X2 = np.meshgrid(length, diam) 
        x1_test = X1.flatten()
        x2_test = X2.flatten()

        data = np.ones((len(x1_test),8)) * np.nan
        data[:, 3] = x1_test
        data[:, 7] = x2_test

        rng = jax.random.PRNGKey(100)
        out = apply_fun_nodrop(params, data, rng)

        scores, score_match_loss, z, z_c  = out
        z_mu = z
        print(score_match_loss.mean())

        dist = np.abs((z_mu[:, 3, :] - z_mu[:, 7, :])**2).sum(-1).reshape((100, 100))

        fig, axes = plt.subplots(1,1, sharex=True, sharey=True)
        font_size = 10
        axes.set_title('Length vs Diameter (Distance)', fontsize=font_size)
        im = axes.imshow(dist, extent=[0, 1, 0, 1], aspect='auto')
        plt.show()