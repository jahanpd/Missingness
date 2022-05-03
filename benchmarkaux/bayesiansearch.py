# BAYESIAN HYPERPARAMETER  SEARCH
            # will search if cannot load params from file
            print(row[1], row[2], row[3])
            key = rng.integers(9999)
            X, y, classes, cat_bin = data.prepOpenML(row[0], row[1])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=key)
            key = rng.integers(9999)
            X_train, X_test, X_valid, y_train, y_test, y_valid, diagnostics = data.openml_ds(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    row[1],
                    cat_bin=cat_bin,
                    classes=classes,
                    missing=missing,
                    imputation=None,  # one of none, simple, iterative, miceforest
                    train_complete=False,
                    test_complete=True,
                    split=0.2,
                    rng_key=key,
                    prop=0.7,
                    corrupt=args.corrupt,
                    cols_miss=int(X.shape[1] * 0.8)
                )
            key = rng.integers(9999)
            if row[1] == "Supervised Classification":
                objective = 'softmax'
                X_train, y_train = ros.fit_resample(X_train, y_train)
            else:
                objective = 'regression'
                resample=False
            ## set up transformer model hp search
            path = "results/openml/hyperparams"
            filenames = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
            subset = [f for f in filenames if row[2] in f]

            # attempt to get params from file
            try:
                trans_subset = [f for f in subset if 'trans' in f]
                with (open(trans_subset[0], "rb")) as handle:
                    trans_results = pickle.load(handle)
                loaded_hps_trans = True
            except Exception as e:
                loaded_hps_trans = False
            try:
                gbm_subset = [f for f in subset if 'gbm' in f]
                with (open(gbm_subset[0], "rb")) as handle:
                    gbm_results = pickle.load(handle)
                loaded_hps_gbm = True
            except Exception as e:
                loaded_hps_gbm = False

            if not loaded_hps_trans:
                # implement bayesian hyperparameter optimization with sequential domain reduction
                make_key = rng.integers(9999)
                make_model = create_make_model(X_train.shape[1], X_train.shape[0], row[1], make_key)
                # find LR range for cyclical training / super convergence
                search_steps = 4e3
                # model, batch_size_base2, loss_fun = make_model(128, 128, 12, X_valid, y_valid, search_steps)
                # model.fit(X_train, y_train)
                # loss_hx = [h["test_current"] for h in model._history]
                # lr_hx = [h["lr"] for h in model._history]
                # lr_max = lr_hx[int(np.argmin(loss_hx) * 0.8)]

                def black_box(
                        lr_max=np.log(5e-3),
                        d_model=32,
                        reg=6,
                        embed_depth=5,
                        depth=5,
                        batch_size=6,
                        b2=0.99
                ):
                    model, batch_size_base2, loss_fun = make_model(
                        X_valid, y_valid, classes=classes,
                        reg=reg, lr_max=lr_max, embed_depth=embed_depth,
                        depth=depth, batch_size=batch_size, b2=b2,
                        early_stop=True, d_model=d_model
                        )
                    model.fit(X_train, y_train)
                    # break test into 'batches' to avoid OOM errors
                    test_mod = X_test.shape[0] % batch_size_base2 if batch_size_base2 < X_test.shape[0] else 0
                    test_rows = np.arange(X_test.shape[0] - test_mod)
                    test_batches = np.split(test_rows,
                                np.maximum(1, X_test.shape[0] // batch_size_base2))

                    loss_loop = 0
                    acc_loop = 0
                    pbar1 = tqdm(total=len(test_batches), position=0, leave=False)
                    @jax.jit
                    def loss_calc(params, x_batch, y_batch, rng):
                        out = model.apply_fun(params, x_batch, rng, False)
                        loss, _ = loss_fun(params, out, y_batch)
                        class_o = np.argmax(jnp.squeeze(out[0]), axis=1)
                        correct_o = class_o == y_batch
                        acc = np.sum(correct_o) / y_batch.shape[0]
                        return loss, acc
                    for tbatch in test_batches:
                        key_ = jnp.ones((X_test[np.array(tbatch), ...].shape[0], 2))
                        loss, acc = loss_calc(model.params, X_test[np.array(tbatch), ...], y_test[np.array(tbatch)], key_)
                        loss_loop += loss
                        acc_loop += acc
                        pbar1.update(1)
                    # make nan loss high, and average metrics over test batches
                    if np.isnan(loss_loop) or np.isinf(loss_loop):
                        loss_loop = 999999
                    unique, counts = np.unique(y_test, return_counts=True)
                    baseline = np.sum(unique[np.argmax(counts)] == y_test) / y_test.shape[0]
                    acc = acc_loop / len(test_batches)
                    diff = np.abs(acc - 0.5) - np.abs(baseline - 0.5)
                    print(
                        "test loss: {}, acc: {}, diff {}".format(
                            loss_loop/len(test_batches),
                            acc_loop/len(test_batches),
                            diff)
                    )
                    # return - loss_loop / len(test_batches)
                    return loss_loop / (loss_loop / len(test_batches))

                pbounds={
                        "lr_max":(np.log(5e-5), np.log(1e-3)),
                        "d_model":(1,8),
                        "reg":(2,10),
                        }

                # bounds_transformer = SequentialDomainReductionTransformer()
                key = rng.integers(9999)
                mutating_optimizer = BayesianOptimization(
                    f=black_box,
                    pbounds=pbounds,
                    verbose=0,
                    random_state=int(key),
                    # bounds_transformer=bounds_transformer
                )
                mutating_optimizer.probe(params={
                    "reg":8, "lr_max":np.log(1e-3), "d_model":5
                })
                kappa = 10  # parameter to control exploitation vs exploration. higher = explore
                xi =1e-1
                mutating_optimizer.maximize(init_points=5, n_iter=args.iters, acq="ei", xi=xi, kappa=kappa)
                print(mutating_optimizer.res)
                print(mutating_optimizer.max)
                trans_results = {"max": mutating_optimizer.max, "all":mutating_optimizer.res, "key": make_key}
                
                with open('results/openml/hyperparams/{},{},trans_hyperparams.pickle'.format(row[2], missing), 'wb') as handle:
                    pickle.dump(trans_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
            if not loaded_hps_gbm:
                def black_box_gbm(max_depth, learning_rate, max_bin):
                    param = {'objective':objective, 'num_class':classes}
                    param['max_depth'] = int(max_depth)
                    param['num_leaves'] = int(0.8 * (2**max_depth))
                    param['learning_rate'] = np.exp(learning_rate)
                    param['max_bin']=int(max_bin)
                    param['verbosity']=-1
                    dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=list(np.argwhere(cat_bin == 1)))
                    history = lgb.cv(
                        params=param,
                        train_set=dtrain,
                        num_boost_round=1000,
                        nfold=5,
                        early_stopping_rounds=50,
                        stratified=False,
                        categorical_feature=list(np.argwhere(cat_bin == 1))
                        )
                    loss = np.mean(history[list(history.keys())[0]])
                    print(loss)
                    return - loss
                pbounds_gbm={
                        "max_depth":(3,12),
                        "learning_rate":(np.log(0.001), np.log(1)),
                        "max_bin":(10, 100)
                        }
            
                key = rng.integers(9999)
                mutating_optimizer_gbm = BayesianOptimization(
                    f=black_box_gbm,
                    pbounds=pbounds_gbm,
                    verbose=0,
                    random_state=int(key),
                )
                kappa = 10  # parameter to control exploitation vs exploration. higher = explore
                xi =1e-1
                mutating_optimizer_gbm.maximize(init_points=5, n_iter=args.iters, acq="ei", xi=xi, kappa=kappa)
                print(mutating_optimizer_gbm.res)
                print(mutating_optimizer_gbm.max)
                best_params_gbm = mutating_optimizer_gbm.max
                best_params_gbm["params"]["objective"]=objective
                best_params_gbm["params"]["num_class"]=classes
                best_params_gbm["params"]['num_leaves'] = int(0.8 * (2**best_params_gbm["params"]["max_depth"]))
                gbm_results = {"max": best_params_gbm, "all":mutating_optimizer_gbm.res} 

                with open('results/openml/hyperparams/{},{},gbm_hyperparams.pickle'.format(row[2], missing),'wb') as handle:
                    pickle.dump(gbm_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
     

