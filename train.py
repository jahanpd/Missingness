import argparse
import wandb
import numpy as np
import pandas as pd
from benchmarkaux.openmlrun import run

# build the argparse 
parser = argparse.ArgumentParser()
# dataset characteristics
parser.add_argument("--dataset", type=int, default=0,
                    help="An integer selecting for one of the datasets")    
parser.add_argument("--corrupt", action='store_true')
parser.add_argument("--missing", choices=["None", "MCAR", "MAR", "MNAR"], default="None",
                    help="Type of missingness to corrupt the dataset with. Only relevant if corrupt flag is given.")
parser.add_argument("--imputation", choices=["None", "simple", "iterative", "miceforest"], default="None",
                    help="Whether to impute data and by what mechanism.")
parser.add_argument("--p_missing_per_col", default=0.4, type=float,
                    help="Percentage of missing data in each column. Only relevant if corrupt flag is given")
parser.add_argument("--p_cols_missing", default=0.4, type=float,
                    help="Percentage of missing colums. Only relevant if corrupt flag is given.")
parser.add_argument("--min_p_missing", default=0.2, type=float,
                    help="Minimum percentage of missing data in dataset. Only relevant if corrupt flag is ABSENT.")

# which model(s) to train
parser.add_argument("--model", choices=["LSAM", "LightGBM", "Both"], default="Both")

# LSAM model hyperparameters
parser.add_argument("--d_model", default=42, type=int)
parser.add_argument("--depth", default=-1, type=int)
parser.add_argument("--embedding_size", default=32, type=int)
parser.add_argument("--embedding_layers", default=2, type=int)
parser.add_argument("--encoder_heads", default=3, type=int)
parser.add_argument("--encoder_layers", default=2, type=int)
parser.add_argument("--decoder_heads", default=3, type=int)
parser.add_argument("--decoder_layers", default=3, type=int)
parser.add_argument("--net_size", default=32, type=int)
parser.add_argument("--net_layers", default=1, type=int)

# LSAM training hyperparameters
parser.add_argument("--max_steps", default=5e3, type=int)
parser.add_argument("--learning_rate", default=5e-3, type=float)
parser.add_argument("--early_stopping", default=0.0, type=float, help="Percentage of epochs when to start ES")
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--noise_std", default=0.0, type=float, help="Injected noise std in the latent space")
parser.add_argument("--drop_reg", default=1e-5, type=float, help="Dropout regularization")
parser.add_argument("--weight_decay", default=1e-5, type=float, help="Weight decay")
parser.add_argument("--l2", default=1e-3, type=float, help="Last layer weight regularization")
parser.add_argument("--optimizer", default="adam", choices=["adam", "adabelief", "sgd"])

# LightGBM model hyperparameters
parser.add_argument("--num_leaves", default=31, type=int, help="Must be smaller that max_depth")
parser.add_argument("--max_bin", default=155, type=int)
parser.add_argument("--max_depth", default=-1, type=int)
parser.add_argument("--min_data_in_leaf", default=20, type=int)
parser.add_argument("--boosting", choices=["gbdt", "rf", "dart", "goss"], default="gbdt")
parser.add_argument("--bagging_fraction", default=0.99, type=float)
parser.add_argument("--bagging_freq", default=1, type=int)

# LightGBM training hyperparameters
parser.add_argument("--lightgbm_learning_rate", default=1e-3, type=float)
parser.add_argument("--num_iterations", default=100, type=int)

# cross-validation settings
parser.add_argument("--k", default=5, type=int)
parser.add_argument("--repeats", default=4, type=int)

# Randomness seed   
parser.add_argument("--seed", type=int, default=0) 

# Notes for the wandb init explaining run
parser.add_argument("--notes", default="empty", type=str)
parser.add_argument("--run_name", default="sweep", type=str)
parser.add_argument("--sweep", action="store_true")


args = parser.parse_args()

# set to predefined seed or be completely random
if args.seed > 0:
    seed = args.seed
else:
    rng = np.random.default_rng()
    seed = int(rng.integers(1,9999))

# import datainfo
if args.corrupt:
    datalist = pd.read_csv("results/openml/corrupted_tasklist.csv")
else:
    datalist = pd.read_csv("results/openml/noncorrupted_tasklist.csv")

row = datalist.values[args.dataset, :]
did = row[1]
task = row[2]
print(row)
# Define metrics
if task == "Supervised Classification":
    metrics = ["nll", "accuracy"]
elif task == "Supervised Regression":
    metrics = ["rmse"]
else:
    AssertionError("task not recognized")
# initialize wandb
wandb.init(
    config=args, 
    name="{}-{}-{}-{}".format(row[3], args.missing, args.imputation, args.sweep), 
    entity="cardiac-ml", project="LSAM")
config = wandb.config
wandb.config.update({"dataset_name":row[3]})
wandb.config.update({"run_name":args.run_name})
wandb.config.update({"seed":seed}, allow_val_change=True)

# define summary metrics
wandb.define_metric("gbm_accuracy", summary="mean")
wandb.define_metric("gbm_nll", summary="mean")
wandb.define_metric("gbm_lognll", summary="mean")
wandb.define_metric("gbm_rmse", summary="mean")
wandb.define_metric("lsam_accuracy", summary="mean")
wandb.define_metric("lsam_nll", summary="mean")
wandb.define_metric("lsam_lognll", summary="mean")
wandb.define_metric("lsam_rmse", summary="mean")


# initialize/make model params
lsam_params = None
if (args.model == "LSAM") or (args.model == "Both"):
    lsam_params = {
        "d_model":config.d_model,
        "embedding_size":config.d_model,
        "embedding_layers":config.embedding_layers,
        "encoder_heads":config.encoder_heads,
        "encoder_layers":config.encoder_layers if config.depth < 0 else config.depth,
        "decoder_heads":config.decoder_heads,
        "decoder_layers":config.decoder_layers if config.depth < 0 else config.depth*2,
        "net_size":config.d_model,
        "net_layers":config.net_layers,
        "max_steps":config.max_steps,
        "learning_rate":config.learning_rate,
        "batch_size":config.batch_size,
        "early_stop":config.early_stopping,
        "noise_std":config.noise_std,
        "dropreg":config.drop_reg,
        "weight_decay":config.weight_decay,
        "l2":config.l2,
        "optimizer":config.optimizer,
    }

gbm_params = None
if (args.model == "LightGBM") or (args.model == "Both"):
    objective = 'softmax'
    if task == "Supervised Regression":
        objective = 'regression'
    gbm_params = {
        "num_leaves":config.num_leaves,
        "max_bin":config.max_bin,
        "max_depth":config.max_depth,
        "min_data_in_leaf":int(config.min_data_in_leaf),
        "learning_rate":config.lightgbm_learning_rate,
        "num_iterations":config.num_iterations,
        "boosting":config.boosting,
        "bagging_freq":0 if config.boosting == "goss" else config.bagging_freq,
        "bagging_fraction":config.bagging_fraction,
        "objective":objective
    }

# Train Models
if args.missing == "None":
    missing = None
else:
    missing = args.missing

if args.imputation == "None":
    imputation = None
else:
    imputation = args.imputation

metrics_df, perc_missing = run(
    dataset=did,
    task=task,
    missing=missing,
    imputation=imputation,
    train_complete=False,
    test_complete=True,
    rng_init=seed,
    lsam_params=lsam_params,
    gbm_params=gbm_params,
    corrupt=args.corrupt,
    noncorrupt=not args.corrupt,
    row_data=row,
    folds=args.k,
    repeats=args.repeats,
    perc_missing=args.p_missing_per_col,
    cols_miss=args.p_cols_missing,
    sweep=args.sweep,
    wandb=wandb # for incremental logging
)

# Log metrics to wandb

metrics_calc = {}

if task == "Supervised Classification":
    if args.model == "LightGBM":
        metrics_calc["gbm_combo"] = np.nanmean(metrics_df[("nll", "gbm")]) - np.nanmean(metrics_df[("accuracy", "gbm")])
    if args.model == "LSAM":
        metrics_calc["lsam_combo"] = np.nanmean(metrics_df[("nll", "lsam")]) - np.nanmean(metrics_df[("accuracy", "lsam")])

wandb.log(metrics_calc)
