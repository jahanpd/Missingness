import os
import wandb
import pandas as pd
import argparse
import numpy as np

ENTITY="cardiac-ml"
PROJECT="missingness"

api = wandb.Api()

parser = argparse.ArgumentParser()
parser.add_argument("--missing", choices=["None", "MCAR", "MAR", "MNAR"], default="None", nargs='+')
parser.add_argument("--imputation", choices=["None", "simple", "iterative", "miceforest"], default="None", nargs='+')
parser.add_argument("--overwrite", action='store_true')
parser.add_argument("--corrupt", action='store_true')
parser.add_argument("--seed", type=int, default=69)
args = parser.parse_args()

def check_sweepids(dataframe, path):
    if ("lsam_sweepid" not in dataframe) or ("gbm_sweepid" not in dataframe):
        AssertionError("No sweepids in dataframe at {}".format(path))

# import datainfo
if args.corrupt:
    path = "results/openml/corrupted_tasklist.csv"
    datalist = pd.read_csv(path)
else:
    path = "results/openml/noncorrupted_tasklist.csv"
    datalist = pd.read_csv(path)

check_sweepids(datalist, path)

rng = np.random.default_rng(args.seed)

print(datalist)

def get_sweep(entity, project, sweep_id):
    try:
        sweep = api.sweep("{}/{}/{}".format(entity,project,sweep_id))
        return sweep
    except Exception as e:
        print(e)
        return "FAILED"


def check_run(entity, project, run_name, overwrite):
    runs = api.runs(
        path="{}/{}".format(entity,"LSAM"),
        filters={"config.run_name":"{}".format(run_name)}
    )
    print("Benchmark exists with {} runs, will run? {}".format(len(runs), overwrite))
    return len(runs) == 0 or overwrite


for i, row in enumerate(datalist.values):
    print(row)
    gbm_sweep = get_sweep(ENTITY, PROJECT, row[-1])
    lsam_sweep = get_sweep(ENTITY, PROJECT, row[-2])
    if gbm_sweep == "FAILED" or gbm_sweep == "FAILED":
        continue
    if row[2] == "Supervised Classification":
        gbm_hps = gbm_sweep.best_run(order="+gbm_nll.mean").config
        lsam_hps = lsam_sweep.best_run(order="+lsam_nll.mean").config
    elif row[2] == "Supervised Regression":
        gbm_hps = gbm_sweep.best_run().config
        lsam_hps = lsam_sweep.best_run().config
    else:
        print("Goal {} not specified".format(row[2]))
        continue
    assert lsam_hps["dataset"] == gbm_hps["dataset"]
    parameters = {
        "dataset":lsam_hps["dataset"],
        "d_model":lsam_hps["d_model"],
        "embedding_layers":lsam_hps["embedding_layers"],
        "encoder_layers":lsam_hps["encoder_layers"],
        "decoder_layers":lsam_hps["decoder_layers"],
        "net_layers":lsam_hps["net_layers"],
        "early_stopping":lsam_hps["early_stopping"],
        "learning_rate":lsam_hps["learning_rate"],
        "batch_size":lsam_hps["batch_size"],
        "noise_std":lsam_hps["noise_std"],
        "drop_reg":lsam_hps["drop_reg"],
        "weight_decay":lsam_hps["weight_decay"],
        "optimizer":lsam_hps["optimizer"],
        "num_leaves":gbm_hps["num_leaves"],
        "max_bin":gbm_hps["max_bin"],
        "max_depth":gbm_hps["max_depth"],
        "min_data_in_leaf":gbm_hps["min_data_in_leaf"],
        "lightgbm_learning_rate":gbm_hps["lightgbm_learning_rate"],
        "num_iterations":gbm_hps["num_iterations"],
     }
    if args.corrupt:
        missingness_list = args.missing
    else:
        missingness_list = ["None"]
    imputation_list = args.imputation
    for missingness in missingness_list:
        seed = int(rng.integers(1,9999))
        for imputation in imputation_list:
            if imputation != "None" and missingness == "None" and args.corrupt:
                print("imputation {} and missingess {} so skipping".format(imputation, missingness))
                continue
            run_name = "{}-{}-{}-{}".format(row[3],missingness,imputation,args.corrupt)
            if check_run(ENTITY, PROJECT, run_name, args.overwrite):
                params = parameters.copy()
                params["run_name"] = run_name
                params["seed"] = seed
                params["dataset"] = i
                params["missing"] = missingness
                params["imputation"] = imputation
                command = ("python train.py "
                    "--dataset {dataset} "
                    "--missing {missing} "
                    "--imputation {imputation} "
                    "--seed {seed} "
                    "--run_name {run_name} "
                    "--d_model {d_model} "
                    "--embedding_layers {embedding_layers} "
                    "--encoder_layers {encoder_layers} "
                    "--decoder_layers {decoder_layers} "
                    "--net_layers {net_layers} "
                    "--early_stopping {early_stopping} "
                    "--optimizer {optimizer} "
                    "--learning_rate {learning_rate} "
                    "--batch_size {batch_size} "
                    "--noise_std {noise_std} "
                    "--drop_reg {drop_reg} "
                    "--weight_decay {weight_decay} "
                    "--num_leaves {num_leaves} "
                    "--max_bin {max_bin} "
                    "--max_depth {max_depth} "
                    "--min_data_in_leaf {min_data_in_leaf} "
                    "--lightgbm_learning_rate {lightgbm_learning_rate} "
                    "--num_iterations {num_iterations} "
                    "--k 4 "
                    "--repeats 2 "
                ).format(**params)
                if args.corrupt:
                    command = command + "--corrupt"
                print(command)
                os.system(command)

# runs = api.runs(
#     path="{}/{}".format(ENTITY,PROJECT),
#     filters={"config.":"{}".format(name)}
# )
# print(runs)
