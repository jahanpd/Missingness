import os
import wandb
import pandas as pd
import argparse
import numpy as np

COUNT=40
ENTITY="cardiac-ml"
PROJECT="missingness"

api = wandb.Api()

parser = argparse.ArgumentParser()
parser.add_argument("--overwrite", action='store_true')
parser.add_argument("--ds", type=int, default=-1)
parser.add_argument("--recover", action='store_true')
args = parser.parse_args()

sweep_config_lsam = {
    "program":"train.py",
    "name":"dataset",
    "method":"bayes",
    "metric":{"goal":"minimize","name":"placeholder"},
    "parameters":{
        # "d_model":{"max":40,"min":4,"distribution":"int_uniform"},
        "d_model":{"values":[16, 32, 64],"distribution":"categorical"},
        "depth":{"max":4,"min":1,"distribution":"int_uniform"},
        "embedding_layers":{"max":4,"min":1,"distribution":"int_uniform"},
        # "encoder_layers":{"max":6,"min":3,"distribution":"int_uniform"},
        # "decoder_layers":{"max":3,"min":1,"distribution":"int_uniform"},
        # "net_layers":{"max":3,"min":1,"distribution":"int_uniform"},
        # "early_stopping":{"max":0.9,"min":0.0,"distribution":"uniform"},
        "learning_rate":{"max":0.001,"min":0.0000001,"distribution":"log_uniform_values"},
        # "batch_size":{"values":[64, 128],"distribution":"categorical"},
        # "noise_std":{"values":[0.01, 0.0],"distribution":"categorical"},
        "noise_std":{"max":10.0,"min":0.000001,"distribution":"log_uniform_values"},
        # "drop_reg":{"max":0.001,"min":0.0000001,"distribution":"log_uniform_values"},
        # "weight_decay":{"max":100.0,"min":0.0000001,"distribution":"log_uniform_values"},
        # "l2":{"max":0.001,"min":0.000000001,"distribution":"log_uniform_values"},
        # "optimizer":{"values":["adam","adabelief"],"distribution":"categorical"},
     },
    "early_terminate":{
        "type": "hyperband",
        "min_iter":1
    },
    "command":[
        "${env}",
        "${interpreter}",
        "${program}",
        "--model",
        "LSAM",
        "--dataset",
        "rownumber",
        "--k",
        "4",
        "--repeats",
        "1",
        "--notes",
        "hyperparameter search",
        "--sweep"
    ]
} 

sweep_config_gbm = {
    "program":"train.py",
    "name":"dataset",
    "method":"bayes",
    "metric":{"goal":"minimize","name":"placeholder"},
    "parameters":{
        "num_leaves":{"max":1000,"min":1,"distribution":"int_uniform"},
        "max_bin":{"max":1000,"min":100,"distribution":"int_uniform"},
        # "max_depth":{"max":50,"min":4,"distribution":"int_uniform"},
        "min_data_in_leaf":{"max":500,"min":1,"distribution":"int_uniform"},
        "lightgbm_learning_rate":{"max":0.1,"min":0.000001,"distribution":"log_uniform_values"},
        # "num_iterations":{"max":10000,"min":100,"distribution":"int_uniform"},
        "boosting":{"values":["gbdt", "rf", "dart", "goss"],"distribution":"categorical"},
     },
    "early_terminate":{
        "type": "hyperband",
        "min_iter":1
    },
    "command":[
        "${env}",
        "${interpreter}",
        "${program}",
        "--model",
        "LightGBM",
        "--dataset",
        "rownumber",
        "--k",
        "4",
        "--repeats",
        "1",
        "--notes",
        "hyperparameter search",
        "--sweep"
    ]
} 

# import datainfo
corrupted_path = "results/openml/corrupted_tasklist.csv"
noncorrupted_path = "results/openml/noncorrupted_tasklist.csv"
datalist_corrupted = pd.read_csv(corrupted_path)
datalist_noncorrupted = pd.read_csv(noncorrupted_path)

def check_sweepids(dataframe):
    if "lsam_sweepid" not in dataframe:
        dataframe["lsam_sweepid"] = ["placeholder"]*len(dataframe)
    if "gbm_sweepid" not in dataframe:
        dataframe["gbm_sweepid"] = ["placeholder"]*len(dataframe)

check_sweepids(datalist_corrupted)
check_sweepids(datalist_noncorrupted)

def check_sweep(entity, project, sweep_id):
    try:
        sweep = api.sweep("{}/{}/{}".format(entity,project,sweep_id))
        if not args.overwrite:
            return sweep_id, len(sweep.runs)
        else:
            return "placeholder", 0
    except Exception as e:
        print(entity, project, sweep_id)
        print(e)
        return "placeholder", 0

def run_hpsearch(datalist, corrupt):
    if args.ds < 0:
        datasets = range(len(datalist))
    else:
        datasets = [args.ds]
    for i in datasets:
        row = datalist.values[i,:]
        new_lsam = sweep_config_lsam.copy()
        new_lsam["command"] = sweep_config_lsam["command"].copy()
        new_gbm = sweep_config_gbm.copy()
        new_gbm["command"] = sweep_config_gbm["command"].copy()
        if row[2] == "Supervised Classification":
            metric_lsam = "lsam_lognll.mean"
            metric_gbm = "gbm_lognll.mean"
        elif row[2] == "Supervised Regression":
            metric_lsam = "lsam_rmse.mean"
            metric_gbm = "gbm_rmse.mean"

        new_lsam["metric"]["name"] = metric_lsam
        new_gbm["metric"]["name"] = metric_gbm
        new_lsam["command"] = [str(i) if x == "rownumber" else x for x in new_lsam["command"]]
        new_gbm["command"] = [str(i) if x == "rownumber" else x for x in new_gbm["command"]]
        if corrupt:
            new_lsam["command"].append("--corrupt")
            new_lsam["command"].append("${args}")
            new_gbm["command"].append("--corrupt")
            new_gbm["command"].append("${args}")
        else:
            new_lsam["command"].append("${args}")
            new_gbm["command"].append("${args}")
 
        new_lsam["name"] = row[3]
        new_gbm["name"] = row[3]
        print(datalist)

        datalist.lsam_sweepid.values[i], reslsam = check_sweep(ENTITY, PROJECT, datalist.lsam_sweepid.values[i])
        datalist.gbm_sweepid.values[i], resgbm = check_sweep(ENTITY, PROJECT, datalist.gbm_sweepid.values[i])

        if (datalist.gbm_sweepid.values[i]=="placeholder") or resgbm < COUNT:
            if datalist.gbm_sweepid.values[i]=="placeholder":
                sweep_id_gbm = wandb.sweep(new_gbm,entity="cardiac-ml",project="missingness")
            else:
                sweep_id_gbm = datalist.gbm_sweepid.values[i]
            os.environ["SWEEPIDGBM"] = sweep_id_gbm
            os.system("wandb agent --count {} cardiac-ml/missingness/{}".format(COUNT - resgbm, sweep_id_gbm))
            datalist.gbm_sweepid.values[i] = sweep_id_gbm
            if corrupt:
                datalist.to_csv(corrupted_path, index=False)
            else:
                datalist.to_csv(noncorrupted_path, index=False)
        if (datalist.lsam_sweepid.values[i]=="placeholder") or reslsam < COUNT:
            if datalist.lsam_sweepid.values[i]=="placeholder":
                sweep_id_lsam = wandb.sweep(new_lsam,entity="cardiac-ml",project="missingness")
            else:
                sweep_id_lsam = datalist.lsam_sweepid.values[i]
            os.environ["SWEEPIDLSAM"] = sweep_id_lsam
            os.system("wandb agent --count {} cardiac-ml/missingness/{}".format(COUNT - reslsam, sweep_id_lsam))
            datalist.lsam_sweepid.values[i] = sweep_id_lsam
            if corrupt:
                datalist.to_csv(corrupted_path, index=False)
            else:
                datalist.to_csv(noncorrupted_path, index=False)

        

def recover_filt(sweep, model, dsname):
    m = model in sweep.config["metric"]["name"]
    n = dsname == sweep.config["name"]
    return m and n

if not args.recover:
    run_hpsearch(datalist_noncorrupted, False)
    run_hpsearch(datalist_corrupted, True)
else:
    project = api.project(PROJECT, entity=ENTITY)
    sweeps = project.sweeps()
    print(datalist_noncorrupted)
    for i, dsname in enumerate(datalist_noncorrupted.name.values):
        subsweeps_lsam = [s for s in sweeps if recover_filt(s, "lsam", dsname)]
        subsweeps_gbm = [s for s in sweeps if recover_filt(s, "gbm", dsname)]
        if len(subsweeps_lsam) > 0:
            datalist_noncorrupted.loc[i, "lsam_sweepid"] = subsweeps_lsam[0].id
        if len(subsweeps_gbm) > 0:
            datalist_noncorrupted.loc[i, "gbm_sweepid"] = subsweeps_gbm[0].id
    print(datalist_noncorrupted)
    datalist_noncorrupted.to_csv(noncorrupted_path, index=False)
    print(datalist_corrupted)
    for i, dsname in enumerate(datalist_corrupted.name.values):
        subsweeps_lsam = [s for s in sweeps if recover_filt(s, "lsam", dsname)]
        subsweeps_gbm = [s for s in sweeps if recover_filt(s, "gbm", dsname)]
        if len(subsweeps_lsam) > 0:
            datalist_corrupted.loc[i, "lsam_sweepid"] = subsweeps_lsam[0].id
        if len(subsweeps_gbm) > 0:
            datalist_corrupted.loc[i, "gbm_sweepid"] = subsweeps_gbm[0].id
    print(datalist_corrupted)
    datalist_corrupted.to_csv(corrupted_path, index=False)
