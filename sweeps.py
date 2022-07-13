import os
import wandb
import pandas as pd
import argparse
import numpy as np

COUNT=30
ENTITY="cardiac-ml"
PROJECT="missingness"

api = wandb.Api()

parser = argparse.ArgumentParser()
parser.add_argument("--overwrite", action='store_true')
args = parser.parse_args()

sweep_config_lsam = {
    "program":"train.py",
    "name":"dataset",
    "method":"bayes",
    "metric":{"goal":"minimize","name":"placeholder"},
    "parameters":{
        "d_model":{"max":128,"min":32,"distribution":"int_uniform"},
        "embedding_layers":{"max":8,"min":1,"distribution":"int_uniform"},
        "encoder_layers":{"max":12,"min":1,"distribution":"int_uniform"},
        "encoder_heads":{"max":5,"min":1,"distribution":"int_uniform"},
        "decoder_layers":{"max":12,"min":1,"distribution":"int_uniform"},
        "decoder_heads":{"max":5,"min":1,"distribution":"int_uniform"},
        "net_layers":{"max":8,"min":1,"distribution":"int_uniform"},
        "max_steps":{"max":5000,"min":2000,"distribution":"int_uniform"},
        "early_stopping":{"max":0.9,"min":0.0,"distribution":"uniform"},
        "learning_rate":{"max":0.001,"min":0.0001,"distribution":"log_uniform_values"},
        "batch_size":{"values":[32, 64, 128],"distribution":"categorical"},
        "noise_std":{"max":3,"min":0.1,"distribution":"log_uniform_values"},
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
    ]
} 

sweep_config_gbm = {
    "program":"train.py",
    "name":"dataset",
    "method":"bayes",
    "metric":{"goal":"minimize","name":"placeholder"},
    "parameters":{
        "num_leaves":{"max":40,"min":5,"distribution":"int_uniform"},
        "max_bin":{"max":255,"min":5,"distribution":"int_uniform"},
        "max_depth":{"max":32,"min":4,"distribution":"int_uniform"},
        "min_data_in_leaf":{"max":30,"min":10,"distribution":"int_uniform"},
        "lightgbm_learning_rate":{"max":0.1,"min":0.00001,"distribution":"log_uniform_values"},
        "num_iterations":{"max":1000,"min":100,"distribution":"int_uniform"},
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
    ]
} 

# import datainfo
corrupted_path = "results/openml/corrupted_tasklist.csv"
noncorrupted_path = "results/openml/noncorrupted_tasklist.csv"
datalist_corrupted = pd.read_csv(corrupted_path)
datalist_noncorrupted = pd.read_csv(noncorrupted_path)

def check_sweepids(dataframe):
    if "lsam_sweepid" not in dataframe:
        dataframe["lsam_sweepid"] = [""]*len(dataframe)
    if "gbm_sweepid" not in dataframe:
        dataframe["gbm_sweepid"] = [""]*len(dataframe)

check_sweepids(datalist_corrupted)
check_sweepids(datalist_noncorrupted)

def check_sweep(entity, project, sweep_id):
    try:
        sweep = api.sweep("{}/{}/{}".format(entity,project,sweep_id))
        if len(sweep.runs) >= COUNT:
            return sweep_id
        else:
            return ""
    except Exception as e:
        print(e)
        return ""

def run_hpsearch(datalist, corrupt):
    for i in range(len(datalist)):
        row = datalist.values[i,:]
        new_lsam = sweep_config_lsam.copy()
        new_gbm = sweep_config_gbm.copy()
        if row[2] == "Supervised Classification":
            metric_lsam = "LSAM_combo"
            metric_gbm = "LightGBM_combo"
        elif row[2] == "Supervised Regression":
            metric_lsam = "LSAM_rmse"
            metric_gbm = "LightGBM_rmse"

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

        if datalist.lsam_sweepid.values[i] != "":
            datalist.lsam_sweepid.values[i] = check_sweep(ENTITY, PROJECT, datalist.lsam_sweepid.values[i])

        if datalist.gbm_sweepid.values[i] != "":
            datalist.gbm_sweepid.values[i] = check_sweep(ENTITY, PROJECT, datalist.gbm_sweepid.values[i])

        if (datalist.lsam_sweepid.values[i]=="") or args.overwrite:
            sweep_id_lsam = wandb.sweep(new_lsam,entity="cardiac-ml",project="missingness")
            os.system("wandb agent --count {} cardiac-ml/missingness/{}".format(COUNT, sweep_id_lsam))
            datalist.lsam_sweepid.values[i] = sweep_id_lsam
            if corrupt:
                datalist.to_csv(corrupted_path, index=False)
            else:
                datalist.to_csv(noncorrupted_path, index=False)

        if (datalist.gbm_sweepid.values[i]=="") or args.overwrite:
            sweep_id_gbm = wandb.sweep(new_gbm,entity="cardiac-ml",project="missingness")
            os.system("wandb agent --count {} cardiac-ml/missingness/{}".format(COUNT, sweep_id_gbm))
            datalist.gbm_sweepid.values[i] = sweep_id_gbm
            if corrupt:
                datalist.to_csv(corrupted_path, index=False)
            else:
                datalist.to_csv(noncorrupted_path, index=False)


run_hpsearch(datalist_noncorrupted, False)
run_hpsearch(datalist_corrupted, True)
