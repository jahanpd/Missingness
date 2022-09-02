import os
import wandb
import pandas as pd
import argparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier

COUNT=30
ENTITY="cardiac-ml"
PROJECT_HP="missingness"
PROJECT_BM="LSAM"

parser = argparse.ArgumentParser()
parser.add_argument("--corrupt", action='store_true')
parser.add_argument("--noncorrupt", action='store_true')
parser.add_argument("--hps", action='store_true')
parser.add_argument("--save", action='store_true')
args = parser.parse_args()

assert args.corrupt or args.noncorrupt or args.hps
# import datainfo
if args.corrupt:
    feature_analysis = False
    path = "results/openml/corrupted_tasklist.csv"
    datalist = pd.read_csv(path)
elif args.noncorrupt:
    feature_analysis = False
    path = "results/openml/noncorrupted_tasklist.csv"
    datalist = pd.read_csv(path)
else:
    feature_analysis = True
    path = "results/openml/noncorrupted_tasklist.csv"
    datalist1 = pd.read_csv(path)
    path = "results/openml/corrupted_tasklist.csv"
    datalist2 = pd.read_csv(path)
    datalist = pd.concat([datalist1, datalist2], ignore_index=True)

api = wandb.Api()

def get_sweep(entity, project, sweep_id):
    try:
        sweep = api.sweep("{}/{}/{}".format(entity,project,sweep_id))
        return sweep
    except Exception as e:
        print(e)
        return "FAILED"

print(datalist)

if args.hps:
# review wins during hyperparameter training
    store_results = dict(
        dsname=[],
        nll_lsam=[],
        acc_lsam=[],
        nll_gbm=[],
        acc_gbm=[],
        NumberOfFeatures=[],
        NumberOfInstances=[],
        NumberOfClasses=[],
        NumberOfNumericFeatures=[],
        NumberOfSymbolicFeatures=[],
        winner_lsam_nll=[],
        winner_lsam_acc=[],
    )
    for i, row in enumerate(datalist.values):
        gbm_sweep = get_sweep(ENTITY, PROJECT_HP, row[-1])
        lsam_sweep = get_sweep(ENTITY, PROJECT_HP, row[-2])
        if gbm_sweep == "FAILED" or gbm_sweep == "FAILED":
            continue
        try:
            gbm_hps = gbm_sweep.best_run(order="+gbm_nll.mean").summary
            lsam_hps = lsam_sweep.best_run(order="+lsam_nll.mean").summary
            print(lsam_sweep.best_run(order="+lsam_nll.mean").config["weight_decay"])
        except Exception as e:
            print(e)
            print(row)
            continue
        store_results["dsname"].append(row[3])
        store_results["NumberOfFeatures"].append(row[4])
        store_results["NumberOfInstances"].append(row[5])
        store_results["NumberOfClasses"].append(row[6])
        store_results["NumberOfNumericFeatures"].append(row[7])
        store_results["NumberOfSymbolicFeatures"].append(row[8])
        store_results["nll_lsam"].append(lsam_hps["lsam_nll"]["mean"])
        store_results["acc_lsam"].append(lsam_hps["lsam_accuracy"]["mean"])
        store_results["nll_gbm"].append(gbm_hps["gbm_nll"]["mean"])
        store_results["acc_gbm"].append(gbm_hps["gbm_accuracy"]["mean"])
        store_results["winner_lsam_nll"].append(lsam_hps["lsam_nll"]["mean"] < gbm_hps["gbm_nll"]["mean"])
        store_results["winner_lsam_acc"].append(lsam_hps["lsam_accuracy"]["mean"] > gbm_hps["gbm_accuracy"]["mean"])

    if feature_analysis:
        predictors = ["NumberOfFeatures", "NumberOfInstances", "NumberOfClasses", "NumberOfNumericFeatures", "NumberOfSymbolicFeatures", ]
        outcomes = ["winner_lsam_acc", "winner_lsam_nll"]
        df_hps=pd.DataFrame(store_results)
        print(df_hps)
        X=df_hps[predictors]
        y=df_hps[outcomes]
        X["NumericRatio"]=X.NumberOfNumericFeatures.values / X.NumberOfFeatures.values
        X["FeatureInstanceRatio"]=X.NumberOfFeatures.values / X.NumberOfInstances.values
        rf_acc = RandomForestClassifier()
        rf_acc.fit(X.values, y.winner_lsam_acc.values)
        acc_feat_importance = sorted(list(zip(rf_acc.feature_importances_,list(X))), key=lambda x: x[0])
        rf_nll = RandomForestClassifier()
        rf_nll.fit(X.values, y.winner_lsam_nll.values)
        nll_feat_importance = sorted(list(zip(rf_nll.feature_importances_,list(X))),key=lambda x: x[0])
        print(acc_feat_importance)
        print(nll_feat_importance)

# review wins for benchmarking
store_results = dict(
        dsname=[],
        missingness=[],
        imputation=[],
        nll_lsam=[],
        acc_lsam=[],
        nll_gbm=[],
        acc_gbm=[],
        winner_lsam_nll=[],
        winner_lsam_acc=[],
    )

if args.corrupt or args.noncorrupt:
    for i, row in enumerate(datalist.values):
        if args.corrupt:
            missingness_list = ["None", "MCAR", "MAR", "MNAR"]
        else:
            missingness_list = ["None"]

        for missingness in missingness_list:
            for imputation in ["None", "Drop", "simple", "iterative", "miceforest"]:
                run_name = "{}-{}-{}-{}".format(row[3],missingness,imputation,args.corrupt)
                runs = api.runs(
                    path="{}/{}".format(ENTITY,PROJECT_BM),
                    filters={"config.run_name":"{}".format(run_name)}
                )
                if len(runs) > 0:
                    run = runs[0].summary
                    try:
                        store_results["nll_lsam"].append(run["lsam_nll"]["mean"])
                        store_results["acc_lsam"].append(run["lsam_accuracy"]["mean"])
                        store_results["nll_gbm"].append(run["gbm_nll"]["mean"])
                        store_results["acc_gbm"].append(run["gbm_accuracy"]["mean"])
                        store_results["winner_lsam_nll"].append(run["lsam_nll"]["mean"] < run["gbm_nll"]["mean"])
                        store_results["winner_lsam_acc"].append(run["lsam_accuracy"]["mean"] > run["gbm_accuracy"]["mean"])
                        store_results["dsname"].append(row[3])
                        store_results["missingness"].append(missingness)
                        store_results["imputation"].append(imputation)
                    except Exception as e:
                        print(run_name)
                        print(e)
    df=pd.DataFrame(store_results)
    print(df)
    if args.save:
        if args.corrupt:
            dataset="corrupt"
        else:
            dataset="noncorrupt"
        df.to_csv("./results/openml/benchmark_results_{}.csv".format(dataset), index=False)
