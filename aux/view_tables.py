import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import itertools
import sys

## tables for latent space experiments
ed1 = pd.read_csv('../results/latent_space/Ensemble_Distances1_None.csv')
ed2 = pd.read_csv('../results/latent_space/Ensemble_Distances2_None.csv')
uat1 = pd.read_csv('../results/latent_space/LSAM_Distances1_None.csv')
uat2 = pd.read_csv('../results/latent_space/LSAM_Distances2_None.csv')
uat_missing = pd.read_csv('../results/latent_space/LSAM_Distances1_missingness_None.csv')
uat_probs = pd.read_csv('../results/latent_space/LSAM_drop_probs_missingness_None.csv')

# table 1 prep
rows = [7, 11, 3, 13, 14]
indexes = ['{x1}', '{x2}', '{x1, x2}', '{x3 (noise)}', '{x4 (signal)}']
colnames = [('LSAM', '{}'), ('LSAM', 'p-value'), ('Ensemble', '{}'), ('Ensemble', 'p-value')]
result_strs = []

data = np.hstack([uat2.values[:,1:], ed2.values[:,1:]])
for r in rows:
    row = []
    for c, col in enumerate(colnames):
        val = "{:.2f}".format(data[r, c])
        if col[1] == 'p-value':
           val = "{:.2e}".format(data[r, c])
        row.append(val)
    result_strs.append(row)

multiindex = pd.MultiIndex.from_tuples(colnames, names=["Model", "Measure"])
df = pd.DataFrame(result_strs, index=indexes, columns=multiindex)
s = df.style.format(precision=2)
print(s.to_latex())


# table 2 prep
rows = [7, 11, 3, 15]
indexes = ['{x1}', '{x2}', '{x1, x2}', '{}']
colnames = [('LSAM', '+{noise}'), ('LSAM', '+{signal}'), ('LSAM', 'p-value'), ('Ensemble', '+{noise}'), ('Ensemble', '+{signal}'), ('Ensemble', 'p-value')]
result_strs = []

data = np.hstack([uat1.values[:,1:], ed1.values[:,1:]])
for r in rows:
    row = []
    for c, col in enumerate(colnames):
        val = "{:.2f}".format(data[r, c])
        if col[1] == 'p-value':
           val = "{:.2e}".format(data[r, c])
        row.append(val)
    result_strs.append(row)

multiindex = pd.MultiIndex.from_tuples(colnames, names=["Model", "Measure"])
df = pd.DataFrame(result_strs, index=indexes, columns=multiindex)
s = df.style.format(precision=2)
print(s.to_latex())


# table 3 prep
rows = [7, 11, 3, 15]
indexes = ['{x1}', '{x2}', '{x1, x2}', '{}']
colnames = ['0%', '20%', '40%', '60%', '80%', '99%']
result_strs = []

data = uat_missing.values[:, 1:]
for r in rows:
    row = []
    for c, col in enumerate(colnames):
        val = "{:.2f}".format(data[r, c])
        row.append(val)
    result_strs.append(row)

df = pd.DataFrame(result_strs, index=indexes, columns=colnames)
s = df.style.format(precision=2)
print(s.to_latex())

# table 4 prep
rows = [0,1,2,3]
indexes = ['{x1}', '{x2}', '{x1, x2}', '{}']
colnames = ['0%', '20%', '40%', '60%', '80%', '99%']
result_strs = []

print(uat_probs)
data = uat_probs.values[:, 1:]
for r in rows:
    row = []
    for c, col in enumerate(colnames):
        val = "{:.2f}".format(data[r, c])
        row.append(val)
    result_strs.append(row)

df = pd.DataFrame(result_strs, index=indexes, columns=colnames)
s = df.style.format(precision=2)
print(s.to_latex())

## tables for performance experiments
## GENERATE TABLES FOR CORRUPTED DATA
datasets = pd.read_csv('../results/openml/corrupted_tasklist.csv')
name_list_full = datasets.name.values

cores = pd.read_csv('../results/openml/benchmark_results_corrupt.csv')

name_list = set(cores.dsname)
print("fraction of datasets run: ", len(name_list) / len(name_list_full))
print("number of datasets: ", len(name_list_full))


def table_maker(metric="accuracy"):
    keys = itertools.product(["MCAR", "MNAR", "MAR"], ["None", "Simple", "Iterative", "Miceforest"], ["LSAM", "LightGBM"])
    process_dict = {
        (" ", " ", "Dataset"):[],
        (" ", " ", "Metric"):[],
        ("None", "None", "LSAM"):[],
        ("None", "None", "LightGBM"):[],
    }
    for key in keys:
        process_dict[key] = []

    key_sub = list(itertools.product(["MCAR", "MNAR", "MAR"], ["None", "Simple", "Iterative", "Miceforest"]))
    for name in name_list:
        process_dict[(" ", " ", "Dataset")].append(name)
        process_dict[(" ", " ", "Metric")].append(metric)
            
        # get baseline data
        if len(cores.loc[(cores.dsname==name) & (cores.missingness=='None') & (cores.imputation=='None')]) > 0:
            label = 'nll' if metric == 'nll' else 'acc'
            baseline_lsam = cores.loc[
                (cores.dsname==name) & (cores.missingness=='None') & (cores.imputation=='None'), 
                "{}_lsam".format(label)].values[0]
            baseline_gbm = cores.loc[
                (cores.dsname==name) & (cores.missingness=='None') & (cores.imputation=='None'), 
                "{}_gbm".format(label)].values[0]
        else:
            baseline_lsam = np.nan
            baseline_gbm = np.nan
        process_dict[("None", "None", "LSAM")].append(baseline_lsam)
        process_dict[("None", "None", "LightGBM")].append(baseline_gbm)
        # get the same as above but for each missingness and imputation pattern
        for key in key_sub:
            imp = key[1] if key[1] == 'None' else key[1].lower()
            if len(cores.loc[(cores.dsname==name) & (cores.missingness==key[0]) & (cores.imputation==imp)]) > 0:
                test_lsam = cores.loc[
                    (cores.dsname==name) & (cores.missingness==key[0]) & (cores.imputation==imp), 
                    "{}_lsam".format(label)].values[0]
                test_gbm = cores.loc[
                    (cores.dsname==name) & (cores.missingness==key[0]) & (cores.imputation==imp), 
                    "{}_gbm".format(label)].values[0]
                if metric == 'accuracy':
                    lsam = (test_lsam - baseline_lsam)
                    gbm = (test_gbm - baseline_gbm)
                else:
                    lsam = (baseline_lsam - test_lsam)
                    gbm = (baseline_gbm - test_gbm)
            else:
                lsam = np.nan
                gbm = np.nan

            process_dict[(key[0], key[1], "LightGBM")].append(gbm)
            process_dict[(key[0], key[1], "LSAM")].append(lsam)
    return process_dict

process_dict = table_maker("accuracy")
final_results = pd.DataFrame(process_dict).fillna(np.inf)
datasets = final_results[" "][" "]["Dataset"].values
final_results.index = datasets
final_results.to_pickle('../results/openml/openml_acc.pickle')
print("None")
sub = final_results["None"] 
s = sub.style.format(precision=2).highlight_max(axis=1, props="textbf:--rwrap;")
print(s.to_latex())
print("MCAR")
sub = final_results["MCAR"] 
s = sub.style.format(precision=2).highlight_max(axis=1, props="textbf:--rwrap;")
print(s.to_latex())
print("MAR")
sub = final_results["MAR"] 
s = sub.style.format(precision=2).highlight_max(axis=1, props="textbf:--rwrap;")
print(s.to_latex())
print("MNAR")
sub = final_results["MNAR"] 
s = sub.style.format(precision=2).highlight_max(axis=1, props="textbf:--rwrap;")
print(s.to_latex())

print('ACCURACY')
print("win ratio baseline: {}".format(np.sum(final_results["None"]['None']['LSAM'].values > final_results["None"]['None']['LightGBM'].values) / len(final_results['None']['None'])))
print("win ratio for MNAR none vs none: {}".format(np.sum(final_results["MNAR"]['None']['LSAM'].values > final_results["MNAR"]['None']['LightGBM'].values) / len(final_results['MNAR']['None'])))
print("win ratio for MCAR none vs none: {}".format(np.sum(final_results["MCAR"]['None']['LSAM'].values > final_results["MCAR"]['None']['LightGBM'].values) / len(final_results['MCAR']['None'])))
print("win ratio for MAR none vs none: {}".format(np.sum(final_results["MAR"]['None']['LSAM'].values > final_results["MAR"]['None']['LightGBM'].values) / len(final_results['MAR']['None'])))

print('vs self')
print("win ratio for MNAR none vs simple: {}".format(np.sum(final_results["MNAR"]['None']['LSAM'].values > final_results["MNAR"]['Simple']['LSAM'].values) / len(final_results['MNAR']['None'])))
print("win ratio for MCAR none vs simple: {}".format(np.sum(final_results["MCAR"]['None']['LSAM'].values > final_results["MCAR"]['Simple']['LSAM'].values) / len(final_results['MCAR']['None'])))
print("win ratio for MAR none vs simple: {}".format(np.sum(final_results["MAR"]['None']['LSAM'].values > final_results["MAR"]['Simple']['LSAM'].values) / len(final_results['MAR']['None'])))

print("win ratio for MNAR none vs iterative: {}".format(np.sum(final_results["MNAR"]['None']['LSAM'].values > final_results["MNAR"]['Iterative']['LSAM'].values) / len(final_results['MNAR']['None'])))
print("win ratio for MCAR none vs iterative: {}".format(np.sum(final_results["MCAR"]['None']['LSAM'].values > final_results["MCAR"]['Iterative']['LSAM'].values) / len(final_results['MCAR']['None'])))
print("win ratio for MAR none vs iterative: {}".format(np.sum(final_results["MAR"]['None']['LSAM'].values > final_results["MAR"]['Iterative']['LSAM'].values) / len(final_results['MAR']['None'])))

print("win ratio for MNAR none vs mice: {}".format(np.sum(final_results["MNAR"]['None']['LSAM'].values > final_results["MNAR"]['Miceforest']['LSAM'].values) / len(final_results['MNAR']['None'])))
print("win ratio for MCAR none vs mice: {}".format(np.sum(final_results["MCAR"]['None']['LSAM'].values > final_results["MCAR"]['Miceforest']['LSAM'].values) / len(final_results['MCAR']['None'])))
print("win ratio for MAR none vs mice: {}".format(np.sum(final_results["MAR"]['None']['LSAM'].values > final_results["MAR"]['Miceforest']['LSAM'].values) / len(final_results['MAR']['None'])))

print('vs gbm')
print("win ratio for MNAR none vs simple: {}".format(np.sum(final_results["MNAR"]['None']['LSAM'].values > final_results["MNAR"]['Simple']['LightGBM'].values) / len(final_results['MNAR']['None'])))
print("win ratio for MCAR none vs simple: {}".format(np.sum(final_results["MCAR"]['None']['LSAM'].values > final_results["MCAR"]['Simple']['LightGBM'].values) / len(final_results['MCAR']['None'])))
print("win ratio for MAR none vs simple: {}".format(np.sum(final_results["MAR"]['None']['LSAM'].values > final_results["MAR"]['Simple']['LightGBM'].values) / len(final_results['MAR']['None'])))

print("win ratio for MNAR none vs iterative: {}".format(np.sum(final_results["MNAR"]['None']['LSAM'].values > final_results["MNAR"]['Iterative']['LightGBM'].values) / len(final_results['MNAR']['None'])))
print("win ratio for MCAR none vs iterative: {}".format(np.sum(final_results["MCAR"]['None']['LSAM'].values > final_results["MCAR"]['Iterative']['LightGBM'].values) / len(final_results['MCAR']['None'])))
print("win ratio for MAR none vs iterative: {}".format(np.sum(final_results["MAR"]['None']['LSAM'].values > final_results["MAR"]['Iterative']['LightGBM'].values) / len(final_results['MAR']['None'])))

print("win ratio for MNAR none vs mice: {}".format(np.sum(final_results["MNAR"]['None']['LSAM'].values > final_results["MNAR"]['Miceforest']['LightGBM'].values) / len(final_results['MNAR']['None'])))
print("win ratio for MCAR none vs mice: {}".format(np.sum(final_results["MCAR"]['None']['LSAM'].values > final_results["MCAR"]['Miceforest']['LightGBM'].values) / len(final_results['MCAR']['None'])))
print("win ratio for MAR none vs mice: {}".format(np.sum(final_results["MAR"]['None']['LSAM'].values > final_results["MAR"]['Miceforest']['LightGBM'].values) / len(final_results['MAR']['None'])))

process_dict = table_maker("nll")
final_results = pd.DataFrame(process_dict)
datasets = final_results[" "][" "]["Dataset"].values
final_results.index = datasets
final_results.to_pickle('../results/openml/openml_nll.pickle')
final_results.fillna(np.inf, inplace=True)
print("None")
sub = final_results["None"] 
s = sub.style.format(precision=2).highlight_min(axis=1, props="textbf:--rwrap;")
print(s.to_latex())
print("MCAR")
sub = final_results["MCAR"] 
s = sub.style.format(precision=2).highlight_min(axis=1, props="textbf:--rwrap;")
print(s.to_latex())
print("MAR")
sub = final_results["MAR"] 
s = sub.style.format(precision=2).highlight_min(axis=1, props="textbf:--rwrap;")
print(s.to_latex())
print("MNAR")
sub = final_results["MNAR"] 
s = sub.style.format(precision=2).highlight_min(axis=1, props="textbf:--rwrap;")
print(s.to_latex())
print('NLL')
print("win ratio baseline: {}".format(np.sum(final_results["None"]['None']['LSAM'].values < final_results["None"]['None']['LightGBM'].values) / len(final_results['None']['None'])))
print("win ratio for MNAR: {}".format(np.sum(final_results["MNAR"]['None']['LSAM'].values < final_results["MNAR"]['None']['LightGBM'].values) / len(final_results['MNAR']['None'])))
print("win ratio for MCAR: {}".format(np.sum(final_results["MCAR"]['None']['LSAM'].values < final_results["MCAR"]['None']['LightGBM'].values) / len(final_results['MCAR']['None'])))
print("win ratio for MAR: {}".format(np.sum(final_results["MAR"]['None']['LSAM'].values < final_results["MAR"]['None']['LightGBM'].values) / len(final_results['MAR']['None'])))


print("win ratio for MNAR none vs simple: {}".format(np.sum(final_results["MNAR"]['None']['LSAM'].values > final_results["MNAR"]['Simple']['LSAM'].values) / len(final_results['MNAR']['None'])))
print("win ratio for MCAR none vs simple: {}".format(np.sum(final_results["MCAR"]['None']['LSAM'].values > final_results["MCAR"]['Simple']['LSAM'].values) / len(final_results['MCAR']['None'])))
print("win ratio for MAR none vs simple: {}".format(np.sum(final_results["MAR"]['None']['LSAM'].values > final_results["MAR"]['Simple']['LSAM'].values) / len(final_results['MAR']['None'])))

print("win ratio for MNAR none vs iterative: {}".format(np.sum(final_results["MNAR"]['None']['LSAM'].values > final_results["MNAR"]['Iterative']['LSAM'].values) / len(final_results['MNAR']['None'])))
print("win ratio for MCAR none vs simple: {}".format(np.sum(final_results["MCAR"]['None']['LSAM'].values > final_results["MCAR"]['Iterative']['LSAM'].values) / len(final_results['MCAR']['None'])))
print("win ratio for MAR none vs simple: {}".format(np.sum(final_results["MAR"]['None']['LSAM'].values > final_results["MAR"]['Iterative']['LSAM'].values) / len(final_results['MAR']['None'])))

print("win ratio for MNAR none vs mice: {}".format(np.sum(final_results["MNAR"]['None']['LSAM'].values > final_results["MNAR"]['Miceforest']['LSAM'].values) / len(final_results['MNAR']['None'])))
print("win ratio for MCAR none vs mice: {}".format(np.sum(final_results["MCAR"]['None']['LSAM'].values > final_results["MCAR"]['Miceforest']['LSAM'].values) / len(final_results['MCAR']['None'])))
print("win ratio for MAR none vs mice: {}".format(np.sum(final_results["MAR"]['None']['LSAM'].values > final_results["MAR"]['Miceforest']['LSAM'].values) / len(final_results['MAR']['None'])))

print('vs gbm')
print("win ratio for MNAR none vs simple: {}".format(np.sum(final_results["MNAR"]['None']['LSAM'].values > final_results["MNAR"]['Simple']['LightGBM'].values) / len(final_results['MNAR']['None'])))
print("win ratio for MCAR none vs simple: {}".format(np.sum(final_results["MCAR"]['None']['LSAM'].values > final_results["MCAR"]['Simple']['LightGBM'].values) / len(final_results['MCAR']['None'])))
print("win ratio for MAR none vs simple: {}".format(np.sum(final_results["MAR"]['None']['LSAM'].values > final_results["MAR"]['Simple']['LightGBM'].values) / len(final_results['MAR']['None'])))

print("win ratio for MNAR none vs iterative: {}".format(np.sum(final_results["MNAR"]['None']['LSAM'].values > final_results["MNAR"]['Iterative']['LightGBM'].values) / len(final_results['MNAR']['None'])))
print("win ratio for MCAR none vs iterative: {}".format(np.sum(final_results["MCAR"]['None']['LSAM'].values > final_results["MCAR"]['Iterative']['LightGBM'].values) / len(final_results['MCAR']['None'])))
print("win ratio for MAR none vs iterative: {}".format(np.sum(final_results["MAR"]['None']['LSAM'].values > final_results["MAR"]['Iterative']['LightGBM'].values) / len(final_results['MAR']['None'])))

print("win ratio for MNAR none vs mice: {}".format(np.sum(final_results["MNAR"]['None']['LSAM'].values > final_results["MNAR"]['Miceforest']['LightGBM'].values) / len(final_results['MNAR']['None'])))
print("win ratio for MCAR none vs mice: {}".format(np.sum(final_results["MCAR"]['None']['LSAM'].values > final_results["MCAR"]['Miceforest']['LightGBM'].values) / len(final_results['MCAR']['None'])))
print("win ratio for MAR none vs mice: {}".format(np.sum(final_results["MAR"]['None']['LSAM'].values > final_results["MAR"]['Miceforest']['LightGBM'].values) / len(final_results['MAR']['None'])))
    
# GENERATE TABLES FOR UNCORRUPTED DATA
datasets = pd.read_csv('../results/openml/noncorrupted_tasklist.csv')
name_list_full = datasets.name.values
unres = pd.read_csv('../results/openml/benchmark_results_noncorrupt.csv')
name_list = set(unres.dsname)
print("number of datasets: ", len(name_list_full))
def table_maker2(metric="accuracy"):
    label = 'nll' if metric == 'nll' else 'acc'
    keys = itertools.product(["None"],["None", "Simple", "Iterative", "Miceforest"], ["LSAM", "LightGBM"])
    process_dict = {
        (" ", " ", "Dataset"):[],
        (" ", " ", "Metric"):[],
    }
    for key in keys:
        process_dict[key] = []

    key_sub = ["None", "Simple", "Iterative", "Miceforest"]
    for name in name_list:
        process_dict[(" ", " ", "Dataset")].append(name)
        process_dict[(" ", " ", "Metric")].append(metric)
            
        for key in key_sub:
            imp = key if key == 'None' else key.lower()
            if len(unres.loc[(unres.dsname==name) & (unres.imputation==imp)]) > 0:
                test_lsam = unres.loc[
                    (unres.dsname==name) & (unres.imputation==imp), 
                    "{}_lsam".format(label)].values[0]
                test_gbm = unres.loc[
                    (unres.dsname==name) & (unres.imputation==imp), 
                    "{}_gbm".format(label)].values[0]
                lsam = test_lsam
                gbm = test_gbm
            else:
                lsam = np.nan
                gbm = np.nan
            process_dict[("None", key, "LightGBM")].append(gbm)
            process_dict[("None", key, "LSAM")].append(lsam)
    return process_dict

process_dict = table_maker2("nll")
final_results = pd.DataFrame(process_dict)
datasets = final_results[" "][" "]["Dataset"].values
final_results.index = datasets
final_results.to_pickle('../results/openml/openml_nll_uncorrupted.pickle')
sub = final_results["None"] 
sub.fillna(np.inf, inplace=True)
s = sub.style.format(precision=2).highlight_min(axis=1, props="textbf:--rwrap;")
print(s.to_latex())


