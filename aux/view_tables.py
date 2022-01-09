import pandas as pd
from os import listdir
from os.path import isfile, join
import numpy as np
import itertools

## tables for latent space experiments
ed1 = pd.read_csv('../results/latent_space/Ensemble_Distances1_None.csv')
ed2 = pd.read_csv('../results/latent_space/Ensemble_Distances2_None.csv')
uat1 = pd.read_csv('../results/latent_space/UAT_Distances1_None.csv')
uat2 = pd.read_csv('../results/latent_space/UAT_Distances2_None.csv')
uat_missing = pd.read_csv('../results/latent_space/UAT_Distances1_missingness_None.csv')
uat_probs = pd.read_csv('../results/latent_space/UAT_drop_probs_missingness_None.csv')

# table 1 prep
rows = [7, 11, 3, 13, 14]
indexes = ['{x1}', '{x2}', '{x1, x2}', '{x3 (noise)}', '{x4 (signal)}']
colnames = [('Transformer', '{}'), ('Transformer', 'p-value'), ('Ensemble', '{}'), ('Ensemble', 'p-value')]
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
print(df.to_latex())

# table 2 prep
rows = [7, 11, 3, 15]
indexes = ['{x1}', '{x2}', '{x1, x2}', '{}']
colnames = [('Transformer', '+{noise}'), ('Transformer', '+{signal}'), ('Transformer', 'p-value'), ('Ensemble', '+{noise}'), ('Ensemble', '+{signal}'), ('Ensemble', 'p-value')]
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
print(df.to_latex())

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
print(df.to_latex())

# table 4 prep
rows = [0,1,2,3]
indexes = ['{x1}', '{x2}', '{x1, x2}', '{}']
colnames = ['0%', '20%', '40%', '60%', '80%', '99%']
result_strs = []

data = uat_probs.values[:, 1:]
for r in rows:
    row = []
    for c, col in enumerate(colnames):
        val = "{:.2f}".format(data[r, c])
        row.append(val)
    result_strs.append(row)

df = pd.DataFrame(result_strs, index=indexes, columns=colnames)
print(df.to_latex())

## tables for performance experiments

datasets = pd.read_csv('../results/openml/corrupted_tasklist.csv')
name_list_full = datasets.name.values

path = '../results/openml/'
result_files = [f for f in listdir(path) if isfile(join(path, f))]

def file_filter(filename, dsname, missingness, imputation):
    splt = filename[:-7].split(",")
    # print(dsname.lower() == splt[0].lower())
    #print(missingness.lower() == splt[2].lower())
    # print(imputation.lower() == splt[3].lower())
    try:
        return dsname.lower() == splt[0].lower() and missingness.lower() == splt[2].lower() and imputation.lower() == splt[3].lower()
    except:
        return False

def file_filter_name(filename):
    splt = filename[:-7].split(",")
    return splt[0]

name_list = set([file_filter_name(n) for n in result_files if 'True' in n])
print(name_list)
print("fraction of datasets run: ", len(name_list) / len(name_list_full))


def table_maker(metric="accuracy"):
    keys = itertools.product(["MCAR", "MNAR", "MAR"], ["None", "Simple", "Iterative", "MiceForest"], ["Transformer", "LightGBM"])
    process_dict = {
        (" ", " ", "Dataset"):[],
        (" ", " ", "Metric"):[],
        ("None", "None", "Transformer"):[],
        ("None", "None", "LightGBM"):[],
    }
    for key in keys:
        process_dict[key] = []

    key_sub = list(itertools.product(["MCAR", "MNAR", "MAR"], ["None", "Simple", "Iterative", "MiceForest"]))
    for name in name_list:
        process_dict[(" ", " ", "Dataset")].append(name)
        # get metric information
        try:
            s = [f for f in result_files if file_filter(f, name, "None", "None")]
            path = join('../results/openml/', str(s[0]))
            ds = pd.read_pickle(path)
            colnames = list(ds)
            process_dict[(" ", " ", "Metric")].append(metric)
        except Exception as e:
            metric = np.nan
            process_dict[(" ", " ", "Metric")].append(metric)
            
        # get baseline data
        try:
            s = [f for f in result_files if file_filter(f, name, "None", "None")]
            path = join('../results/openml/', str(s[0]))
            ds = pd.read_pickle(path)
            # print(ds)
            baseline_trans = np.nanmean(ds[metric]["full"].values)
            baseline_gbm = np.nanmean(ds[metric]["gbmoost"].values)
            process_dict[("None", "None", "Transformer")].append(baseline_trans)
            process_dict[("None", "None", "LightGBM")].append(baseline_gbm)
        except Exception as e:
            # print(e)
            process_dict[("None", "None", "Transformer")].append(np.nan)
            process_dict[("None", "None", "LightGBM")].append(np.nan)
        # get the same as above but for each missingness and imputation pattern
        for key in key_sub:
            try:
                s = [f for f in result_files if file_filter(f, name, key[0], key[1])]
                path = join('../results/openml/', str(s[0]))
                ds = pd.read_pickle(path)
                if metric in ["accuracy", "nll"]:
                    trans = (np.nanmean(ds[metric]["full"].values) - baseline_trans) / baseline_trans
                    gbm = (np.nanmean(ds[metric]["gbmoost"].values) - baseline_gbm) / baseline_gbm
                elif metric == "rmse":
                    trans = (np.nanmean(ds[metric]["full"].values) - baseline_trans) / baseline_trans
                    gbm = (np.nanmean(ds[metric]["gbmoost"].values) - baseline_gbm) / baseline_gbm
                process_dict[(key[0], key[1], "Transformer")].append(trans)
                process_dict[(key[0], key[1], "LightGBM")].append(gbm)
            except Exception as e:
                # print(e)
                process_dict[(key[0], key[1], "Transformer")].append(np.nan)
                process_dict[(key[0], key[1], "LightGBM")].append(np.nan)
    return process_dict

process_dict = table_maker("accuracy")
final_results = pd.DataFrame(process_dict).fillna(np.inf)
datasets = final_results[" "][" "]["Dataset"].values
final_results.index = datasets
print("None")
print(final_results["None"].to_latex())
print("MCAR")
print(final_results["MCAR"].to_latex())
print("MAR")
print(final_results["MAR"].to_latex())
print("MNAR")
print(final_results["MNAR"].to_latex())

print("win ratio baseline: {}".format(np.sum(final_results["None"]['None']['Transformer'].values > final_results["None"]['None']['LightGBM'].values) / len(final_results['None']['None'].dropna())))
print("win ratio for MNAR: {}".format(np.sum(final_results["MNAR"]['None']['Transformer'].values > final_results["MNAR"]['None']['LightGBM'].values) / len(final_results['MNAR']['None'].dropna())))
print("win ratio for MCAR: {}".format(np.sum(final_results["MCAR"]['None']['Transformer'].values > final_results["MCAR"]['None']['LightGBM'].values) / len(final_results['MCAR']['None'].dropna())))
print("win ratio for MAR: {}".format(np.sum(final_results["MAR"]['None']['Transformer'].values > final_results["MAR"]['None']['LightGBM'].values) / len(final_results['MAR']['None'].dropna())))

process_dict = table_maker("nll")
final_results = pd.DataFrame(process_dict)
datasets = final_results[" "][" "]["Dataset"].values
final_results.index = datasets
print("None")
print(final_results["None"].to_latex())
print("MCAR")
print(final_results["MCAR"].to_latex())
print("MAR")
print(final_results["MAR"].to_latex())
print("MNAR")
print(final_results["MNAR"].to_latex())
print("win ratio baseline: {}".format(np.sum(final_results["None"]['None']['Transformer'].values < final_results["None"]['None']['LightGBM'].values) / len(final_results['None']['None'].dropna())))
print("win ratio for MNAR: {}".format(np.sum(final_results["MNAR"]['None']['Transformer'].values < final_results["MNAR"]['None']['LightGBM'].values) / len(final_results['MNAR']['None'].dropna())))
print("win ratio for MCAR: {}".format(np.sum(final_results["MCAR"]['None']['Transformer'].values < final_results["MCAR"]['None']['LightGBM'].values) / len(final_results['MCAR']['None'].dropna())))
print("win ratio for MAR: {}".format(np.sum(final_results["MAR"]['None']['Transformer'].values > final_results["MAR"]['None']['LightGBM'].values) / len(final_results['MAR']['None'].dropna())))



