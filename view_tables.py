import pandas as pd
from os import listdir
from os.path import isfile, join
import numpy as np
import itertools

## tables for latent space experiments
ed1 = pd.read_csv('results/latent_space/Ensemble_Distances1_None.csv')
ed2 = pd.read_csv('results/latent_space/Ensemble_Distances2_None.csv')
uat1 = pd.read_csv('results/latent_space/UAT_Distances1_None.csv')
uat2 = pd.read_csv('results/latent_space/UAT_Distances2_None.csv')
uat_missing = pd.read_csv('results/latent_space/UAT_Distances1_missingness_None.csv')
uat_probs = pd.read_csv('results/latent_space/UAT_drop_probs_missingness_None.csv')

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
# print(df.to_latex(multirow=True))

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
# print(df.to_latex(multirow=True))

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
# print(df.to_latex(multirow=True))

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
# print(df.to_latex(multirow=True))

## tables for performance experiments

datasets = pd.read_csv('./results/openml/corrupted_tasklist.csv')
name_list_full = datasets.name.values

path = './results/openml/'
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


keys = itertools.product(["MCAR", "MNAR"], ["None", "Simple", "Iterative", "MiceForest"], ["Transformer", "LightGBM"])
process_dict = {
    (" ", " ", "Dataset"):[],
    (" ", " ", "Metric"):[],
    ("None", "None", "Transformer"):[],
    ("None", "None", "LightGBM"):[],
}
for key in keys:
    process_dict[key] = []

key_sub = list(itertools.product(["MCAR", "MNAR"], ["None", "Simple", "Iterative", "MiceForest"]))
for name in name_list:
    process_dict[(" ", " ", "Dataset")].append(name)
    # get metric information
    try:
        s = [f for f in result_files if file_filter(f, name, "None", "None")]
        path = join('./results/openml/', str(s[0]))
        ds = pd.read_pickle(path)
        colnames = list(ds)
        metric = "accuracy" if len(colnames) > 5 else "rmse"
        process_dict[(" ", " ", "Metric")].append(metric)
    except:
        metric = np.nan
        process_dict[(" ", " ", "Metric")].append(metric)
        
    # get baseline data
    try:
        s = [f for f in result_files if file_filter(f, name, "None", "None")]
        path = join('./results/openml/', str(s[0]))
        ds = pd.read_pickle(path)
        # print(ds)
        baseline_trans = np.mean(ds[metric]["full"].values)
        baseline_gbm = np.mean(ds[metric]["gbmoost"].values)
        process_dict[("None", "None", "Transformer")].append(baseline_trans)
        process_dict[("None", "None", "LightGBM")].append(baseline_gbm)
    except Exception as e:
        print(e)
        process_dict[("None", "None", "Transformer")].append(np.nan)
        process_dict[("None", "None", "LightGBM")].append(np.nan)
    # get the same as above but for each missingness and imputation pattern
    for key in key_sub:
        try:
            s = [f for f in result_files if file_filter(f, name, key[0], key[1])]
            path = join('./results/openml/', str(s[0]))
            ds = pd.read_pickle(path)
            if metric in ["accuracy", "NLL"]:
                trans = (np.mean(ds[metric]["full"].values) - baseline_trans) / baseline_trans
                gbm = (np.mean(ds[metric]["gbmoost"].values) - baseline_gbm) / baseline_gbm
            elif metric == "rmse":     
                trans = (np.mean(ds[metric]["full"].values) - baseline_trans) / baseline_trans
                gbm = (np.mean(ds[metric]["gbmoost"].values) - baseline_gbm) / baseline_gbm
            process_dict[(key[0], key[1], "Transformer")].append(trans)
            process_dict[(key[0], key[1], "LightGBM")].append(gbm)
        except Exception as e:
            print(e)
            process_dict[(key[0], key[1], "Transformer")].append(np.nan)
            process_dict[(key[0], key[1], "LightGBM")].append(np.nan)

print(process_dict)
for key in process_dict.keys():
    print(key, len(process_dict[key]))
final_results = pd.DataFrame(process_dict)
final_results.to_pickle('./results/openml/openml_results.pickle')
print(final_results)
print("win ratio baseline: {}".format(np.sum(final_results["None"]['None']['Transformer'].values < final_results["None"]['None']['LightGBM'].values) / len(final_results)))
print("win ratio for MNAR: {}".format(np.sum(final_results["MNAR"]['None']['Transformer'].values < final_results["MNAR"]['None']['LightGBM'].values) / len(final_results)))
asd


test_ = pd.read_pickle('/home/jahan/missing/results/openml/abalone,0.000000,None,None,True,True.pickle')
print(test_.mean())
test_ = pd.read_pickle('/home/jahan/missing/results/openml/abalone,0.642724,MCAR,None,True,True.pickle')
print(test_.mean())
test_ = pd.read_pickle('/home/jahan/missing/results/openml/abalone,0.000000,MCAR,simple,True,True.pickle')
print(test_.mean())
test_ = pd.read_pickle('/home/jahan/missing/results/openml/abalone,0.000000,MCAR,iterative,True,True.pickle')
print(test_.mean())
test_ = pd.read_pickle('/home/jahan/missing/results/openml/abalone,0.000000,MCAR,miceforest,True,True.pickle')
print(test_.mean())
asd

def make_table(missingness, columns, indexes, imputation_list, path="results/openml"):
    result_files = [f for f in listdir(path) if isfile(join(path, f))]
    index_success = []
    prepped = []
    for idx in indexes.values:
        out = []
        try:
            # convenience function for filter of filenames
            def result_filter(filename, ds, mis, imp):
                splt = filename[:-7].split(",")
                try:
                    return ds == splt[0]  and str(mis) == splt[2] and str(imp) == splt[3].lower()
                except:
                    return False
            full = np.nan
            dropped = np.nan
            gbm = np.nan
            gbmdrop = np.nan
            for cidx in columns.values:
            # for imp in imputation_list:
                if cidx[0] == "Transformer":
                    gbmoost = False
                else:
                    gbmoost = True
                if cidx[2] == "Dropped":
                    impn = "None"
                else:
                    impn = cidx[2]

                try:
                    # print(result_files)
                    fs = [f for f in result_files if result_filter(f, idx[0],  missingness, impn.lower())]
                    # print("SUBSET", fs)
                    data = pd.read_pickle(join(path, fs[0]))
                    # print(data.mean())
                    
                    if cidx[2] == "None":
                        # print("MADE IT")
                        full = data[idx[1].lower()]["full"].values
                        dropped = data[idx[1].lower()]["drop"].values
                        gbm = data[idx[1].lower()]["gbmoost"].values
                        gbmdrop = data[idx[1].lower()]["gbmoost_drop"].values
                        if gbmoost:
                            temp = np.mean(gbm[~np.isnan(np.array([float(i) for i in gbm]))])
                        else:
                            temp = np.mean(full[~np.isnan(np.array([float(i) for i in full]))])
                        if temp < 0.01 or np.abs(1 - temp) < 0.01:
                            val = "{:.2e}".format(temp)
                        else:
                            val = "{:.2f}".format(temp)
                        out.append(val)
                    elif cidx[2] == "Dropped":
                        if gbmoost:
                            temp = np.mean(gbmdrop[~np.isnan(np.array([float(i) for i in gbmdrop]))])
                        else:
                            temp = np.mean(dropped[~np.isnan(np.array([float(i) for i in dropped]))])
                        if temp < 0.01 or np.abs(1 - temp) < 0.01:
                            val = "{:.2e}".format(temp)
                        else:
                            val = "{:.2f}".format(temp)
                        out.append(val)
                    else:
                        if gbmoost:
                            name = "gbmoost"
                        else:
                            name = "full"
                        ser = data[idx[1].lower()][name].values
                        temp = np.mean(ser[~np.isnan(np.array([float(i) for i in ser]))])
                        if temp < 0.01 or np.abs(1 - temp) < 0.01:
                            val = "{:.2e}".format(temp)
                        else:
                            val = "{:.2f}".format(temp)
                        out.append(val)
                except Exception as e:
                    print(e)
                    out.append("")
            prepped.append(out)
            index_success.append(idx)
        except Exception as e:
            print(e)
    # print(out)
    multiindex = pd.MultiIndex.from_tuples(index_success, names=["Dataset", "Metric"])
    df = pd.DataFrame(prepped, index=multiindex, columns=columns)

    return df



## synthetic and controlled corrupted missingness
# datasets_cat = ["profb"]
# datasets_reg = []
# # missingness = ["None", "MCAR", "MAR", "MNAR"]
# imputation = ["None", "Dropped", "Simple", "Iterative", "Miceforest", "XGBoost"]
# metrics_cat = ["Accuracy", "NLL"]
# metrics_reg = ["RMSE"]

# multiindex_cat = pd.MultiIndex.from_product([datasets_cat, metrics_cat], names=["Dataset", "Imputation"])
# multiindex_reg = pd.MultiIndex.from_product([datasets_reg, metrics_reg], names=["Dataset", "Imputation"])
# multiindex = pd.MultiIndex.from_tuples(list(multiindex_cat.values) + list(multiindex_reg.values), names=["Dataset", "Imputation"])
# print(multiindex)

colnames = [
    ('Transformer',' ','None'), ('Transformer',' ','Dropped'),('Transformer', 'Imputation', 'Simple'), ('Transformer', 'Imputation', 'Iterative'), ('Transformer', 'Imputation', 'Miceforest'),
    ('XGBoost',' ','None'), ('XGBoost',' ','Dropped'),('XGBoost', 'Imputation', 'Simple'), ('XGBoost', 'Imputation', 'Iterative'), ('XGBoost', 'Imputation', 'Miceforest')
    ]
colmulti = pd.MultiIndex.from_tuples(colnames, names=["", "", ""]) 
print(colmulti)

# mcar = make_table('mcar', colmulti, multiindex, imputation)
# mar = make_table('mar', colmulti, multiindex, imputation)
# mnar = make_table('mnar', colmulti, multiindex, imputation)

# print(mcar.to_latex(multirow=True))
# print(mar.to_latex(multirow=True))
# print(mnar.to_latex(multirow=True))

## unknown missingness pattern
datasets = pd.read_csv("results/openml/corrupted_tasklist.csv")
datasets_cat = datasets.name.values[datasets.task_type.values == "Supervised Classification"]
datasets_reg = datasets.name.values[datasets.task_type.values == "Supervised Regression"]
# datasets_cat = ["sick", "hypothyroid", "ipums_la_99-small"]
# datasets_reg = ["Moneyball", "dating_profile", "colleges", "employee_salaries"]
# missingness = ["None", "MCAR", "MAR", "MNAR"]
imputation = ["None", "Dropped", "Simple", "Iterative", "Miceforest", "XGBoost"]
metrics_cat = ["Accuracy", "NLL"]
metrics_reg = ["RMSE"]

multiindex_cat = pd.MultiIndex.from_product([datasets_cat, metrics_cat], names=["Dataset", "Imputation"])
multiindex_reg = pd.MultiIndex.from_product([datasets_reg, metrics_reg], names=["Dataset", "Imputation"])
multiindex = pd.MultiIndex.from_tuples(list(multiindex_cat.values) + list(multiindex_reg.values), names=["Dataset", "Imputation"])
print(multiindex)

real_world = make_table('None', colmulti, multiindex, imputation)
print(real_world)
real_world.to_csv("results/openml/results.csv")
# print(real_world.to_latex(multirow=True))

