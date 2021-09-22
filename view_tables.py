import pandas as pd
from os import listdir
from os.path import isfile, join
import numpy as np


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

def make_table(missingness, columns, indexes, imputation_list, path="results/openml"):
    result_files = [f for f in listdir(path) if isfile(join(path, f))]
    index_success = []
    prepped = []
    for idx in indexes.values:
        out = []
        try:
            # convenience function for filter of filenames
            def result_filter(filename, ds, rpts, mis, imp):
                splt = filename[:-7].split(",")
                try:
                    return ds == splt[0] and rpts == int(splt[1]) and str(mis) == splt[2] and str(imp) == splt[3].lower()
                except:
                    return False
            full = np.nan
            dropped = np.nan
            xgb = np.nan
            xgbdrop = np.nan
            for cidx in columns.values:
            # for imp in imputation_list:
                if cidx[0] == "Transformer":
                    xgboost = False
                else:
                    xgboost = True
                if cidx[2] == "Dropped":
                    impn = "None"
                else:
                    impn = cidx[2]

                try:
                    print(result_files)
                    fs = [f for f in result_files if result_filter(f, idx[0], 30, missingness, impn.lower())]
                    print("SUBSET", fs)
                    data = pd.read_pickle(join(path, fs[0]))
                    print(data.mean())
                    
                    if cidx[2] == "None":
                        print("MADE IT")
                        full = data[idx[1].lower()]["full"].values
                        dropped = data[idx[1].lower()]["drop"].values
                        xgb = data[idx[1].lower()]["xgboost"].values
                        xgbdrop = data[idx[1].lower()]["xgboost_drop"].values
                        if xgboost:
                            temp = np.mean(xgb)
                        else:
                            temp = np.mean(full)
                        if temp < 0.01 or np.abs(1 - temp) < 0.01:
                            val = "{:.2e}".format(temp)
                        else:
                            val = "{:.2f}".format(temp)
                        out.append(val)
                    elif cidx[2] == "Dropped":
                        if xgboost:
                            temp = np.mean(xgbdrop)
                        else:
                            temp = np.mean(dropped)
                        if temp < 0.01 or np.abs(1 - temp) < 0.01:
                            val = "{:.2e}".format(temp)
                        else:
                            val = "{:.2f}".format(temp)
                        out.append(val)
                    else:
                        if xgboost:
                            name = "xgboost"
                        else:
                            name = "full"
                        ser = data[idx[1].lower()][name].values
                        temp = np.mean(ser)
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
    print(out)
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
datasets_cat = ["profb", "cjs", "meta"]
datasets_reg = []
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
print(real_world.to_latex(multirow=True))

