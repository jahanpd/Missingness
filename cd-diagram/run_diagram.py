import pandas as pd
from main import draw_cd_diagram
import itertools
import os

for missingness in ["MNAR", "MAR", "MCAR"]:
    raw = pd.read_pickle('../results/openml/openml_acc.pickle').dropna()
    raw = raw.set_index((" ", " ", "Dataset"))
    # build 3 column input dataframe
    df_perf = {
        'classifier_name':[],
        'dataset_name':[],
        'accuracy':[]
    }

    print(raw.index)

    for r in list(itertools.product(["Transformer", "LightGBM"], ["None", "Simple", "Iterative", "MiceForest"], raw.index)):
        name = "LSAM" if r[0] == "Transformer" else "LightGBM"
        if r[1] == "None":
            df_perf['classifier_name'].append(name)
        else:
            df_perf['classifier_name'].append(name + "+" + r[1])
        df_perf['dataset_name'].append(r[2])
        df_perf['accuracy'].append(-float(raw.loc[r[2]][missingness][r[1]][r[0]]))

    df_perf = pd.DataFrame(df_perf)
    print(df_perf)
    draw_cd_diagram(df_perf=df_perf, title='Change in Accuracy', labels=True)


    os.rename("./cd-diagram.png", "./{}/cd-diagram-acc.png".format(missingness.lower()))

for missingness in ["MNAR", "MAR", "MCAR"]:
    raw = pd.read_pickle('../results/openml/openml_acc.pickle').dropna()
    raw = raw.set_index((" ", " ", "Dataset"))
    # build 3 column input dataframe
    df_perf = {
        'classifier_name':[],
        'dataset_name':[],
        'accuracy':[]
    }

    print(raw.index)

    for r in list(itertools.product(["Transformer"], ["None", "Simple", "Iterative", "MiceForest"], raw.index)):
        name = "LSAM" if r[0] == "Transformer" else "LightGBM"
        if r[1] == "None":
            df_perf['classifier_name'].append(name)
        else:
            df_perf['classifier_name'].append(name + "+" + r[1])
        df_perf['dataset_name'].append(r[2])
        df_perf['accuracy'].append(-float(raw.loc[r[2]][missingness][r[1]][r[0]]))

    df_perf = pd.DataFrame(df_perf)
    print(df_perf)
    draw_cd_diagram(df_perf=df_perf, title='Change in Accuracy', labels=True)


    os.rename("./cd-diagram.png", "./{}/cd-diagram-lsam.png".format(missingness.lower()))

for missingness in ["MNAR", "MAR", "MCAR"]:
    raw = pd.read_pickle('../results/openml/openml_acc.pickle').dropna()
    raw = raw.set_index((" ", " ", "Dataset"))
    # build 3 column input dataframe
    df_perf = {
        'classifier_name':[],
        'dataset_name':[],
        'accuracy':[]
    }

    print(raw.index)

    for r in list(itertools.product(["Transformer", "LightGBM"], ["None", "Simple", "Iterative", "MiceForest"], raw.index)):
        if (r[0] == "Transformer" and r[1] == "None") or (r[0] == "LightGBM" and r[1] != "None"):
            name = "LSAM" if r[0] == "Transformer" else "LightGBM"
            if r[1] == "None":
                df_perf['classifier_name'].append(name)
            else:
                df_perf['classifier_name'].append(name + "+" + r[1])
            df_perf['dataset_name'].append(r[2])
            df_perf['accuracy'].append(-float(raw.loc[r[2]][missingness][r[1]][r[0]]))

    df_perf = pd.DataFrame(df_perf)
    print(df_perf)
    draw_cd_diagram(df_perf=df_perf, title='Change in Accuracy', labels=True)


    os.rename("./cd-diagram.png", "./{}/cd-diagram-lgbm.png".format(missingness.lower()))

for missingness in ["MNAR", "MAR", "MCAR"]:
    raw = pd.read_pickle('../results/openml/openml_nll.pickle').dropna()
    raw = raw.set_index((" ", " ", "Dataset"))
    # build 3 column input dataframe
    df_perf = {
        'classifier_name':[],
        'dataset_name':[],
        'accuracy':[]
    }

    print(raw.index)

    for r in list(itertools.product(["Transformer", "LightGBM"], ["None", "Simple", "Iterative", "MiceForest"], raw.index)):
        if (r[0] == "Transformer" and r[1] == "None") or (r[0] == "LightGBM" and r[1] != "None"):
            name = "LSAM" if r[0] == "Transformer" else "LightGBM"
            if r[1] == "None":
                df_perf['classifier_name'].append(name)
            else:
                df_perf['classifier_name'].append(name + "+" + r[1])
            df_perf['dataset_name'].append(r[2])
            df_perf['accuracy'].append(-float(raw.loc[r[2]][missingness][r[1]][r[0]]))

    df_perf = pd.DataFrame(df_perf)
    print(df_perf)
    draw_cd_diagram(df_perf=df_perf, title='Change in Accuracy', labels=True)


    os.rename("./cd-diagram.png", "./{}/cd-diagram-nll.png".format(missingness.lower()))
