import pandas as pd
from main import draw_cd_diagram
import itertools
import os

missingness = "MNAR"
for missingness in ["MNAR", "MAR", "MCAR"]:
    raw = pd.read_pickle('../results/openml/openml_results.pickle').dropna()
    raw = raw.set_index((" ", " ", "Dataset"))
    # build 3 column input dataframe
    df_perf = {
        'classifier_name':[],
        'dataset_name':[],
        'accuracy':[]
    }

    print(raw.index)

    for r in list(itertools.product(["Transformer", "LightGBM"], ["None", "Simple", "Iterative", "MiceForest"], raw.index)):
        df_perf['classifier_name'].append(r[0] + " (" + r[1] + ")")
        df_perf['dataset_name'].append(r[2])
        df_perf['accuracy'].append(-float(raw.loc[r[2]][missingness][r[1]][r[0]]))

    df_perf = pd.DataFrame(df_perf)
    print(df_perf)
    draw_cd_diagram(df_perf=df_perf, title='Accuracy', labels=True)


    os.rename("./cd-diagram.png", "./{}/cd-diagram.png".format(missingness.lower()))
