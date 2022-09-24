import numpy as np
import pandas as pd
import matplotlib
import itertools
# matplotlib.use('agg')
import os
import matplotlib.pyplot as plt

matplotlib.rcParams['font.size'] = 13
matplotlib.rcParams['font.family'] = "Noto Serif"
import operator
import math
from scipy.stats import wilcoxon
from scipy.stats import friedmanchisquare
import networkx
import Orange


def get_ranks(alpha=0.05, df_perf=None):
    """
    Determine the ranks after a friedman test
    """
    print(pd.unique(df_perf['classifier_name']))
    # count the number of tested datasets per classifier
    df_counts = pd.DataFrame({'count': df_perf.groupby(
        ['classifier_name']).size()}).reset_index()
    classifiers = list(df_counts['classifier_name'])
    # test the null hypothesis using friedman before doing a post-hoc analysis
    friedman_p_value = friedmanchisquare(*(
        np.array(df_perf.loc[df_perf['classifier_name'] == c]['accuracy'])
        for c in classifiers))[1]
    if friedman_p_value >= alpha:
        # then the null hypothesis over the entire classifiers cannot be rejected
        print('the null hypothesis over the entire classifiers cannot be rejected')
    # get the number of classifiers
    m = len(classifiers)
    # compute the average ranks to be returned (useful for drawing the cd diagram)
    # sort the dataframe of performances
    sorted_df_perf = df_perf.loc[df_perf['classifier_name'].isin(classifiers)]. \
        sort_values(['classifier_name', 'dataset_name'])
    # get the rank data
    rank_data = np.array(sorted_df_perf['accuracy']).reshape(m, -1)

    # create the data frame containg the accuracies
    df_ranks = pd.DataFrame(data=rank_data, index=np.sort(classifiers), columns=
    np.unique(sorted_df_perf['dataset_name']))

    # number of wins
    dfff = df_ranks.rank(ascending=False)
    print(dfff[dfff == 1.0].sum(axis=1))

    # average the ranks
    average_ranks = df_ranks.rank(ascending=False).mean(axis=1).sort_values(ascending=False)
    # return the p-values and the average ranks
    return average_ranks


def draw_cd_diagram(df_perf, title="Accuracy", labels=True, savename=None):
    average_ranks = get_ranks(alpha=0.05, df_perf=df_perf)
    cd = Orange.evaluation.compute_CD(average_ranks.values, len(df_perf), alpha="0.05", test="nemenyi") 
    Orange.evaluation.graph_ranks(average_ranks.values, average_ranks.index, cd=cd, width=4, textspace=0.5, reverse=True)
    # plt.title(title, y=0.9, x=0.5)
    if savename is not None:
        plt.savefig(savename,bbox_inches='tight', dpi=300)
    else:
        plt.show()

