import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

# calculate accuracy
for met in ["acc", "nll"]:
    results = pd.read_pickle("../results/openml/openml_{}.pickle".format(met))
    x = results["None"]["None"]["LSAM"].values
    y = results["None"]["None"]["LightGBM"].values
    d = x - y
    wins = np.sum(d > 0) if met == "acc" else np.sum(d < 0)
    print("{} wins: {}/{}".format(met, wins, len(d)))
    w, p = wilcoxon(d)
    print("{} two-tailed p-value: {}".format(met, p))
    w, p = wilcoxon(d, alternative="greater")
    print("{} one-tailed trans 'greater' p-value: {}".format(met, p))
    w, p = wilcoxon(d, alternative="less")
    print("{} one-tailed trans 'less' p-value: {}".format(met, p))
