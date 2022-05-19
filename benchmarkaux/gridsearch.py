import numpy as np
import pandas as pd
import itertools
from .openmlrun import run


# missing= "MNAR"
missing= None

def gridsearchattn(searchspace, k, row, args):
    """
    Args:
        searchspace: [("hyperparam", [vals...])...]
        k: number of k folds for each combination
        row: dataset information
        args: input args from command line
    """
    hpnames = [x[0] for x in searchspace]
    prod = list(itertools.product(*[x[1] for x in searchspace]))
    print("Combinations in ss {}, total runs: {}".format(
        len(prod), len(prod)*k))
    results = []
    for i, hps in enumerate(prod):
        print("COMBINATION {} of {}".format(i, len(prod)))
        hp_dict = {}
        for hn, hs in zip(hpnames, hps):
            hp_dict[hn] = hs
        output, _ = run(
                    dataset=row[0],
                    task=row[1],
                    missing=missing,
                    train_complete=args.train_complete,
                    test_complete=args.test_complete,
                    imputation=None,
                    trans_params = hp_dict,
                    gbm_params = None,
                    corrupt=args.corrupt,
                    row_data=row,
                    folds=k
                    )
        loss = np.nanmean(output["nll"]["attn"].values)
        acc = np.nanmean(output["accuracy"]["attn"].values)
        print(output.mean())
        results.append((loss, acc, hp_dict))
    results.sort(key= lambda x: x[0])
    print(results)
    best = results[0]
    return {
        "best": best,
        "results": results
    }


def gridsearchgbm(searchspace, k, row, args):
    """
    Args:
        searchspace: [("hyperparam", [vals...])...]
        k: number of k folds for each combination
        row: dataset information
        args: input args from command line
    """
    if row[1] == "Supervised Classification":
        objective = 'softmax'
    else:
        objective = 'regression'
        resample=False
    hpnames = [x[0] for x in searchspace]
    prod = list(itertools.product(*[x[1] for x in searchspace]))
    print("Combinations in ss {}, total runs: {}".format(
        len(prod), len(prod)*k))
    results = []
    for i, hps in enumerate(prod):
        print("COMBINATION {} of {}".format(i, len(prod)))
        hp_dict = {}
        for hn, hs in zip(hpnames, hps):
            hp_dict[hn] = hs
        hp_dict['objective']=objective
        output, _ = run(
                    dataset=row[0],
                    task=row[1],
                    missing=None,
                    train_complete=args.train_complete,
                    test_complete=args.test_complete,
                    imputation=None,
                    trans_params = None,
                    gbm_params = hp_dict,
                    corrupt=args.corrupt,
                    row_data=row,
                    folds=k
                    )
        loss = np.nanmean(output["nll"]["gbm"].values)
        acc = np.nanmean(output["accuracy"]["gbm"].values)
        print(output.mean())
        results.append((loss, acc, hp_dict))
    results.sort(key= lambda x: x[0])
    print(results)
    best = results[0]
    return {
        "best": best,
        "results": results
    }
