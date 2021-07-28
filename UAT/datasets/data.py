import numpy as np
import pandas as pd
from scipy.io.arff import loadarff
# functions to prep datasets
import os

# define path locations relative to this file
dir_path = os.path.dirname(os.path.realpath(__file__))
thoracic_path = os.path.join(dir_path, "ThoracicSurgery.arff")
abalone_path = os.path.join(dir_path, "abalone.data")
bank_path = os.path.join(dir_path, "bank-additional/bank-additional.csv")

def spiral(N, missing="MAR", rng_key=0):
    rng = np.random.default_rng(rng_key)
    theta = np.sqrt(rng.uniform(0,1,N))*2*np.pi # np.linspace(0,2*pi,100)
    r_a = 2*theta + np.pi
    data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
    x_a = data_a + rng.standard_normal((N,2))

    r_b = -2*theta - np.pi
    data_b = np.array([np.cos(theta)*r_b, np.sin(theta)*r_b]).T
    x_b = data_b + rng.standard_normal((N,2))

    res_a = np.append(x_a, np.zeros((N,1)), axis=1)
    res_b = np.append(x_b, np.ones((N,1)), axis=1)

    res = np.append(res_a, res_b, axis=0)
    rng.shuffle(res)


    X = res[:, :2]
    y = res[:, 2]
    
    # create a noise column x3 and x4 transformation using x1, x2
    x3 = rng.standard_normal((N*2,1)) * 5
    x4 = (y).reshape((-1,1)) + rng.uniform(0,1,(N*2, 1)) # y with noise - should be highly informative...
    
    X = np.hstack([X, x3, x4])
    # create missingness mask
    if missing == "None":
        return X, y, (x_a, x_b)
    if missing == "MCAR":
        rand_arr = rng.uniform(0,1,(N*2, 3))
        nan_arr = np.where(rand_arr > 0.5, np.nan, 1.0)
        X[:, 1:] *= nan_arr
        return X, y, (x_a, x_b)
    if missing == "MAR":
        correction = np.where(X[:,0] > 0, 0.0, 1.0).reshape((-1,1))
        rand_arr = rng.uniform(0,1,(N*2, 3)) * correction
        # missingness is dependent on X1 which is observed, and only occurs if X1 is greater than 0
        nan_arr = np.where(rand_arr > 0.5, np.nan, 1.0)
        X[:, 1:] *= nan_arr
        return X, y, (x_a, x_b)
    if missing == "MNAR":
        correction = np.where(X[:,1:] > X[:,1:].mean(0, keepdims=True), 0.0, 1.0).reshape((-1,3))
        rand_arr = rng.uniform(0,1,(N*2, 3)) * correction
        # missingness is dependent on unobserved missing values
        nan_arr = np.where(rand_arr > 0.5, np.nan, 1.0)
        X[:, 1:] *= nan_arr
        return X, y, (x_a, x_b)

def thoracic(missing="MAR", rng_key=0, cols_miss=4, p=0.15):
    # import data
    rng = np.random.default_rng(rng_key)
    data, meta = loadarff(thoracic_path)
    d = pd.DataFrame(data)
    # convert categorical variables to integer encoding
    for name in meta.names():
        m = meta[name]
        if m[0] == 'nominal':
            l = list(m[1])
            d[name] = [l.index(x.decode('UTF-8')) for x in d[name].values]

    X = d.values[:, :-1]
    y = d.values[:, -1]
    
    if missing == "None":
        return X, y
    if missing == "MCAR":
        rand_arr = rng.uniform(0, 1, (X.shape[0], cols_miss))
        nan_arr = np.where(rand_arr < p, np.nan, 1.0)
        X[:, -cols_miss:] *= nan_arr
        return X, y
    if missing == "MAR":
        correction = np.where(X[:,1] > np.median(X[:,1]), 0.0, 1.0).reshape((-1,1))
        rand_arr = rng.uniform(0, 1, (X.shape[0], cols_miss)) * correction
        nan_arr = np.where(rand_arr > 1 - p, np.nan, 1.0)
        X[:, -cols_miss:] *= nan_arr
        return X, y
    if missing == "MNAR":
        correction = np.where(X[:,-cols_miss:] >=
                    np.median(X[:,-cols_miss:], axis=0, keepdims=True), 0.0, 1.0).reshape((-1,cols_miss))
        rand_arr = rng.uniform(0, 1, (X.shape[0], cols_miss)) * correction
        nan_arr = np.where(rand_arr > 1 - p, np.nan, 1.0)
        X[:, -cols_miss:] *= nan_arr
        return X, y

def abalone(missing="MAR", rng_key=0, cols_miss=4, p=0.15):
    rng = np.random.default_rng(rng_key)
    data = pd.read_csv(abalone_path, header=None)
    cat = list(data[0].unique())
    data[0] = [cat.index(i) for i in data[0].values]
    X = data.values[:, :-1]
    y = data.values[:, -1]
    unique = list(np.unique(y))
    y = np.array([unique.index(v) for v in y])
    if missing == "None":
        return X, y, len(unique)
    if missing == "MCAR":
        rand_arr = rng.uniform(0, 1, (X.shape[0], cols_miss))
        nan_arr = np.where(rand_arr < p, np.nan, 1.0)
        X[:, -cols_miss:] *= nan_arr
        return X, y, len(unique)
    if missing == "MAR":
        correction = np.where(X[:,1] > np.median(X[:,1]), 0.0, 1.0).reshape((-1,1))
        rand_arr = rng.uniform(0, 1, (X.shape[0], cols_miss)) * correction
        nan_arr = np.where(rand_arr > 1 - p, np.nan, 1.0)
        X[:, -cols_miss:] *= nan_arr
        return X, y, len(unique)
    if missing == "MNAR":
        correction = np.where(X[:,-cols_miss:] >=
                    np.median(X[:,-cols_miss:], axis=0, keepdims=True), 0.0, 1.0).reshape((-1,cols_miss))
        rand_arr = rng.uniform(0, 1, (X.shape[0], cols_miss)) * correction
        nan_arr = np.where(rand_arr > 1 - p, np.nan, 1.0)
        X[:, -cols_miss:] *= nan_arr
        return X, y, len(unique)

def banking():
    data = pd.read_csv(bank_path, sep=";")
    cont = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
    cat = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome', 'y']
    def lab_2_num(array):
        unique_list = [l for l in list(np.unique(array)) if l != "unknown"]
        return np.array([unique_list.index(l) if l != "unknown"  else np.nan for l in array])
    
    for c in cat:
        data[c] = lab_2_num(data[c].values)
    
    data = data[cont + cat]
    X = data.values[:, :-1]
    y = data.values[:, -1]
    return X, y
    