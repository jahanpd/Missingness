import numpy as np
import pandas as pd
from scipy.io.arff import loadarff

# functions to prep datasets

def spiral(N):
    rng = np.random.default_rng(0)
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
    # plt.scatter(x_a[:,0],x_a[:,1])
    # plt.scatter(x_b[:,0],x_b[:,1])
    # plt.show()
    rng.shuffle(res)

    X = res[:, :2]
    y = res[:, 2]
    
    return X, y, (x_a, x_b)

def spiral_missing(N, missing="MAR", rng_key=0):
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
    # plt.scatter(x_a[:,0],x_a[:,1])
    # plt.scatter(x_b[:,0],x_b[:,1])
    # plt.show()
    rng.shuffle(res)


    X = res[:, :2]
    y = res[:, 2]
    
    # create a noise column x3 and x4 transformation using x1, x2
    x3 = rng.standard_normal((N*2,1)) * 5
    x4 = (X[:, 0] + X[:, 1] + X[:, 0]**2 + X[:, 1]**2 + X[:, 0] * X[:, 1]).reshape((-1,1))  # polynomial transformation - should be highly informative
    
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
    if missing == "NMAR":
        correction = np.where(X[:,1:] > X[:,1:].mean(0, keepdims=True), 0.0, 1.0).reshape((-1,3))
        rand_arr = rng.uniform(0,1,(N*2, 3)) * correction
        # missingness is dependent on unobserved missing values
        nan_arr = np.where(rand_arr > 0.5, np.nan, 1.0)
        X[:, 1:] *= nan_arr
        return X, y, (x_a, x_b)

def thoracic(path='ThoracicSurgery.arff'):
    # import data
    data, meta = loadarff(path)
    d = pd.DataFrame(data)
    # convert categorical variables to integer encoding
    for name in meta.names():
        m = meta[name]
        if m[0] == 'nominal':
            l = list(m[1])
            d[name] = [l.index(x.decode('UTF-8')) for x in d[name].values]

    X = d.values[:, :-1]
    y = d.values[:, -1]
    return X, y

def abalone(path='abalone.data'):
    data = pd.read_csv(path, header=None)
    cat = list(data[0].unique())
    data[0] = [cat.index(i) for i in data[0].values]
    X = data.values[:, :-1]
    y = data.values[:, -1]
    return X, y