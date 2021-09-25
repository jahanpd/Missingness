import numpy as np
import pandas as pd
from scipy.io.arff import loadarff
# functions to prep dataset
import sklearn.datasets as skdata
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
import miceforest as mf
# import tensorflow_datasets
import os
import openml

def get_list(missing, sample, key=42, test=lambda x, m: x > m):
    rng = np.random.default_rng(key)
    # to generate list of datasets with high proportion of missingness
    datasets = openml.datasets.list_datasets(output_format='dataframe')

    class_tasks = datasets[datasets.NumberOfClasses > 0.0]
    reg_tasks = datasets[datasets.NumberOfClasses == 0.0]
    # where there are greater than 50% rows with missing data
    class_subset = class_tasks[test(class_tasks.NumberOfInstancesWithMissingValues / class_tasks.NumberOfInstances, missing)]
    reg_subset = reg_tasks[test(reg_tasks.NumberOfInstancesWithMissingValues / reg_tasks.NumberOfInstances, missing)]
    # limit number of features to less than 500 for tractability
    class_subset = class_subset[
        (class_subset.NumberOfFeatures < 100) & 
        (class_subset.NumberOfInstances < 100000) & 
        (class_subset.NumberOfInstances > 1000) &
        (class_subset.NumberOfInstancesWithMissingValues != class_subset.NumberOfMissingValues)
        ].drop_duplicates(subset=["name"])
    reg_subset = reg_subset[
        (reg_subset.NumberOfFeatures < 100) & 
        (reg_subset.NumberOfInstances < 100000) & 
        (reg_subset.NumberOfInstances > 1000) &
        (reg_subset.NumberOfInstancesWithMissingValues != reg_subset.NumberOfMissingValues)
        ].drop_duplicates(subset=["name"])
    
    # test if datasets are gettable and add to list
    did_list_class = []
    for did in class_subset.did.values:
        try:
            print(did_list_class, len(class_subset))
            # did = rng.choice(class_subset.did.values, 1)
            ds = openml.datasets.get_dataset(dataset_id=int(did))
            X, y, categorical_indicator, attribute_names = ds.get_data(target = ds.default_target_attribute)
            did_list_class.append(int(did))
            did_list_class = list(set(did_list_class))
        except Exception as e:
            print(e)
            pass
    did_list_reg = []
    for did in reg_subset.did.values:
        try:
            print(did_list_reg, len(reg_subset))
            # did = rng.choice(reg_subset.did.values, 1)
            ds = openml.datasets.get_dataset(dataset_id=int(did))
            X, y, categorical_indicator, attribute_names = ds.get_data(target = ds.default_target_attribute)
            did_list_reg.append(int(did))
            did_list_reg = list(set(did_list_reg))
        except Exception as e:
            print(e)
            pass
    class_subset = class_subset[class_subset['did'].isin(did_list_class)]
    reg_subset = reg_subset[reg_subset['did'].isin(did_list_reg)]
    df = pd.concat([class_subset, reg_subset])
    return pd.concat([class_subset, reg_subset])

# define path locations relative to this file
dir_path = os.path.dirname(os.path.realpath(__file__))
thoracic_path = os.path.join(dir_path, "thoracic/ThoracicSurgery.arff")
abalone_path = os.path.join(dir_path, "abalone/abalone.data")
bank_path = os.path.join(dir_path, "banking/bank-additional.csv")
anneal_path_train = os.path.join(dir_path, "anneal/anneal.data")
anneal_path_test = os.path.join(dir_path, "anneal/anneal.test")


# convenience imputation functions
def simple(train, valid, test, dtypes=None, all_cat=False):
    if dtypes is None:
        if all_cat:
            imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        else:
            imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp.fit(train)
        train = imp.transform(train)
        if valid is not None:
            valid = imp.transform(valid)
        if test is not None:    
            test = imp.transform(test)
    if dtypes is not None:
        cont = np.array(dtypes) == 0
        cat = np.array(dtypes) == 1
        imp1 = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp2 = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        imp1.fit(train[:, cont])
        imp2.fit(train[:, cat])
        train[:, cont] = imp1.transform(train[:, cont])
        train[:, cat] = imp2.transform(train[:, cat])
        if valid is not None:
            valid[:, cont] = imp1.transform(valid[:, cont])
            valid[:, cat] = imp2.transform(valid[:, cat])
        if test is not None:    
            test[:, cont] = imp1.transform(test[:, cont])
            test[:, cat] = imp2.transform(test[:, cat])
    return train, valid, test

def iterative(train, rng_key, dtypes=None, valid=None, test=None):
    imp = IterativeImputer(max_iter=10, random_state=rng_key)
    imp.fit(train)
    train = imp.transform(train)
    if valid is not None:
            valid = imp.transform(valid)
    if test is not None:    
        test = imp.transform(test)
    return train, valid, test

def miceforest(train, rng_key, dtypes=None, valid=None, test=None):
    colnames = [str(i) for i in range(train.shape[1])]
    df = pd.DataFrame(train, columns=colnames)
    kernel = mf.MultipleImputedKernel(
                df,
                datasets=20,
                save_all_iterations=True,
                random_state=10,
                mean_match_candidates=0
                )
    kernel.mice(3)
    train = kernel.complete_data(0).values
    if valid is not None:
        valid_imp = kernel.impute_new_data(
            new_data=pd.DataFrame(valid, columns=colnames))
        valid = valid_imp.complete_data(0).values
    if test is not None:
        test_imp = kernel.impute_new_data(
            new_data=pd.DataFrame(test, columns=colnames))
        test = test_imp.complete_data(0).values
    return train, valid, test

# dataset generating functions
def spiral(
        N,
        missing=None,
        imputation=None,  # one of none, simple, iterative, miceforest
        train_complete=False,
        test_complete=True,
        split=0.33,
        rng_key=0,
        p=0.5,
        cols_miss=1
    ):
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


    X_ = res[:, :2]
    y = res[:, 2]
    
    # create a noise column x3 and x4 transformation using x1, x2
    x3 = rng.standard_normal((N*2,1)) * 5
    x4 = (y).reshape((-1,1)) + rng.uniform(0,1,(N*2, 1)) # y with noise - should be highly informative...
    
    X_ = np.hstack([X_, x3, x4])
    
    key = rng.integers(9999)
    if missing is None:
        train_complete = True
        test_complete = True

    if train_complete and test_complete:
        X = X_
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=key)
        key = rng.integers(9999)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.33, random_state=key)

    elif train_complete and not test_complete: # TRAIN COMPLETE IS TRUE AND TEST COMPLETE IS FALSE
        X_train, X, y_train, y_test = train_test_split(X_, y, test_size=0.33, random_state=key)
        key = rng.integers(9999)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.33, random_state=key)
    
    elif not train_complete and test_complete:
        X, X_test, y_train, y_test = train_test_split(X_, y, test_size=0.33, random_state=key)
    
    elif not train_complete and not test_complete:
        X = X_

    # create missingness mask
    cols = X.shape[1]
    if missing == "MAR":
        cols_miss = np.minimum(cols - 1, cols_miss) # clip cols missing 
        q = rng.uniform(0.3,0.7,(cols-1,))
        corrections = []
        for col in range(cols-1):
            correction = X[:,col] > np.quantile(X[:,col], q[col], keepdims=True) # dependency on each x
            corrections.append(correction)
        corrections = np.concatenate(corrections)
        corrections = np.where(corrections, 0.0, 1.0).reshape((-1,cols - 1))
        print(corrections.shape, X.shape)
        rand_arr = rng.uniform(0,1,(X.shape[0], cols - 1)) * corrections
        nan_arr = np.where(rand_arr > (1-p), np.nan, 1.0)
        X[:, -cols_miss:] *= nan_arr[:, -cols_miss:]  # dependency is shifted to the left, therefore MAR

    if missing == "MNAR":
        cols_miss = np.minimum(cols, cols_miss) # clip cols missing 
        q = rng.uniform(0.3,0.7,(cols,))
        corrections = []
        for col in range(cols):
            correction = X[:,col] > np.quantile(X[:,col], q[col], keepdims=True) # dependency on each x
            corrections.append(correction)
        corrections = np.concatenate(corrections)
        corrections = np.where(corrections, 0.0, 1.0).reshape((-1,cols))
        rand_arr = rng.uniform(0,1,(X.shape[0], cols)) * corrections
        nan_arr = np.where(rand_arr > (1-p), np.nan, 1.0)
        X[:, -cols_miss:] *= nan_arr[:, -cols_miss:]  # dependency is not shifted to the left, therefore MNAR

    if type(missing) == float or missing == "MCAR":
        cols_miss = np.minimum(cols, cols_miss) # clip cols missing
        if type(missing) == float: p = missing
        rand_arr = rng.uniform(0,1,(X.shape[0], cols_miss))
        nan_arr = np.where(rand_arr < p, np.nan, 1.0)
        X[:, -cols_miss:] *= nan_arr

    if type(missing) == tuple and missing[1] == "MNAR":
        correction1 = X[:,-1:] < np.quantile(X[:,-1:], 0.2, keepdims=True) # dependency on x4 MNAR
        correction2 = X[:,:1] < np.quantile(X[:,:1], 0.2, keepdims=True) # dependency on x1 MAR
        correction3 = X[:,1:2] < np.quantile(X[:,1:2], 0.5, keepdims=True) # dependency on x2 MAR
        correction = (correction1 | correction2) | correction3
        correction = np.where(correction, 0.0, 1.0).reshape((-1,1))  # dependency on x4
        rand_arr = rng.uniform(0,1,(X.shape[0], 1)) * correction
        # missingness is dependent on unobserved missing values
        nan_arr = np.where(rand_arr > (1 - missing[0]), np.nan, 1.0)
        X[:, -1:] *= nan_arr
    
    # generate train, validate, test datasets and impute training 
    key = rng.integers(9999)
    if train_complete and test_complete:
        pass

    elif train_complete and not test_complete:
        X_test = X
    
    elif not train_complete and test_complete:
        X_train = X
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.33, random_state=key)
    
    elif not train_complete and not test_complete:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=key)
        key = rng.integers(9999)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.33, random_state=key)

    # missingness diagnostics
    # diagnostics = {"X_train":{}, "X_valid":{}, "X_test":{}}
    # diagnostics["X_train"]["cols"] = np.isnan(X_train).sum(0) / X_train.shape[0]
    # diagnostics["X_train"]["rows"] = np.any(np.isnan(X_train), axis=1).sum() / X_train.shape[0]
    # diagnostics["X_valid"]["cols"] = np.isnan(X_valid).sum(0) / X_valid.shape[0]
    # diagnostics["X_valid"]["rows"] = np.any(np.isnan(X_valid), axis=1).sum() / X_valid.shape[0]
    # diagnostics["X_test"]["cols"] = np.isnan(X_test).sum(0) / X_test.shape[0]
    # diagnostics["X_test"]["rows"] = np.any(np.isnan(X_test), axis=1).sum() / X_test.shape[0]
    # print(diagnostics)

    # perform desired imputation strategy
    if imputation == "simple" and missing is not None:
        X_train, X_valid, X_test = simple(
            X_train,
            dtypes=None,
            valid=X_valid,
            test=X_test)
    
    key = rng.integers(9999)
    if imputation == "iterative" and missing is not None:
        X_train, X_valid, X_test = iterative(
            X_train,
            key,
            dtypes=None,
            valid=X_valid,
            test=X_test)
    
    key = rng.integers(9999)
    if imputation == "miceforest" and missing is not None:
        if test_complete:
            test_input = None
        else:
            test_input = X_test
        X_train, X_valid, test_input = miceforest(
            X_train,
            int(key),
            dtypes=None,
            valid=X_valid,
            test=test_input)
        if test_complete:
            X_test = X_test
        else:
            X_test = test_input

    return X_train, X_valid, X_test, y_train, y_valid, y_test, (x_a, x_b), 2

# convenience function for prepping openML datasets
def prepOpenML(did, task):
    # takes did, target colname and task type string
    # returns prepared X, y numpy arrays
    ds = openml.datasets.get_dataset(dataset_id=int(did))
    X, y, categorical_indicator, attribute_names = ds.get_data(target = ds.default_target_attribute)

    cat_list = []
    for cat, name in zip(categorical_indicator, list(X)):
        if cat:
            cat_list.append(name)
            d = X[name].values.astype(str)
            vals = np.unique(d[d != 'nan'])
            vals = list(vals)
            int_enc = [np.nan if j == 'nan' else vals.index(j) for j in d]
            X[name] = int_enc
    
    # get rid of features that are objects but not in categorical indicator - these are usually unhelpful columns like names etc
    # also filter out variables with >95% missing data as imputation will likely fail here when test sets have all np.nan
    col_list = []
    for name in list(X):
        try:
            X[name] = X[name].astype(np.float32)
            if np.sum(np.isnan(X[name].values)) / np.size(X[name].values) < 0.95:
                col_list.append(name)
        except:
            pass

    # X = X.select_dtypes(exclude=['object'])
    X = X[col_list]
    cat_bin = [1 if cname in cat_list else 0 for cname in col_list]
    # ensure integer encoding of categorical outcome
    if task == "Supervised Classification":
        d = y.values.astype(str)
        vals = np.unique(d[d != 'nan'])
        vals = list(vals)
        int_enc = [np.nan if j == 'nan' else vals.index(j) for j in d]
        y = np.array(int_enc)
        classes = len(vals)
    elif task == "Supervised Regression":
        y = y.values
        classes = 1
    
    # ensure no missing outcomes in y included in analysis
    # coerce to 
    nan_mask = np.isnan(y)
    X_ = X.values[~nan_mask, :].astype(np.float32)
    y = y[~nan_mask].astype(np.float32)
    return X_, y, classes, cat_bin

# convenience function for stratification
def stratify(classes, y):
    if classes == 1:
        bins = np.linspace(0, y.max(), classes)
        y_binned = np.digitize(y, bins)
    else:
        bins = np.linspace(0, y.max(), 50)
        y_binned = np.digitize(y, bins)
    return y_binned

def openml_ds(
        X_train,
        y_train,
        X_test,
        y_test,
        task,
        cat_bin,
        missing=None,
        imputation=None,  # one of none, simple, iterative, miceforest
        train_complete=False,
        test_complete=True,
        split=0.33,
        rng_key=0,
        prop=0.5, # proportion of rows missing when corrupting
        cols_miss=100,
        corrupt=False
    ):
    rng = np.random.default_rng(rng_key)

    key = rng.integers(9999)
    if missing is None:
        train_complete = True
        test_complete = True

    if train_complete and test_complete:
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=split, random_state=key)
        
    elif train_complete and not test_complete: # TRAIN COMPLETE IS TRUE AND TEST COMPLETE IS FALSE
        X_train, X, y_train, y_valid = train_test_split(X_train, y_train, test_size=split, random_state=key)
    
    elif not train_complete and test_complete:        
        X, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=split, random_state=key)
    
    elif not train_complete and not test_complete:
        X = X_train

    # create missingness mask
    cols = X_train.shape[1]
    if missing == "MCAR":
        p = 1 - (prop)**(1/cols)
        cols_miss = np.minimum(cols, cols_miss) # clip cols missing 
        rand_arr = rng.uniform(0, 1, (X.shape[0], cols_miss))
        nan_arr = np.where(rand_arr < p, np.nan, 1.0)
        X[:, -cols_miss:] *= nan_arr

    if missing == "MAR":
        p = 1 - (prop/2)**(1/cols)
        cols_miss = np.minimum(cols - 1, cols_miss) # clip cols missing 
        q = rng.uniform(0.3,0.7,(cols-1,))
        corrections = []
        for col in range(cols-1):
            correction = X[:,col] > np.quantile(X[:,col], q[col], keepdims=True) # dependency on each x
            corrections.append(correction)
        corrections = np.concatenate(corrections)
        corrections = np.where(corrections, 0.0, 1.0).reshape((-1,cols - 1))
        print(corrections.shape, X.shape)
        rand_arr = rng.uniform(0,1,(X.shape[0], cols - 1)) * corrections
        nan_arr = np.where(rand_arr > (1-p), np.nan, 1.0)
        X[:, -cols_miss:] *= nan_arr[:, -cols_miss:]  # dependency is shifted to the left, therefore MAR

    if missing == "MNAR":
        p = 1 - (prop/2)**(1/cols)
        cols_miss = np.minimum(cols, cols_miss) # clip cols missing 
        q = rng.uniform(0.3,0.7,(cols,))
        corrections = []
        for col in range(cols):
            correction = X[:,col] > np.quantile(X[:,col], q[col], keepdims=True) # dependency on each x
            corrections.append(correction)
        corrections = np.concatenate(corrections)
        corrections = np.where(corrections, 0.0, 1.0).reshape((-1,cols))
        rand_arr = rng.uniform(0,1,(X.shape[0], cols)) * corrections
        nan_arr = np.where(rand_arr > (1-p), np.nan, 1.0)
        X[:, -cols_miss:] *= nan_arr[:, -cols_miss:]  # dependency is not shifted to the left, therefore MNAR
    
    # generate train, validate, test datasets and impute training 
    key = rng.integers(9999)
    if train_complete and test_complete:
        pass

    elif train_complete and not test_complete:
        X_test = X
    
    elif not train_complete and test_complete:
        X_train = X

    elif not train_complete and not test_complete:
        y_binned = stratify(classes, y)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, stratify=y_binned, random_state=key)
        key = rng.integers(9999)
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=split, random_state=key)
        # y_binned = stratify(classes, y_train)
        # X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=split, stratify=y_binned, random_state=key)
        # X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=split, random_state=key)

    # missingness diagnostics
    diagnostics = {"X_train":{}, "X_valid":{}, "X_test":{}}
    diagnostics["X_train"]["cols"] = np.mean(np.isnan(X_train).sum(0) / X_train.shape[0])
    diagnostics["X_train"]["rows"] = np.any(np.isnan(X_train), axis=1).sum() / X_train.shape[0]
    diagnostics["X_valid"]["cols"] = np.mean(np.isnan(X_valid).sum(0) / X_valid.shape[0])
    diagnostics["X_valid"]["rows"] = np.any(np.isnan(X_valid), axis=1).sum() / X_valid.shape[0]
    diagnostics["X_test"]["cols"] = np.mean(np.isnan(X_test).sum(0) / X_test.shape[0])
    diagnostics["X_test"]["rows"] = np.any(np.isnan(X_test), axis=1).sum() / X_test.shape[0]
    # print(diagnostics)
    
    # final check on cat_bin

    all_cat = False    
    if np.sum(cat_bin) == 0:
        cat_bin = None
        all_cat = False
    if np.sum(cat_bin) == np.size(cat_bin):
        cat_bin = None
        all_cat = True

    # perform desired imputation strategy
    if imputation == "simple" and ((missing is not None and corrupt) or (missing is None and not corrupt)):
        X_train, X_valid, X_test = simple(
            X_train,
            dtypes=cat_bin,
            valid=X_valid,
            test=X_test,
            all_cat=all_cat
            )
    
    key = rng.integers(9999)
    if imputation == "iterative" and ((missing is not None and corrupt) or (missing is None and not corrupt)):
        X_train, X_valid, X_test = iterative(
            X_train,
            key,
            dtypes=cat_bin,
            valid=X_valid,
            test=X_test)
    
    key = rng.integers(9999)
    if imputation == "miceforest" and ((missing is not None and corrupt) or (missing is None and not corrupt)):
        if test_complete:
            test_input = None
        else:
            test_input = X_test
        X_train, X_valid, test_input = miceforest(
            X_train,
            int(key),
            dtypes=cat_bin,
            valid=X_valid,
            test=test_input)
        if test_complete:
            X_test = X_test
        else:
            X_test = test_input

    return X_train, X_test, X_valid, y_train, y_test, y_valid, diagnostics



def thoracic(
        missing="MAR", 
        imputation=None,  # one of none, simple, iterative, miceforest
        train_complete=False,
        test_complete=True,
        split=0.33,
        rng_key=0,
        p=0.5,
        cols_miss=1
    ):
    # import data
    rng = np.random.default_rng(rng_key)
    data, meta = loadarff(thoracic_path)
    d = pd.DataFrame(data)
    # convert categorical variables to integer encoding
    cols = []
    for name in meta.names():
        m = meta[name]
        if m[0] == 'nominal':
            cols.append(1)
            l = list(m[1])
            d[name] = [l.index(x.decode('UTF-8')) for x in d[name].values]
        else:
            cols.append(0)
    cols = cols[:-1]
    X_ = d.values[:, :-1]
    y = d.values[:, -1]
    
    if missing is None:
        train_complete = True
        test_complete = True

    if train_complete and test_complete:
        X = X_
        key = rng.integers(9999)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=key)
        key = rng.integers(9999)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=split, random_state=key)

    elif train_complete and not test_complete: # TRAIN COMPLETE IS TRUE AND TEST COMPLETE IS FALSE
        key = rng.integers(9999)
        X_train, X, y_train, y_test = train_test_split(X_, y, test_size=split, random_state=key)
        key = rng.integers(9999)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=split, random_state=key)
    
    elif not train_complete and test_complete:
        key = rng.integers(9999)
        X, X_test, y_train, y_test = train_test_split(X_, y, test_size=split, random_state=key)
    
    elif not train_complete and not test_complete:
        X = X_

    cols = X.shape[1]
    if missing == "MCAR":
        cols_miss = np.minimum(cols, cols_miss) # clip cols missing 
        rand_arr = rng.uniform(0, 1, (X.shape[0], cols_miss))
        nan_arr = np.where(rand_arr < p, np.nan, 1.0)
        X[:, -cols_miss:] *= nan_arr

    if missing == "MAR":
        cols_miss = np.minimum(cols - 1, cols_miss) # clip cols missing 
        q = rng.uniform(0.3,0.7,(cols-1,))
        corrections = []
        for col in range(cols-1):
            correction = X[:,col] > np.quantile(X[:,col], q[col], keepdims=True) # dependency on each x
            corrections.append(correction)
        corrections = np.concatenate(corrections)
        corrections = np.where(corrections, 0.0, 1.0).reshape((-1,cols - 1))
        print(corrections.shape, X.shape)
        rand_arr = rng.uniform(0,1,(X.shape[0], cols - 1)) * corrections
        nan_arr = np.where(rand_arr > (1-p), np.nan, 1.0)
        X[:, -cols_miss:] *= nan_arr[:, -cols_miss:]  # dependency is shifted to the left, therefore MAR

    if missing == "MNAR":
        cols_miss = np.minimum(cols, cols_miss) # clip cols missing 
        q = rng.uniform(0.3,0.7,(cols,))
        corrections = []
        for col in range(cols):
            correction = X[:,col] > np.quantile(X[:,col], q[col], keepdims=True) # dependency on each x
            corrections.append(correction)
        corrections = np.concatenate(corrections)
        corrections = np.where(corrections, 0.0, 1.0).reshape((-1,cols))
        rand_arr = rng.uniform(0,1,(X.shape[0], cols)) * corrections
        nan_arr = np.where(rand_arr > (1-p), np.nan, 1.0)
        X[:, -cols_miss:] *= nan_arr[:, -cols_miss:]  # dependency is not shifted to the left, therefore MNAR

    # generate train, validate, test datasets and impute training 
    if train_complete and test_complete:
        pass

    elif train_complete and not test_complete:
        X_test = X
    
    elif not train_complete and test_complete:
        X_train = X
        key = rng.integers(9999)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=split, random_state=key)
    
    elif not train_complete and not test_complete:
        key = rng.integers(9999)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=key)
        key = rng.integers(9999)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=split, random_state=key)

    # missingness diagnostics
    # diagnostics = {"X_train":{}, "X_valid":{}, "X_test":{}}
    # diagnostics["X_train"]["cols"] = np.isnan(X_train).sum(0) / X_train.shape[0]
    # diagnostics["X_train"]["rows"] = np.any(np.isnan(X_train), axis=1).sum() / X_train.shape[0]
    # diagnostics["X_valid"]["cols"] = np.isnan(X_valid).sum(0) / X_valid.shape[0]
    # diagnostics["X_valid"]["rows"] = np.any(np.isnan(X_valid), axis=1).sum() / X_valid.shape[0]
    # diagnostics["X_test"]["cols"] = np.isnan(X_test).sum(0) / X_test.shape[0]
    # diagnostics["X_test"]["rows"] = np.any(np.isnan(X_test), axis=1).sum() / X_test.shape[0]
    # print(diagnostics)

    # perform desired imputation strategy
    if imputation == "simple" and missing is not None:
        X_train, X_valid, X_test = simple(
            X_train,
            dtypes=None,
            valid=X_valid,
            test=X_test)
    
    key = rng.integers(9999)
    if imputation == "iterative" and missing is not None:
        X_train, X_valid, X_test = iterative(
            X_train,
            key,
            dtypes=None,
            valid=X_valid,
            test=X_test)
    
    key = rng.integers(9999)
    if imputation == "miceforest" and missing is not None:
        if test_complete:
            test_input = None
        else:
            test_input = X_test
        X_train, X_valid, test_input = miceforest(
            X_train,
            int(key),
            dtypes=None,
            valid=X_valid,
            test=test_input)
        if test_complete:
            X_test = X_test
        else:
            X_test = test_input

    return X_train, X_valid, X_test, y_train, y_valid, y_test, 2

def abalone(
    missing="MAR", 
    imputation=None,  # one of none, simple, iterative, miceforest
    train_complete=False,
    test_complete=True,
    split=0.33,
    rng_key=0,
    p=0.5,
    cols_miss=1
    ):
    rng = np.random.default_rng(rng_key)
    data = pd.read_csv(abalone_path, header=None)
    cat = list(data[0].unique())
    data[0] = [cat.index(i) for i in data[0].values]
    X_ = data.values[:, :-1]
    y = data.values[:, -1]
    unique = list(np.unique(y))
    y = np.array([unique.index(v) for v in y])
    coltypes = [1] + [0 for i in range(X_.shape[1] - 1)]

    if missing is None:
        train_complete = True
        test_complete = True

    if train_complete and test_complete:
        X = X_
        key = rng.integers(9999)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=key)
        key = rng.integers(9999)
        X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=split, random_state=key)

    elif train_complete and not test_complete: # TRAIN COMPLETE IS TRUE AND TEST COMPLETE IS FALSE
        key = rng.integers(9999)
        X_train, X, y_train, y_test = train_test_split(X_, y, test_size=split, random_state=key)
        key = rng.integers(9999)
        X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=split, random_state=key)
    
    elif not train_complete and test_complete:
        key = rng.integers(9999)
        X, X_test, y_train, y_test = train_test_split(X_, y, test_size=split, random_state=key)
    
    elif not train_complete and not test_complete:
        X = X_

    cols = X.shape[1]
    if missing == "MCAR":
        cols_miss = np.minimum(cols, cols_miss) # clip cols missing 
        rand_arr = rng.uniform(0, 1, (X.shape[0], cols_miss))
        nan_arr = np.where(rand_arr < p, np.nan, 1.0)
        X[:, -cols_miss:] *= nan_arr

    if missing == "MAR":
        cols_miss = np.minimum(cols - 1, cols_miss) # clip cols missing 
        q = rng.uniform(0.3,0.7,(cols-1,))
        corrections = []
        for col in range(cols-1):
            correction = X[:,col] > np.quantile(X[:,col], q[col], keepdims=True) # dependency on each x
            corrections.append(correction)
        corrections = np.concatenate(corrections)
        corrections = np.where(corrections, 0.0, 1.0).reshape((-1,cols - 1))
        print(corrections.shape, X.shape)
        rand_arr = rng.uniform(0,1,(X.shape[0], cols - 1)) * corrections
        nan_arr = np.where(rand_arr > (1-p), np.nan, 1.0)
        X[:, -cols_miss:] *= nan_arr[:, -cols_miss:]  # dependency is shifted to the left, therefore MAR

    if missing == "MNAR":
        cols_miss = np.minimum(cols, cols_miss) # clip cols missing 
        q = rng.uniform(0.3,0.7,(cols,))
        corrections = []
        for col in range(cols):
            correction = X[:,col] > np.quantile(X[:,col], q[col], keepdims=True) # dependency on each x
            corrections.append(correction)
        corrections = np.concatenate(corrections)
        corrections = np.where(corrections, 0.0, 1.0).reshape((-1,cols))
        rand_arr = rng.uniform(0,1,(X.shape[0], cols)) * corrections
        nan_arr = np.where(rand_arr > (1-p), np.nan, 1.0)
        X[:, -cols_miss:] *= nan_arr[:, -cols_miss:]  # dependency is not shifted to the left, therefore MNAR

    # generate train, validate, test datasets and impute training 
    if train_complete and test_complete:
        pass

    elif train_complete and not test_complete:
        X_test = X
    
    elif not train_complete and test_complete:
        X_train = X
        key = rng.integers(9999)
        X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=split, random_state=key)
    
    elif not train_complete and not test_complete:
        key = rng.integers(9999)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=key)
        key = rng.integers(9999)
        X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=split, random_state=key)

    # missingness diagnostics
    # diagnostics = {"X_train":{}, "X_valid":{}, "X_test":{}}
    # diagnostics["X_train"]["cols"] = np.isnan(X_train).sum(0) / X_train.shape[0]
    # diagnostics["X_train"]["rows"] = np.any(np.isnan(X_train), axis=1).sum() / X_train.shape[0]
    # diagnostics["X_valid"]["cols"] = np.isnan(X_valid).sum(0) / X_valid.shape[0]
    # diagnostics["X_valid"]["rows"] = np.any(np.isnan(X_valid), axis=1).sum() / X_valid.shape[0]
    # diagnostics["X_test"]["cols"] = np.isnan(X_test).sum(0) / X_test.shape[0]
    # diagnostics["X_test"]["rows"] = np.any(np.isnan(X_test), axis=1).sum() / X_test.shape[0]
    # print(diagnostics)

    # perform desired imputation strategy
    if imputation == "simple" and missing is not None:
        X_train, X_valid, X_test = simple(
            X_train,
            dtypes=None,
            valid=X_valid,
            test=X_test)
    
    key = rng.integers(9999)
    if imputation == "iterative" and missing is not None:
        X_train, X_valid, X_test = iterative(
            X_train,
            key,
            dtypes=None,
            valid=X_valid,
            test=X_test)
    
    key = rng.integers(9999)
    if imputation == "miceforest" and missing is not None:
        if test_complete:
            test_input = None
        else:
            test_input = X_test
        X_train, X_valid, test_input = miceforest(
            X_train,
            int(key),
            dtypes=None,
            valid=X_valid,
            test=test_input)
        if test_complete:
            X_test = X_test
        else:
            X_test = test_input

    return X_train, X_valid, X_test, y_train, y_valid, y_test, 1

def banking(imputation=None, split=0.33, rng_key=0):
    rng = np.random.default_rng(rng_key)
    data = pd.read_csv(bank_path, sep=";")
    cont = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
    cat = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome', 'y']
    def lab_2_num(array):
        unique_list = [l for l in list(np.unique(array)) if l != "unknown"]
        return np.array([unique_list.index(l) if l != "unknown"  else np.nan for l in array])
    
    for c in cat:
        data[c] = lab_2_num(data[c].values)
    
    data = data[cont + cat]
    coltype = [1 if i in cat else 0 for i in cont+cat]
    coltype = coltype[:-1]
    X = data.values[:, :-1]
    y = data.values[:, -1]
    # split data
    
    key = rng.integers(9999)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=key)
    key = rng.integers(9999)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=split, random_state=key)

    # diagnostics = {"X_train":{}, "X_valid":{}, "X_test":{}}
    # diagnostics["X_train"]["cols"] = np.isnan(X_train).sum(0) / X_train.shape[0]
    # diagnostics["X_train"]["rows"] = np.any(np.isnan(X_train), axis=1).sum() / X_train.shape[0]
    # diagnostics["X_valid"]["cols"] = np.isnan(X_valid).sum(0) / X_valid.shape[0]
    # diagnostics["X_valid"]["rows"] = np.any(np.isnan(X_valid), axis=1).sum() / X_valid.shape[0]
    # diagnostics["X_test"]["cols"] = np.isnan(X_test).sum(0) / X_test.shape[0]
    # diagnostics["X_test"]["rows"] = np.any(np.isnan(X_test), axis=1).sum() / X_test.shape[0]
    # print(diagnostics)


    # perform desired imputation strategy
    rng = np.random.default_rng(rng_key)
    if imputation == "simple":
        X_train, _, X_test = simple(
            X_train,
            dtypes=coltype,
            valid=None,
            test=X_test)
    
    key = rng.integers(9999)
    if imputation == "iterative":
        X_train, _, X_test = iterative(
            X_train,
            int(key),
            valid=None,
            test=X_test)
    
    key = rng.integers(9999)
    if imputation == "miceforest":
        X_train, _, X_test = miceforest(
            X_train,
            int(key),
            valid=None,
            test=X_test)

    return X_train, X_valid, X_test, y_train, y_valid, y_test, 2

def anneal(imputation=None, split=0.33, rng_key=0):
    cont = [3,4,8,32,33,34]
    def prep_data(train, test):
        cols = []
        for i in range(39):
            if i not in cont:
                d = train.values[:, i].astype(str)
                t = test.values[:, i].astype(str)
                vals = np.unique(np.concatenate([d[d != 'nan'], t[t != 'nan']]))
                vals = list(vals)
                dcoded = [np.nan if j == 'nan' else vals.index(j) for j in d]
                tcoded = [np.nan if j == 'nan' else vals.index(j) for j in t]
                if np.all(np.isnan(dcoded)):
                    pass
                else:
                    cols.append(i)
                    train[i] = dcoded
                    test[i] = tcoded
            else:
                d = train.values[:, i].astype(np.float64)
                t = test.values[:, i].astype(np.float64)
                train[i] = d
                test[i] = t
                if np.all(np.isnan(d)):
                    pass
                else:
                    cols.append(i)
                    train[i] = dcoded
                    test[i] = tcoded
        return train[cols].values, test[cols].values

    training = pd.read_csv(anneal_path_train, header=None, na_values=["?"])
    testing = pd.read_csv(anneal_path_test, header=None, na_values=["?"])
    training, testing = prep_data(training, testing)

    X_train, y_train = training[:,:-1], training[:,-1]
    X_test, y_test = testing[:,:-1], testing[:,-1]

    # perform desired imputation strategy
    rng = np.random.default_rng(rng_key)
    if imputation == "simple":
        X_train, _, X_test = simple(
            X_train,
            dtypes=[0 if i in cont else 1 for i in range(X_train.shape[1])],
            valid=None,
            test=X_test)
    
    key = rng.integers(9999)
    if imputation == "iterative":
        X_train, _, X_test = iterative(
            X_train,
            int(key),
            valid=None,
            test=X_test)
    
    key = rng.integers(9999)
    if imputation == "miceforest":
        X_train, _, X_test = miceforest(
            X_train,
            int(key),
            valid=None,
            test=X_test)

    # can't presplit before imputation as data is too sparse few 
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=split, random_state=rng_key+1)

    return X_train, X_valid, X_test, y_train, y_valid, y_test, 6

