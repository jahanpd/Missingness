import numpy as np
import pandas as pd
from scipy.io.arff import loadarff
# functions to prep dataset
import sklearn.datasets as skdata
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import miceforest as mf
# import tensorflow_datasets
import openml
import sys
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
  from tqdm.notebook import tqdm
else:
  from tqdm import tqdm

def get_list(
        missing,
        min_features=20,
        max_features=250,
        min_instances=1000,
        max_instances=100000,
        test=lambda x, m: x > m,
        include_regression = False,
        from_suite=True,
        min_classes=2,
        minority_class_frac=0.1
        ):
    """ 
        This function returns a complete pandas dataframe of datasets available on OpenML 
        that fulfil the necessary prerequisites.
        Args:
            missing: float [0,1], the fraction of missingness in the data
            key: int, random key
            test: callable, needs to return bool. Used for subsetting based on missing arg
                eg lambda x, m: x > m will select datasets with missingnes greater than the above specified missing
                and lambda x, m: x == 0 will select dataset with no missing data_a
        Returns:
            Pandas Dataframe

    """
    
    if from_suite:
        benchmark_suite_reg = openml.study.get_suite('amlb-regression')
        tasklist = openml.tasks.list_tasks(output_format="dataframe")
        tasklist.to_csv("results/openml/tasklist.csv")
        reglist = tasklist[tasklist.tid.isin(benchmark_suite_reg.tasks)]
        benchmark_suite_cla = openml.study.get_suite('OpenML-CC18')
        clalist = tasklist[tasklist.tid.isin(benchmark_suite_cla.tasks)]
    else:
        tasklist = openml.datasets.list_datasets(output_format="dataframe")
        tasklist.to_csv("results/openml/tasklist.csv")
        reglist = tasklist[tasklist.NumberOfClasses == 0]
        clalist = tasklist[tasklist.NumberOfClasses > 1]

    # to generate list of datasets with high proportion of missingness
    # datasets = openml.datasets.list_datasets(output_format='dataframe')

    # class_tasks = datasets[datasets.NumberOfClasses > 0.0]
    # reg_tasks = datasets[datasets.NumberOfClasses == 0.0]
    # subsets based on missingness
    class_subset = clalist[test(clalist.NumberOfInstancesWithMissingValues / clalist.NumberOfInstances, missing) & (clalist.NumberOfClasses >= min_classes)]
    reg_subset = reglist[test(reglist.NumberOfInstancesWithMissingValues / reglist.NumberOfInstances, missing)]
    # most features need to be mainly numerical or symbolic in order to avoid NLP tasks
    # perc = lambda a, b, c: (a+b)/c > 0.8
    # class_subset = class_subset[perc(class_subset.NumberOfNumericFeatures, class_subset.NumberOfSymbolicFeatures, class_subset.NumberOfFeatures)]
    # reg_subset = reg_subset[perc(reg_subset.NumberOfNumericFeatures, reg_subset.NumberOfSymbolicFeatures, reg_subset.NumberOfFeatures)]
    # limit number of features to less than 500 for tractability
    class_subset = class_subset[
        (class_subset.NumberOfFeatures < max_features) & 
        (class_subset.NumberOfFeatures > min_features) & 
        (class_subset.NumberOfInstances < max_instances) & 
        (class_subset.NumberOfInstances > min_instances) &
        ((class_subset.MinorityClassSize / class_subset.NumberOfInstances) > minority_class_frac) &
        test(class_subset.NumberOfMissingValues, class_subset.NumberOfInstancesWithMissingValues)
        ].drop_duplicates(subset=["name"])
    reg_subset = reg_subset[
        (reg_subset.NumberOfFeatures < max_features) & 
        (reg_subset.NumberOfFeatures > min_features) & 
        (reg_subset.NumberOfInstances < max_instances) & 
        (reg_subset.NumberOfInstances > min_instances) &
        ((class_subset.MinorityClassSize / class_subset.NumberOfInstances) > minority_class_frac) &
        test(reg_subset.NumberOfMissingValues, reg_subset.NumberOfInstancesWithMissingValues)
        ].drop_duplicates(subset=["name"])
    
    # test if datasets are gettable and add to list
    did_list_class = []
    pbar = tqdm()
    for did in tqdm(list(class_subset.did.values)):
        try:
            # print(did_list_class, len(class_subset))
            # did = rng.choice(class_subset.did.values, 1)
            ds = openml.datasets.get_dataset(dataset_id=int(did))
            X, y, categorical_indicator, attribute_names = ds.get_data(target = ds.default_target_attribute)
            did_list_class.append(int(did))
            did_list_class = list(set(did_list_class))
            pbar.update()
        except Exception as e:
            print(e)
            pass
    did_list_reg = []
    for did in tqdm(list(reg_subset.did.values)):
        try:
            # print(did_list_reg, len(reg_subset))
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
    if include_regression:
        return pd.concat([class_subset, reg_subset])
    else:
        return class_subset


# convenience imputation functions
def simple(train, valid, test, dtypes=None, all_cat=False):
    """
    function to impute datasets using simple imputation.
    args:
        train: ndarray, training dataset
        valid: ndarray, validation dataset
        test: ndarray, test dataset
        dtypes: list, column label where 1 is categorical and 0 is continuous
        all_cat: bool, true if all columns are categorical
    returns:
        train, valid, test: ndarray, datasets with impuated values
    """
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
        if np.sum(cont) > 0: 
            imp1.fit(train[:, cont])
            train[:, cont] = imp1.transform(train[:, cont])
        if np.sum(cat) > 0:
            imp2.fit(train[:, cat])
            train[:, cat] = imp2.transform(train[:, cat])
        if valid is not None:
            if np.sum(cont) > 0: 
                valid[:, cont] = imp1.transform(valid[:, cont])
            if np.sum(cat) > 0:
                valid[:, cat] = imp2.transform(valid[:, cat])
        if test is not None:
            if np.sum(cont) > 0: 
                test[:, cont] = imp1.transform(test[:, cont])
            if np.sum(cat) > 0: 
                test[:, cat] = imp2.transform(test[:, cat])
    return train, valid, test

def iterative(train, rng_key, dtypes=None, valid=None, test=None):
    """
        function to impute datasets using multivariate imputation.
        args:
            train: ndarray, training dataset
            rng_key: int, set for rng generator
            valid: ndarray, validation dataset
            test: ndarray, test dataset
            dtypes: list, column label where 1 is categorical and 0 is continuous
        returns:
            train, valid, test: ndarray, datasets with impuated values
    """
    cont = np.array(dtypes) == 0
    cat = np.array(dtypes) == 1
    imp1 = IterativeImputer(max_iter=10, random_state=rng_key, n_nearest_features=np.minimum(10, len(cont)))
    imp2 = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    if np.sum(cont) > 0: 
        imp1.fit(train[:, cont])
        train[:, cont] = imp1.transform(train[:, cont])
    if np.sum(cat) > 0:
        imp2.fit(train[:, cat])
        train[:, cat] = imp2.transform(train[:, cat])
    if valid is not None:
        if np.sum(cont) > 0: 
            valid[:, cont] = imp1.transform(valid[:, cont])
        if np.sum(cat) > 0:
            valid[:, cat] = imp2.transform(valid[:, cat])
    if test is not None:
        if np.sum(cont) > 0: 
            test[:, cont] = imp1.transform(test[:, cont])
        if np.sum(cat) > 0: 
            test[:, cat] = imp2.transform(test[:, cat])
    return train, valid, test

def miceforest(train, rng_key, dtypes=None, valid=None, test=None):
    """
        function to impute datasets using multivariate imputation.
        args:
            train: ndarray, training dataset
            rng_key: int, set for rng generator
            valid: ndarray, validation dataset
            test: ndarray, test dataset
            dtypes: list, column label where 1 is categorical and 0 is continuous
        returns:
            train, valid, test: ndarray, datasets with impuated values
    """
    colnames = [str(i) for i in range(train.shape[1])]
    df = pd.DataFrame(train, columns=colnames)
    # set dtypes of each column
    dtype_dict = {}
    for colname, dt in zip(colnames, dtypes):
        if dt == 0:
            dtype_dict[colname] = 'float'
        else:
            dtype_dict[colname] = 'category'
    df.astype(dtype_dict)
    # set mean match candidates to max of 10
    mms = int(np.minimum(10, train.shape[1]))
    kernel = mf.ImputationKernel(
                df,
                datasets=4,
                save_all_iterations=True,
                random_state=rng_key,
                mean_match_candidates=0,
                data_subset=mms
                )
    # define tree parameters
    print("fitting mice model")
    kernel.mice(
            3,
            n_estimators=5,
            num_leaves=30,
            max_depth=10,
            )
    print("mice model fitted")
    train = kernel.complete_data(0).values
    if valid is not None:
        row_mask = np.any(np.isnan(valid), axis=1)
        if np.sum(row_mask) > 0:
            valid_imp = kernel.impute_new_data(
                new_data=pd.DataFrame(valid, columns=colnames))
            valid = valid_imp.complete_data(0).values
    if test is not None:
        row_mask = np.any(np.isnan(test), axis=1)
        if np.sum(row_mask) > 0:
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
    """
        Function to generate a spiral dataset.
        Args:
            N: int, total number of samples per class (2*N is total datasets size)
            missing: str, one of None, MCAR, MAR, MNAR
            imputation: str, one of None, simple, iterative, miceforest
            train_complete: bool, whether training set is complete or has missingness
            test_complete: bool, whether test set is complete or has missingness
            rng_key: int, rng key generator
            p: float, interval (0,1) prob of missingness
            cols_miss: int, number of cols with missing data
        Returns:
            X_train: ndarray
            X_valid: ndarray
            X_test: ndarray 
            y_train: ndarray
            y_valid: ndarray
            y_test: ndarray 
            (x_a, x_b): tuple of ndarray for plotting
            classes: int, 2 for spiral dataset
    """
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
    """
        Takes did, and task string and returns datasets
        Args:
            did: int, did of openml dataset
            task: str, one of 'Supervised Classification' or 'Supervised Regression'
        Returns:
            X_: ndarray
            y: ndarray
            classes: int, number of classes in y (1 if regression)
            cat_bin: list, label for each cols where 1 is categorical and 0 is continuous
    """
    # takes did, target colname and task type string
    # returns prepared X, y numpy arrays
    ds = openml.datasets.get_dataset(dataset_id=int(did))
    X, y, categorical_indicator, attribute_names = ds.get_data(target = ds.default_target_attribute)

    cont_list = []
    cat_list = []
    dummify = []
    for cat, name in zip(categorical_indicator, list(X)):
        if cat:
            cat_list.append(name)
            d = X[name].values.astype(str)
            vals = np.unique(d[d != 'nan'])
            if len(vals) > 2:
                dummify.append(name)
            vals = list(vals)
            int_enc = [np.nan if j == 'nan' else vals.index(j) for j in d]
            X[name] = int_enc
        else:
            cont_list.append(name)
    # X = pd.get_dummies(X, columns=dummify) 
    # get rid of features that are objects but not in categorical indicator - these are usually unhelpful columns like names etc
    # also filter out variables with >95% missing data as imputation will likely fail here when test sets have all np.nan
    col_list = []
    for name in list(X):
        try:
            X[name] = X[name].astype(np.float32)
            if np.sum(np.isnan(X[name].values)) / np.size(X[name].values) < 0.99:
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
        classes,
        missing=None,
        imputation=None,  # one of none, simple, iterative, miceforest
        train_complete=False,
        test_complete=True,
        split=0.33,
        rng_key=0,
        prop=0.5, # proportion of rows missing when corrupting
        cols_miss=100,
        corrupt=False,
        noncorrupt=False
    ):
    """
        Prepare an OpenML dataset with desired missingness and imputation.
        Args:
            X_train: ndarray
            y_train: ndarray
            X_test: ndarray
            y_test: ndarray
            task: str, one of 'Supervised Classication' or 'Supervised Regression'
            cat_bin: list, labelling of cols 1 if categorical and 0 if continuous
            classes: int, number of classes in y
            missing: str, desired missingness (None, MCAR, MAR, MNAR),
            imputation: str,  desired imputation (none, simple, iterative, miceforest)
            train_complete: bool, if train has missingness
            test_complete: bool, if test set has missingness
            split: float, interval (0,1) for validation split
            rng_key: int, rng key
            prop: float, interval (0,1) proportion of rows missing when corrupting
            cols_miss: int, number of cols with missing vals,
            corrupt: bool, whether corrupting data or not
        Return:
            X_train: ndarray
            X_test: ndarray
            X_valid: ndarray
            y_train: ndarray
            y_test: ndarray
            y_valid: ndarray
            diagnostics: dict, containing values related to levels of missingness
    """
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

    if corrupt:
        # drop data proportional to mutual information
        if task == 'Supervised Regression':
            mi = mutual_info_regression(
                X = X_train,
                y = y_train,
                discrete_features=np.array(cat_bin)==1
            )
        if task == 'Supervised Classification':
            mi = mutual_info_classif(
                X = X_train,
                y = y_train,
                discrete_features=np.array(cat_bin)==1
            )
        sort = np.argsort(mi)
        unsort = np.argsort(sort)
        # create missingness mask
        cols = X_train.shape[1]
        if missing == "MCAR":
            cols_miss = np.minimum(cols, cols_miss) # clip cols missing 
            rand_arr = rng.uniform(0, 1, X.shape)
            nan_arr = np.where(rand_arr < prop, np.nan, 1.0)
            X[:, sort[-cols_miss:]] *= nan_arr[:, sort[-cols_miss:]]

        if missing == "MAR":
            p_master = prop
            cols_miss = np.minimum(cols, cols_miss) # clip cols missing
            nan_list = []
            for i,b in enumerate(cat_bin):
                if b == 1: # categorical
                    unique, counts = np.unique(X[:, i], return_counts=True)
                    if len(unique) == 2:
                        max_val = unique[np.argmax(counts)]
                        prop = counts.max() / counts.sum()
                        p = 0.99 if prop < p_master else p_master / prop
                        rand_arr = rng.uniform(0,1,X[:,i].shape)
                        rand_arr[X[:, i] != max_val] = 1
                        set_nan = rand_arr < p
                        nan_arr = np.where(set_nan, np.nan, 1.0)
                        nan_list.append(nan_arr)
                    elif len(unique) > 2 :
                        argsort = np.argsort(counts)
                        cumsum = np.cumsum(counts[argsort]/counts.sum())
                        cs_bool = cumsum < 0.8
                        cs_max = np.max(cumsum[cs_bool])
                        max_vals = unique[argsort[cs_bool]]
                        p = 0.99 if cs_max < p_master else p_master / prop
                        rand_arr = rng.uniform(0,1,X[:,i].shape)
                        rand_arr[~np.isin(X[:, i], max_vals)] = 1
                        set_nan = rand_arr < p
                        nan_arr = np.where(set_nan, np.nan, 1.0)
                        nan_list.append(nan_arr)
                    else:
                        rand_arr = rng.uniform(0,1,X[:,i].shape)
                        set_nan = rand_arr < p_master
                        nan_arr = np.where(set_nan, np.nan, 1.0)
                        nan_list.append(nan_arr)
                else: 
                    q = np.quantile(X[:, i], p_master)
                    ratio = np.sum(X[:,i] <= q) / X.shape[0]
                    if ratio > p_master:
                        set_nan1 = X[:, i] <= q
                        rand_arr = rng.uniform(0,1,X[:,i].shape)
                        set_nan2 = rand_arr < p_master / ratio
                        set_nan = set_nan1 & set_nan2
                        nan_arr = np.where(set_nan, np.nan, 1.0)
                    else:
                        set_nan = X[:, i] <= q
                        nan_arr = np.where(set_nan, np.nan, 1.0)
                    nan_list.append(nan_arr)

            nan_arr = np.concatenate(nan_list).reshape(len(nan_list), -1).T
            print("total missing: ", np.sum(np.isnan(nan_arr))/np.size(X))
            X[:, sort[0:cols_miss-1]] *= nan_arr[:, sort[1:cols_miss]]  # dependency is shifted to the left, therefore MAR
            print("complete cols: ", np.sum(np.all(~np.isnan(X), axis=0)))

        if missing == "MNAR":
            p_master = prop
            cols_miss = np.minimum(cols, cols_miss) # clip cols missing
            nan_list = []
            for i,b in enumerate(cat_bin):
                # for each column get ~20% that you are keeping for sure
                # based completely on the value, hence MNAR
                # aim to delete 60% per column
                if b == 1: # categorical
                    unique, counts = np.unique(X[:, i], return_counts=True)
                    if len(unique) == 2:
                        max_val = unique[np.argmax(counts)]
                        prop = counts.max() / counts.sum()
                        p = 0.99 if prop < p_master else p_master / prop
                        rand_arr = rng.uniform(0,1,X[:,i].shape)
                        rand_arr[X[:, i] != max_val] = 1
                        set_nan = rand_arr < p
                        nan_arr = np.where(set_nan, np.nan, 1.0)
                        nan_list.append(nan_arr)
                    elif len(unique) > 2 :
                        argsort = np.argsort(counts)
                        cumsum = np.cumsum(counts[argsort]/counts.sum())
                        cs_bool = cumsum < 0.8
                        cs_max = np.max(cumsum[cs_bool])
                        max_vals = unique[argsort[cs_bool]]
                        p = 0.99 if cs_max < p_master else p_master / prop
                        rand_arr = rng.uniform(0,1,X[:,i].shape)
                        rand_arr[~np.isin(X[:, i], max_vals)] = 1
                        set_nan = rand_arr < p
                        nan_arr = np.where(set_nan, np.nan, 1.0)
                        nan_list.append(nan_arr)
                    else:
                        rand_arr = rng.uniform(0,1,X[:,i].shape)
                        set_nan = rand_arr < p_master
                        nan_arr = np.where(set_nan, np.nan, 1.0)
                        nan_list.append(nan_arr)
                else: 
                    q = np.quantile(X[:, i], p_master)
                    ratio = np.sum(X[:,i] <= q) / X.shape[0]
                    if ratio > p_master:
                        set_nan1 = X[:, i] <= q
                        rand_arr = rng.uniform(0,1,X[:,i].shape)
                        set_nan2 = rand_arr < p_master / ratio
                        set_nan = set_nan1 & set_nan2
                        nan_arr = np.where(set_nan, np.nan, 1.0)
                    else:
                        set_nan = X[:, i] <= q
                        nan_arr = np.where(set_nan, np.nan, 1.0)
                    nan_list.append(nan_arr)

            nan_arr = np.concatenate(nan_list).reshape(len(nan_list), -1).T
            print("total missing: ", np.sum(np.isnan(nan_arr))/np.size(X))
            X[:, sort[:cols_miss]] *= nan_arr[:, sort[:cols_miss]]  # dependency is not shifted to the left, therefore MNAR
            print("complete cols: ", np.sum(np.all(~np.isnan(X), axis=0)))
        
    # generate train, validate, test datasets and impute training 
    key = rng.integers(9999)
    if train_complete and test_complete:
        pass

    elif train_complete and not test_complete:
        X_test = X
    
    elif not train_complete and test_complete:
        X_train = X

    elif not train_complete and not test_complete:
        # y_binned = stratify(classes, y)
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
        all_cat = False
    if np.sum(cat_bin) == np.size(cat_bin):
        all_cat = True

    if not all_cat:
        # normalize categorical data
        cont = np.array(cat_bin) == 0
        scale_train = StandardScaler()
        X_train[:, cont] = scale_train.fit_transform(X_train[:, cont])
        X_valid[:, cont] = scale_train.transform(X_valid[:, cont])
        X_test[:, cont] = scale_train.transform(X_test[:, cont])
        
    # perform desired imputation strategy
    if imputation == "simple" and ((missing is not None and corrupt) or (missing is None and not corrupt) or noncorrupt):
        X_train, X_valid, X_test = simple(
            X_train,
            dtypes=cat_bin,
            valid=X_valid,
            test=X_test,
            all_cat=all_cat
            )
    
    key = rng.integers(9999)
    if imputation == "iterative" and ((missing is not None and corrupt) or (missing is None and not corrupt) or noncorrupt):
        X_train, X_valid, X_test = iterative(
            X_train,
            key,
            dtypes=cat_bin,
            valid=X_valid,
            test=X_test)
    
    key = rng.integers(9999)
    if imputation == "miceforest" and ((missing is not None and corrupt) or (missing is None and not corrupt) or noncorrupt):
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

    if task == "Supervised Regression":
        scaler = StandardScaler()
        y_train = scaler.fit_transform(y_train.reshape((-1,1))).flatten()
        y_test = scaler.transform(y_test.reshape((-1,1)))
        y_valid = scaler.transform(y_valid.reshape((-1,1)))

    return X_train, X_test, X_valid, y_train, y_test, y_valid, diagnostics
