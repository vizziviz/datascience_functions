import pandas as pd
from ds_utils.classifier import Classifier
from ds_utils import config

def get_features(featureset: str, file_extension: str = '.txt', 
                 basepath: str = '/Users/Livi/'):
    """Read text file contains list of features and return a cleaned, readable list
    
    Args:
        featureset (str): name of text file containing features, not including file extension
        file_extension (str): name of file extension, defaults to '.txt'
        basepath (str): location of feature file, defaults to 
            '/Users/Livi/'
    
     Returns:
        list: set of features where each element refers to a column in the aggregated e4 dataframe.
    """
    filename = basepath + featureset + '.txt'
    with open(filename, 'r') as f:
        features = f.read()
    features = features.replace('\n','').replace("'", "").replace(' ', '').split(",")
    return features

def equal_sample(df: pd.DataFrame, n: int,  random_state: int, col_name: str = 'target'): 
    """Equally sample n Falses and n Trues from Target column and return sampled dataframe.
    
    Args:
        df (pd.DataFrame): dataframe to sample from.
        n (int): number of samples to pull from each class.
        random_state (int): seed of random state.
        col_name (str): name of column to sample equally from each class, defaults to 'target.'
        
    Returns:
        pd.DataFrame: dataframe with n randomly sampled points from each class in the defined column.
    """
    return df.groupby(col_name).sample(n, random_state=random_state).sort_index()

def get_parts(dataset: int):
    """Return a list of parts associated with a dataset.
    
    Args:
        datset (int): dataset whose parts are to be accessed.
    
    Returns:
        list: list of all parts available for the data of the requested dataset.
    """
    all_datasets = [1, 2, 3, 4, 5, 6]
    all_parts = [[1,2,3,4], [1,2,3,4], [1], [1], [1], [1,2]]
    
    idx = all_pirs.index(dataset)
    parts = all_parts[idx]
    
    return parts

def make_holdout_set(holdout_set: int, excluding: bool, all_datasets = [1, 2, 3, 4, 5, 6],
                     unreliable_datasets = [7, 8]):
    """Create list of datasets with the test dataset withheld and a list of parts for those datasets. 
    If excluding unreliable datasets, do not include these or their parts in return lists.
    
    Args:
        holdout_set (int): dataset to withhold from returned list
        excluding (bool): whether to include unreliable datasets. If True, excludes. If False, includes.
        all_datasets (list): all datasets from which to exclude holdout dataset
        unreliable_datasets (list): datasets which have questionable data
    
    Returns:
        list, list: datasets with holdout dataset removed, parts for those datasets with holdout dataset's parts removed
    """
    
    if excluding:
        all_datasets = [elem for elem in all_datasets if elem not in unreliable_datasets]
        all_parts = [get_parts(i) for i in all_datasets]

        if holdout_dataset in all_datasets:
            idx = all_datasets.index(holdout_dataset)
            trainsets = [x for x in all_datasets if x != holdout_dataset]
            trainparts = all_parts[:idx] + all_parts[idx+1:]
    
        else:
            trainpirs = all_datasets
            trainparts = all_parts
        
    else:
        all_parts = [get_parts(i) for i in all_datasets]

        idx = all_datasets.index(holdout_dataset)
        trainsets = [x for x in all_datasets if x != holdout_dataset]
        trainparts = all_parts[:idx] + all_parts[idx+1:]
    
        
    return traindatasets, trainparts

def read_e4(dataset: int, parts: list, window_hrs: int, sample: bool, data_path: str = (config.pars('data') + 
            '/one_hz/'), random_state: int = None, n: int = None):
    """Read all e4 dataframe for all parts of dataset and return concatenated dataframe
    
    Args:
        dataset (int): dataset number.
        parts (list):  all parts to load for dataset.
        sample (bool): whether to equally sample 250 points from True and False cravings.
        window_hrs (int): length of craving for which wearable data was aggregated
        n (int, optional): number of rows to sample from data if sample=True
        random_state (int, optional): seed to use to randomly sample points if sample=True

    Returns:
        pd.DataFrame: the combined dataframe for specified parts of the dataset 
    """ 
    dfs = []

    for part in parts:
        path = f'{data_path}/full_agg_1hz_dataset{dataset}pt{part}_{window_hrs}hr.pq'
        df = pd.read_parquet(path)
        
        if sample:
            new_df = equal_sample(df=df.dropna(),n=n, random_state=random_state)
            dfs.append(new_df)    
        else:
            dfs.append(df)
            
    df = pd.concat(dfs, axis=0)
    return df

def read_all_data(datasets: list, parts: list, window_hrs: int, data_path: str, sample: bool, 
                  random_state: int = None, n: int = 250):
    """Combine aggregated wearable data for all parts, sample training data and return train and test 
    dataframes.
    
    Args:
        datasets (list): datasets to read
        parts (list): parts of dataset to read
        window_hrs (int): hours for which wearable data was aggregated
        sample (bool): if True samples equally from target classes
        n (int): number of rows to sample from each target class if sample=True, defaults to 250
        random_state (int): seed for sampling training data if sample=True, defaults to None
        data_path (str): path to aggregated data files
        
    Returns:
        pd.DataFrame, pd.DataFrame: dataframe to train classifier, dataframe to test classifier on
    """
        
    dfs = []

    for i in range (len(pirs)):

        pir = pirs[i]
        parts_sublist = parts[i]
        if sample:
            df = read_e4(pir=pir, parts=parts_sublist, sample=True, random_state=random_state, n=n,
                           window_hrs=window_hrs, data_path=data_path)
        else:
            df =read_e4(pir=pir, parts=parts_sublist, sample=False, window_hrs=window_hrs, data_path=data_path)
            
        dfs.append(df)

    df = pd.concat(dfs, axis=0)
    
    return df

def train_and_test_rf(train_df, test_df, features, display):
    """Create a Random Forest Classifier, run 100 times and return a per-point-summary for point accuracy.
    
     Args:
        train_df (pd.DataFrame): data of PIRs to be used to train model
        test_df (pd.DataFrame): data of PIR to test model on
        features (list): list of columns to filter both training and test data
        display (bool): If True, displays metrics and confusion metrics. If False, does not display.
   
    Returns:
        ScikitModel, pd.DataFrame, pd.DataFrame: trained model, dataframe tested on, dataframe of metrics
    """
    
    test_df['hour'] = test_df.index.hour    
    train_df['hour'] = train_df.index.hour
    
    features = list(set(features + ['target']))
    test_df = test_df.reindex(columns=features).dropna()
    train_df = train_df.reindex(columns=features).dropna()
    
    y_train = train_df['target'].values.astype(float)
    y_test = test_df['target'].values.astype(float)

    X_train = train_df.drop(columns='target')
    X_test = test_df.drop(columns='target')
        
    clf = Classifier()
    clf.train(X_train, y_train, iters=100, alg='rf')
    clf.test(X_test, y_test, display=display,smooth=True)
    pps= clf.per_point_summary(plot=False)
    
    return pps


def create_avg_pps(features: list, train_datasets: list, test_datasets: list,train_parts: list, 
                   test_parts: list, window_hrs: int, display: bool = False, 
                   data_path: str = config.pars('data'), subdir: str = None):
    """Train 2 classifiers for labeling accuracy for test datasets and return the average per-point-summary
    
    Args:
        features (list): list of features to use for the classifier.
        train_sets (list): list of datasets to use to train the model e.g. [1, 2, 3, 4].
        test_sets (list): list of datasets to test on and create per-point-summary of label accuracy 
            e.g. [1, 2, 3, 4].
        train_parts (list): list of lists of parts, each element aligns with each element of train_sets
            e.g. [[1,2,3,4], [1,2,3,4], [1], [1]] referring to the part of [1, 2, 3, 4]].
        test_parts (list): list of lists of parts, each element aligns with each element of test_sets
            e.g. [[1,2,3,4], [1,2,3,4], [1], [1]] referring to the part of [1, 2, 3, 4].
        window_hrs (int): length of craving to test on, points to the correct aggregated e4 file
            e.g. aggregated_dataset1_pt1_6hr.pq for the 6-hour target aggregation.
        display (bool): if True, shows metrics and confusion matrix. If False, does not. Defaults False.
        data_path (str): path to e4 data files, defaults to config.pars('data')        
    
    Returns:
        pd.DataFrame: dataframe averaging two per-point-summaries with different random seeds
    """
        
    train_df1 = read_all_data(datasets=trainsets, parts=train_parts, window_hrs=window_hrs, 
                              data_path=data_path, sample=True, random_state=1, n=250)
    train_df2 = read_all_data(datasets=trainsets, parts=train_parts, window_hrs=window_hrs, 
                              data_path=data_path, sample=True, random_state=2, n=250)
    
    test_df = read_all_data(datasets=testsets, parts=test_parts, window_hrs=window_hrs, 
                              data_path=data_path, sample=False)
    
    pps1 = train_and_test_rf(train_df1, test_df,features, display)
    pps2 = train_and_test_rf(train_df2, test_df,features, display)
    pps1and2 = pd.concat([pps1, pps2])
    pps_avg = pps1and2.groupby(pps1and2.index).mean()
            
    return pps_avg

def filter_and_save_pps(dataset:int, pps: pd.DataFrame, excluding: bool, featureset: str, holdout_set:
                        int, window_hrs: int, 
                        path = '/Users/Livi/pointlabeling/pps/'):
    """Filter the per-point-summary (pps) of all datasets and save a separate pps for each dataset
    
    Argus:
        dataset (int): the dataset for which to filter the per-point-summary
        pps (pd.DataFrame): the per-point-summary to filter
        excluding (bool): describes if unreliable datasets were used to train the label model
        features (str): the name of the featureset used (to name the saved file)
        holdout_set (int): the dataset omitted from training the label model to avoid overfitting
        window_hrs (int): the length of the craving for which wearable data was aggregated
        path (str): the subdirectory in which to store the saved file, defaults to
            '/Users/Livi/pointlabeling/pps/'
    
    Returns:
        None
    """
    print('filtering and saving ppses')
    parts = get_parts(dataset)
    dataset_data = read_e4(dataset=dataset, parts=parts, window_hrs=window_hrs, sample=False)
    
    test_df_pps = pps.reindex(dataset_data.index)
    
    if excluding == True:
        filename = (
            f'{path}{window_hrs}hr_target/pps{dataset}_{featureset}_excluding_hold{holdout_set}.pq')
    else:
        filename = (
            f'{path}{window_hrs}hr_craving/pps{dataset}_{featureset}_including_hold{holdout_set}.pq')
        
    test_df_pps.to_parquet(filename)
    
    
                 