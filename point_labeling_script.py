import point_labeling_functions as bootstrap
import pandas as pd
import itertools.product

def make_all_per_point_summaries(holdout_sets: list, windows: list, featuresets: list, is_excluding: list):
    """For each combination of target window, features, and training set (whether training sets
    includes or excludes unreliable datasets), create a per-point-summary and save it to disk.
    
    Args:
        holdout_sets (list): datasets to holdout from labeling so not to overfit models when testing later on that same dataset
        windows (list): length of target state to test on
        featuresets (list): names of featuresets to load
        is_excluding (list): list of booleans. True includes the unreliable datasets, False excludes them.
    """
    combos = itertools.product(windows, featuresets, is_excluding)

    for window, featureset, excluding in combos:
        print('Target window:', window, ', features:', featureset, ', excluding:', excluding)
            
        features = bootstrap.get_features(featureset)
            
        for holdout_set in holdout_sets:
            trainsets, trainparts = bootstrap.make_holdout_set(holdout_set=holdout_set, 
                                                               excluding=excluding)
            testsets, testparts = trainsets.copy(), trainparts.copy()
                
            pps = bootstrap.create_avg_pps(features=features, train_sets=trainsets, test_sets=testsets,
                                           train_parts=trainparts, test_parts=testparts, 
                                           window_hrs=window, subdir = 'one_hz/')
            for testset in testsets:
                bootstrap.filter_and_save_pps(testset=testset, pps=pps, excluding=excluding, 
                                              featureset=featureset, holdout_set=holdout_set,
                                              window_hrs=window)

if __name__ == '__main__':
    make_all_pps(holdout_sets=[1, 2, 3, 4], windows=[1, 2, 3],
                 featuresets=['feats1', 'feats2', 'feats3', 'feats4'], 
                 is_excluding=[True, False])
    
