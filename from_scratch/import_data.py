# Loading packages
import numpy as np
import random
import csv

# Loading and Splitting Data
def load_data(path : str) -> (np.ndarray, np.ndarray, list):
    """
    Parameters
    ----------
    path (str) - the file path of the csv to be loaded, where the csv has headers as the first row,
                 targets as the last column, and floats as the data\n

    Output
    ------
    features (np.ndarray) - array of shape (k, n) containg n samples of k features each\n
    targets (np.ndarray) - array of shape (1, n) containing the target values of n samples\n
    feature_names (list) - list of the names of the features
    """
    reader = csv.reader(open(path, 'r'))

    # Extract headers
    feature_names = list(next(reader))[:-1] # should be list of strings

    # Extract features/targets
    data = np.array(list(reader)).T # array including features and targets
    targets = data[-1,:].reshape(1, data.shape[1])
    features = np.delete(data, -1, axis = 0)

    return features.astype('float'), targets.astype('int'), feature_names # floats, ints, strings

def train_test_split(features : np.ndarray, targets : np.ndarray, fraction : float = 0.8) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Parameters
    ----------
    features (np.ndarray) - array of shape (k, n) containing n samples of k features each\n
    targets (np.ndarray) - array of shape (1, n) containing the target values of n samples\n
    fraction (float in [0, 1]) - the fraction of examples to be drawn for training

    Output
    ------
    train_features (np.ndarray) - subset of features containing n*fraction examples to be used for training\n
    train_targets (np.ndarray) - subset of targets corresponding to train_features containing targets\n
    test_features (np.ndarray) - subset of features containing n - n*fraction examples to be used for testing.
    test_targets (np.ndarray) - subset of targets corresponding to test_features containing targets
    """
    # Edge cases
    if fraction > 1.0 or fraction < 0.0:
        raise ValueError("Fraction must be in range [0, 1]")
    elif fraction == 1.0: # edge case where test_features = train_features
        train_features = features
        test_features = features
        train_targets = targets
        test_targets = targets
        return train_features, train_targets, test_features, test_targets

    # Main case
    ## Find indicies to train
    num_train = int(features.shape[1]*fraction)
    examples_to_train = np.sort(np.random.choice(a = features.shape[1], size = num_train, replace = False))

    ## Break up features and targets
    train_features = features[:,examples_to_train]
    train_targets = targets[:,examples_to_train]
    test_features = np.delete(features, examples_to_train, axis = 1)
    test_targets = np.delete(targets, examples_to_train, axis = 1)

    return train_features, train_targets, test_features, test_targets