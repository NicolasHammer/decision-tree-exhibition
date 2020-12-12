import numpy as np

def confusion_matrix(actual : np.ndarray, predictions : np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    actual (np.ndarray) - actual labels\n
    predictions (np.ndarray) - predicted labels

    Output
    ------
    confusion_matrix (np.ndarray) - 2x2 confusion matric between actual and predictions
    [
        [true_negatives, false_positives],
        [false_negatives, true_positives]
    ]
    """
    if predictions.shape != actual.shape:
        raise ValueError("predictions and actual must be the same shape!")

    return np.array([
        [np.sum(np.logical_and(actual == 0, predictions == 0)), np.sum(np.logical_and(actual == 0, predictions == 1))],
        [np.sum(np.logical_and(actual == 1, predictions == 0)), np.sum(np.logical_and(actual == 1, predictions == 1))]
    ])

def accuracy(actual : np.ndarray, predictions : np.ndarray) -> float:
    """
    Parameters
    ----------
    actual (np.ndarray) - actual labels\n
    predictions (np.ndarray) - predicted labels

    Output
    ------
    accuracy (float) - accuracy score
    """
    if predictions.shape != actual.shape:
        raise ValueError("predictions and actual must be the same length!")
    
    conf_mat = confusion_matrix(actual, predictions)

    return float((conf_mat[0, 0] + conf_mat[1, 1])/np.sum(conf_mat))

def precision_and_recall(actual : np.ndarray, predictions : np.ndarray) -> (float, float):
    """
    Parameters
    ----------
    actual (np.ndarray) - actual labels\n
    predictions (np.ndarray) - predicted labels

    Output
    ------
    precision (float) - precision of the model
    recall (float) - recall of the model
    """
    if predictions.shape != actual.shape:
        raise ValueError("predictions and actual must be the same length!")

    conf_mat = confusion_matrix(actual, predictions)

    precision = float(conf_mat[1, 1]/np.sum(conf_mat[:, 1]))
    recall = float(conf_mat[1, 1]/np.sum(conf_mat[1, :]))

    return precision, recall

def f1_measure(actual : np.ndarray, predictions : np.ndarray) -> float:
    """
    Parameters
    ----------
    actual (np.ndarray) - actual labels\n
    predictions (np.ndarray) - predicted labels

    Output
    ------
    f1_measure (float) - harmonic mean of precision and recall
    """
    if predictions.shape != actual.shape:
        raise ValueError("predictions and actual must be the same length!")

    precision, recall = precision_and_recall(actual, predictions)

    return float(2 * (precision * recall)/(precision + recall))