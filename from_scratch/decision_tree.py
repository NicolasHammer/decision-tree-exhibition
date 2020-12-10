import numpy as np

class Tree():
    def __init__(self, value : (float or int) = None, attribute_name : str = "root",
     attribute_index : int = None, branches : list = None):
     """
     A tree structure with multiple branches at eah node.  The default values correspond
     to the root node. 

     Parameters / Member Variables
     -----------------------------
     value (float or int) - the data value\n
     attribute_name (str) - name of the attribute that the tree splits the data on\n
     attribute_index (float) - index of the attribute in the feature vector\n
     branches (list) - list of Tree objects.  Since this is being used for a binary decision tree,
                       this should either be 2 or 0.
     """
     self.value = value
     self.attribute_name = attribute_name
     self.attribute_index = attribute_index
     self.branches = [] if branches is None else branches

class DecisionTree():
    """
    A binary decision tree learner using the ID3 algorithm (https://en.wikipedia.org/wiki/ID3_algorithm).

    Member Variables
    ----------------
    attribute_names (list) - the name of attributes to split by\n
    self.tree (Tree) - the tree structure used to store the information
    """

def entropy(targets : np.ndarray) -> float:
    """
    Returns entropy (https://en.wikipedia.org/wiki/ID3_algorithm#The_ID3_metrics) given features
    for each example.  The entropy funcion is called H(S).
    """
    _, counts = np.unique(targets, return_counts=True)
    proportion = counts/np.sum(counts)
    return -np.sum(proportion*np.log2(proportion))

def information_gain(features : np.ndarray, attribute_index : int, targets : np.ndarray) -> float:
    """
    Caculates information gain (https://en.wikipedia.org/wiki/ID3_algorithm#The_ID3_metrics).

    Parameters
    ----------
    features (np.ndarray) - features for each example\n
    attribute_index (int) - index corresponding to which colun of features to take when computing
                            information gain\n
    targets (np.ndarray) - labels corresponding to each example

    Output
    ------
    information_gain (float) - information gain when the features are split on attribute_index
    """
    positive_targets = targets[features[attribute_index, :] == 1] 
    negative_targets = targets[features[attribute_index, :] == 0]
    positive_entropy = entropy(positive_targets) # H(targets | feature == 1)
    negative_entropy = entropy(negative_targets) # H(targets | feature == 0)

    prior_entropy = entropy(targets) # H(targets)

    # H(targets) - (Pr(feature == 1)*H(targets | feature == 1) + Pr(feature == 0)*H(targets | feature == 0))
    return prior_entropy - (
        (positive_targets.shape[0]/targets.shape[0])*positive_entropy +
        (negative_targets.shape[0]/targets.shape[0])*negative_entropy)
