import numpy as np
import statistics as stats
from itertools import repeat

class Tree():
    """
     A tree structure with multiple branches at eah node.  The default values correspond
     to the root node. 

     Member Variables
     -----------------------------
     value (float or int) - the data value if a leaf; otherwise, it is the split value\n
     attribute_name (str) - name of the attribute that the tree splits the data on\n
     attribute_index (float) - index of the attribute in the feature vector\n
     branches (list) - list of Tree objects.  Since this is being used for a binary decision tree,\
                       this should either be 2 or 0.
     """
    def __init__(self, value : (float or int) = None, attribute_name : str = "root",
     attribute_index : int = None, branches : list = None):
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
    def __init__(self, attribute_names : list):
        self.attribute_names = attribute_names
        self.tree = None
    
    def _check_input(self, features : np.ndarray) -> None:
        """
        Check that the number of attributes in the features matrix matches up with the length of 
        self.attribute_names.
        """
        if features.shape[0] != len(self.attribute_names):
            raise ValueError("Number of features and number of attribute names must match!")
    
    def fit(self, features : np.ndarray, targets : np.ndarray) -> None:
        """
        Fit this decision tree to the features training data using the ID3 algorithm.  This function
        calls fit_recursive in order to recurse down the decision tree and use all attributes until
        the tree is either perfectly split or all the attributes are exhausted.

        Parameters
        ----------
        features (np.ndarray) - the features we want to fit our model to\n
        targets (np.ndarray) - the targets which correspond to these features
        """
        self._check_input(features)
        feature_indicies = list(range(0, features.shape[0]))
        self.tree = self.fit_recursive(features, targets, feature_indicies)

    def fit_recursive(self, features : np.ndarray, targets : np.ndarray, unvisited_features : list) -> Tree:
        """
        First discern whether or not the there are attributes left to split upon; if there are not,
        assume that the node is a leaf and return a tree whose root value is the mode of the targets.  If 
        there still exist attributes to split upon, check whether all the class labels are 1 or 0; if so,
        return a tree whose root value is the the unanimous value.  In the recursive case, find the attribute
        best splits the tree based on information gain (check each split_value as well) and then make the split.

        Parameters
        ----------
        features (np.ndarray) - the features we want to fit our model to\n
        targets (np.ndarray) - the targets which correspond to these features\n
        unvisited_features (np.ndarray) - an array of indicies corresponding to features yet to be visited

        Output
        ------
        fit_tree (Tree) - a tree which is fit to the reamining, unvisited features data
        """
        fit_tree = Tree()
        
        # Base case 1 where all class labels are 1 or 0
        sum_targets = np.sum(targets)
        if sum_targets == targets.shape[1]:
            fit_tree.value = 1
            return fit_tree
        elif sum_targets == 0:
            fit_tree.value = 0
            return fit_tree

        # Base case 2 with no reamining unvisited features
        if len(unvisited_features) == 0:
            fit_tree.value = stats.mode(targets[0])
            return fit_tree
        
        # Recursive case
        max_gain = None
        highest_info_index = None
        split_value = None
        ## Iterate over all unvisited features
        for feature_index in unvisited_features: 
            ### Map over all split values and find the one that maximizes information gain
            best_split_index, best_info_gain = max(
                enumerate(
                    map(information_gain, repeat(features), repeat(targets), repeat(feature_index),
                    features[feature_index,:])),
                    key = (lambda x : x[1]))
            
            ### Override best values if better than max_gain
            if max_gain == None or best_info_gain > max_gain:
                max_gain = best_info_gain
                highest_info_index = feature_index
                split_value = features[feature_index, best_split_index]

        fit_tree.attribute_index = highest_info_index
        fit_tree.attribute_name = self.attribute_names[highest_info_index]
        fit_tree.value = split_value

        ## Append tree branches
        ### choose samples (columns) where the attribute is less than the split value
        fit_tree.branches.append(self.fit_recursive(
            features = features[:, features[fit_tree.attribute_index, :] < split_value], 
            targets = targets[:, features[fit_tree.attribute_index, :] < split_value],
            unvisited_features = [x for x in unvisited_features if x != highest_info_index]))
        ### choose samples (columns) where the attribute is greater than or equal to the split value
        fit_tree.branches.append(self.fit_recursive(
            features = features[:, features[fit_tree.attribute_index, :] >= split_value],
            targets = targets[:, features[fit_tree.attribute_index, :] >= split_value],
            unvisited_features = [x for x in unvisited_features if x != highest_info_index]))

        return fit_tree
    
    def predict(self, features : np.ndarray) -> np.ndarray:
        """
        Predict the values of the test data.  This function calls predict_recursive to recurse down the tree.
        """
        self._check_input(features)
        return np.array([[
            self.predict_recursive(features, self.tree, test_example) for test_example in range(0, features.shape[1])
        ]])

    def predict_recursive(self, features : np.ndarray, current_tree : Tree, test_example : int) -> int:
        """
        Recurse down the decision tree and find the value at the leaf node whose path corresponds to the
        qualities of our test example.
        """
        # Base Case
        if len(current_tree.branches) == 0:
            return current_tree.value
        else:
            current_tree = (current_tree.branches[0] 
                if features[current_tree.attribute_index, test_example] < current_tree.value
                else current_tree.branches[1])
            
            return self.predict_recursive(features, current_tree, test_example)
    
    def visualize(self, branch : Tree = None, level : int = 0) -> None:
        """
        Visualize a tree recursively.
        """
        # Edge case
        if not branch:
            branch = self.tree
        
        # Print tree
        tab_level = " " * level
        val = branch.value if branch.value is not None else 0
        print(f"{level}: {tab_level}{branch.attribute_name} == {val}")

        # Recurse down tree
        for branch in branch.branches:
            self.visualize(branch, level + 1)

def entropy(targets : np.ndarray) -> float:
    """
    Returns entropy (https://en.wikipedia.org/wiki/ID3_algorithm#The_ID3_metrics) given features
    for each example.  The entropy funcion is called H(S).
    """
    _, counts = np.unique(targets, return_counts=True)
    proportion = counts/np.sum(counts)
    return -np.sum(proportion*np.log2(proportion))

def information_gain(features : np.ndarray, targets : np.ndarray, attribute_index : int, split_value : (int or float)) -> float:
    """
    Caculates information gain (https://en.wikipedia.org/wiki/ID3_algorithm#The_ID3_metrics).

    Parameters
    ----------
    features (np.ndarray) - features for each example\n
    targets (np.ndarray) - labels corresponding to each example\n
    attribute_index (int) - index corresponding to which colun of features to take when computing
                            information gain
    split_value (int or float) - the value to split by

    Output
    ------
    information_gain (float) - information gain when the features are split on attribute_index
    """
    less_targets = targets[:, features[attribute_index, :] < split_value] 
    more_targets = targets[:, features[attribute_index, :] >= split_value]
    less_entropy = entropy(less_targets) # H(targets | feature < split_value)
    more_entropy = entropy(more_targets) # H(targets | feature >= split_value)

    prior_entropy = entropy(targets) # H(targets)

    # H(targets) - (Pr(feature < split_value)*H(targets | feature < split_value) + Pr(feature >= split_value)*H(targets | feature >= split_value))
    return prior_entropy - (
        (less_targets.shape[1]/targets.shape[1])*less_entropy +
        (more_targets.shape[1]/targets.shape[1])*more_entropy)