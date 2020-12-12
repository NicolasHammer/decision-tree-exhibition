# Step 0: Import the necessary functions
from from_scratch.decision_tree import DecisionTree, information_gain
from from_scratch.evaluation_metrics import f1_measure, precision_and_recall, confusion_matrix, accuracy
from from_scratch.import_data import load_data, train_test_split

# Step 1: Import diabetes.csv with load_data
features, targets, attribute_names = load_data("diabetes.csv")
train_features, train_targets, test_features, test_targets = train_test_split(features, targets)

# Step 2: Fit a decisoin tree to the training data
learner = DecisionTree(attribute_names)
learner.fit(train_features, train_targets)

# Step 3: 
predictions = learner.predict(test_features)

confusion_mat = confusion_matrix(test_targets, predictions)
accuracy_num = accuracy(test_targets, predictions)
precision, recall = precision_and_recall(test_targets, predictions)
f1_measure_num = f1_measure(test_targets, predictions)