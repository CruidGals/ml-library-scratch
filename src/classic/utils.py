import numpy as np

##################################################
# Decision Tree Utils
##################################################
def gini(examples: np.ndarray):
    """
    Given an examples array holding the classes of each example,
    calculate the gini impurity of the current node.
    """
    N = len(examples)

    # Get all unique classes
    _, counts = np.unique(examples, return_counts=True)

    # Find proportions and apply gini formula
    probabilities = counts / N

    return 1 - np.sum(probabilities**2)

def weighted_gini(y, mask):
    N = len(y)

    # Get the positive/negative (left/right) examples
    y_left = y[mask]
    y_right = y[~mask]

    # Calculate impurities
    gini_left = gini(y_left)
    gini_right = gini(y_right)

    # Calculate weighted average
    weighted_avg = (len(y_left) / N) * gini_left + (len(y_right) / N) * gini_right

    return weighted_avg

class DTNode:
    def __init__(self, feature=None, threshold=None, mode="numerical", pred=None, left=None, right=None):
        # Store split features
        self.feature = feature
        self.threshold = threshold
        self.mode = mode
        
        self.pred = pred

        # Matches split condition, go left
        self.left: DTNode | None = left

        # Fails split condition, go right
        self.right: DTNode | None = right

    def is_leaf(self):
        return self.left == self.right == None

    # Go down the nodes until find a leaf node--then predict
    def predict(self, x):
        if self.is_leaf():
            return self.pred
        
        if self.mode == "numerical":
            return self.left.predict(x) if x[self.feature] < self.threshold else self.right.predict(x)
        else:
            return self.left.predict(x) if x[self.feature] == self.threshold else self.right.predict(x)

##################################################
# Random Forest Utils
##################################################

def bootstrap_data(X, y):
    N = X.shape[0]

    # Choose N observations with replacements
    gen = np.random.default_rng() # Don't set seed since we'll be calling this multiple times
    row_indices = gen.integers(low=0, high=N, size=N)

    # Get data from bootstrapped row indices
    bootstrapped_X = X[row_indices]
    bootstrapped_y = y[row_indices]

    return bootstrapped_X, bootstrapped_y

def get_majority_class(y):
    unique, counts = np.unique(y, return_counts=True)
    majority_class = unique[np.argmax(counts)]
    return majority_class

##################################################
# Linear Regression Utils
##################################################

def mse_loss(targets: np.ndarray, preds: np.ndarray):
    return (np.sum((preds - targets) ** 2)) / len(targets)

##################################################
# Logistic Regression Utils
##################################################

def BCELoss(y: np.ndarray, y_hat: np.ndarray):
    # Clip y_hat to be between epsilon and 1-epsilon
    # This prevents log(0)
    y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

##################################################
# SVM Utils
##################################################

def HingeLoss(weights: np.ndarray, y: np.ndarray, y_hat: np.ndarray, C = 1.0):
    loss_hinge = np.maximum(0, 1 - y * y_hat)
    return (1/2) * (np.dot(weights.T, weights)) + C * np.sum(loss_hinge)