import numpy as np
from utils import *

class DecisionTreeClassifier:
    def __init__(self, max_depth=10, categorical_features=[]):
        self.max_depth = max_depth

        # Initialize the binary tree
        self.root = None

        # Denote certain columns as categorical
        self.categorical_features = categorical_features

    def find_best_split(self, X: np.ndarray, y: np.ndarray):
        # Keep track of best split
        best_gini = np.inf
        best_split_info = None # In the form (feature_index, split_threshold, split_type)

        # Traverse through each feature
        for i, col in enumerate(X.T):
            unique = np.unique(col) # Get all unique elements
            
            # Don't try if there's only one unique element
            if len(unique) == 1:
                continue

            # Categorical type
            is_categorical = (i in self.categorical_features) or isinstance(col[0], str)
            if is_categorical:

                for category in unique:
                    mask = (col == category)

                    # Get weighted avg
                    weighted_avg = weighted_gini(y, mask)

                    # Update best category
                    if weighted_avg < best_gini:
                        best_gini = weighted_avg
                        best_split_info = (i, category, "categorical")
            else:
                # Sort the numerical array unique values
                sorted_vals = np.sort(unique)

                # Calculate pairs of medians:
                medians = (sorted_vals[:-1] + sorted_vals[1:]) / 2

                # Find best split among medians
                for median in medians:
                    mask = (col < median)

                    # Get weighted avg
                    weighted_avg = weighted_gini(y, mask)

                    # Update best category
                    if weighted_avg < best_gini:
                        best_gini = weighted_avg
                        best_split_info = (i, median, "numerical")

        return best_split_info

    def build_tree(self, X: np.ndarray, y: np.ndarray, depth = 1):
        unique, counts = np.unique(y, return_counts=True)
        majority_class = unique[np.argmax(counts)]

        # Don't go past max_depth
        if depth > self.max_depth:
            return DTNode(pred=majority_class)
        
        # If only one category in y, don't go further
        if len(unique) == 1:
            return DTNode(pred=majority_class)
        
        # Find the best split at the current state
        split = self.find_best_split(X, y)

        # If no best split, make leaf
        if split == None:
            return DTNode(pred=majority_class)

        # Create a new node
        feature_idx, threshold, split_type = split
        node = DTNode(feature_idx, threshold, split_type)

        # Find the best splits for left and right node
        if split_type == "categorical":
            mask = (X[:, feature_idx] == threshold)
        else:
            mask = (X[:, feature_idx] < threshold)
        
        # Get examples and features for left and right node
        X_left, y_left = X[mask], y[mask]
        X_right, y_right = X[~mask], y[~mask]

        # Recursively add nodes
        node.left = self.build_tree(X_left, y_left, depth + 1)
        node.right = self.build_tree(X_right, y_right, depth + 1)

        # Return the final node
        return node
    
    def fit(self, X, y):
        """
        Builds optimal decision tree based on data
        """
        self.root = self.build_tree(X, y)

    def predict(self, X):
        """
        Predits a group of examples
        """
        return np.array([self.root.predict(x) for x in X])

class RandomForestClassifier:
    def __init__(self, max_num_tree, max_depth, categorical_values):
        self.trees = [DecisionTreeClassifier(max_depth=max_depth, categorical_features=categorical_values) for i in range(max_num_tree)]

    def fit(self, X, y):
        for i, tree in enumerate(self.trees):
            # Verbose trainning
            if i % 10 == 0:
                if i != 0:
                    print(". [✔]")
                print(f'Training tree {i+1} to {i+10}\t', end="")
            else:
                print(".", end="")

            X_boot, y_boot = bootstrap_data(X, y)
            tree.fit(X_boot, y_boot)

        # Complete verbose
        print(". [✔]")

    def predict(self, X):
        # Get predictions from each tree for each observation
        votes = np.array([tree.predict(X) for tree in self.trees])

        # Columns will be the votes for each observation
        predicted_classes = np.apply_along_axis(get_majority_class, axis=0, arr=votes)
        print(predicted_classes)
        return predicted_classes

class LinearDiscriminantAnalysis:
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # Here, we fit the data immediately as we have a closed form
        m, n = X.shape
        classes = len(np.unique(y))
        
        self.phi = np.mean(y)

        self.mu = np.zeros((classes, n))
        for i in range(classes):
            # Get all samples that match the class
            class_samples = X[y == i]
            self.mu[i] = np.mean(class_samples, axis=0)
        
        self.sigma = np.zeros((n,n))
        for i in range(m):
            # Compute the difference
            diff = (X[i] - self.mu[y[i]]).reshape(-1, 1)
            intermediate = diff @ diff.T
            self.sigma += intermediate
        self.sigma /= m

    def predict(self, X: np.ndarray):
        # Perform bayes

class UnivariateLinearRegression:
    def __init__(self, learning_rate):
        self.lr = learning_rate
        self.w = 0
        self.b = 0

    # Update weights with mse loss
    def update_weights(self, x: np.ndarray, y: np.ndarray):
        # Length of examples
        n = len(y)

        # Get predictions
        y_hat = self.w * x + self.b

        # Update (Calc derivatives then new equation)
        dw = (np.sum((y_hat - y) * x)) / n
        db = (np.sum(y_hat - y)) / n

        self.w -= self.lr * dw
        self.b -= self.lr * db

    # Fit the function to data
    def fit(self, X: np.ndarray, y: np.ndarray, epochs=1000, verbose=False):
        # Update weights until reach convergence or ran a set number of times (epislon)
        for i in range(epochs):
            # Update the weights
            self.update_weights(X, y)

            # Report loss
            loss = mse_loss(y, self.w * X + self.b)

            if verbose:
                print(f'[{i+1}]: Loss: {loss:.4f}')

class LogisticRegression:
    def __init__(self, num_features, learning_rate):
        self.lr = learning_rate
        self.w = np.zeros((num_features, 1))
        self.b = 0

        # For adam gradient descent
        self.w_m = 0
        self.w_v = 0
        self.b_m = 0
        self.b_v = 0
        self.beta1 = 0.9
        self.beta2 = 0.999

    def forward(self, X: np.ndarray):
        # Get linear component (z) and add a quadratic component (feature scaling)
        z = (X @ self.w) + self.b

        # Apply sigmoid
        return 1 / (1 + np.exp(-z))

    def update_weights(self, X: np.ndarray, y: np.ndarray, iteration: int):
        n = len(y) # Get length of examples

        # Get predictions
        y_hat = self.forward(X)

        # Perform ADAM gd for each feature
        dw = (1/n) * (X.T @ (y_hat - y))
        db = (1/n) * np.sum(y_hat - y)

        # Adam momentums
        self.w_m = self.beta1 * self.w_m + (1 - self.beta1) * dw # SGD + Momentum
        self.w_v = self.beta2 * self.w_v + (1 - self.beta2) * dw * dw # RMS Prop

        self.b_m = self.beta1 * self.b_m + (1 - self.beta1) * db # SGD + Momentum
        self.b_v = self.beta2 * self.b_v + (1 - self.beta2) * db * db # RMS Prop

        # Unbias them
        self.w_m_hat = self.w_m / (1 - self.beta1 ** iteration)
        self.w_v_hat = self.w_v / (1 - self.beta2 ** iteration)

        self.b_m_hat = self.b_m / (1 - self.beta1 ** iteration)
        self.b_v_hat = self.b_v / (1 - self.beta2 ** iteration)

        # Apply GD
        self.w -= (self.lr * self.w_m_hat) / (np.sqrt(self.w_v_hat) + 1e-7)
        self.b -= (self.lr * self.b_m_hat) / (np.sqrt(self.b_v_hat) + 1e-7)

    # Fit the data onto the logistic regression function
    def fit(self, X: np.ndarray, y: np.ndarray, epochs=1000, verbose=False):
        for i in range(epochs):
            # Perform gradient descent
            self.update_weights(X, y, i + 1) # plus 1 to avoid divide by zero

            # Report loss if verbose
            if verbose and i % 100 == 0:
                loss = BCELoss(y, self.forward(X))
                print(f'[{i}]: Loss: {loss:.4f}')

def pca(X: np.ndarray, epsilon: float):
    """
    Be sure that the values here are normalized.
    """
    # Compute covariance matrix
    n = X.shape[0]
    cov = (X.T @ X) / n

    # Compute single value decomposition of the covariance matrix
    u, s, _ = np.linalg.svd(cov)

    # Find the first k features that have relative of less than epsilon (or greater than 1 - epsilon)
    threshold = 1 - epsilon
    total_variance = np.sum(s)
    cumulative_variance = np.cumsum(s) / total_variance
    best_k = np.argmax(cumulative_variance >= threshold) + 1

    # Get first k columns in U, then return modified dataset
    u_reduce = u[:, :best_k]
    z = X @ u_reduce

    return z

class LinearSVM:
    def __init__(self, num_features, learning_rate):
        self.lr = learning_rate
        self.w = np.zeros((num_features, 1))
        self.b = 0

        # For adam gradient descent
        self.w_m = 0
        self.w_v = 0
        self.b_m = 0
        self.b_v = 0
        self.beta1 = 0.9
        self.beta2 = 0.999

    def forward(self, X: np.ndarray):
        return (X @ self.w) + self.b
    
    def predict(self, X: np.ndarray):
        return (self.forward(X) >= 0).astype(int)

    def update_weights(self, X: np.ndarray, y: np.ndarray, iteration: int, C = 1.0):
        n = len(y) # Get length of examples

        # Get predictions
        y_hat = self.forward(X)
        condition = y * y_hat >= 1

        # Extract only the few gradients that have violated the margins
        y_reshaped = y[:, np.newaxis] # make into (n, 1)
        hinge_grad_w = np.where(condition[:, np.newaxis], 0, -C * X.T @ y)

        dw = self.w + np.mean(hinge_grad_w, axis=0)
        db = np.mean(np.where(condition, 0, -C * y), axis=0)

        # Adam momentums
        self.w_m = self.beta1 * self.w_m + (1 - self.beta1) * dw # SGD + Momentum
        self.w_v = self.beta2 * self.w_v + (1 - self.beta2) * dw * dw # RMS Prop

        self.b_m = self.beta1 * self.b_m + (1 - self.beta1) * db # SGD + Momentum
        self.b_v = self.beta2 * self.b_v + (1 - self.beta2) * db * db # RMS Prop

        # Unbias them
        self.w_m_hat = self.w_m / (1 - self.beta1 ** iteration)
        self.w_v_hat = self.w_v / (1 - self.beta2 ** iteration)

        self.b_m_hat = self.b_m / (1 - self.beta1 ** iteration)
        self.b_v_hat = self.b_v / (1 - self.beta2 ** iteration)

        # Apply GD
        self.w -= (self.lr * self.w_m_hat) / (np.sqrt(self.w_v_hat) + 1e-7)
        self.b -= (self.lr * self.b_m_hat) / (np.sqrt(self.b_v_hat) + 1e-7)

    # Fit the data onto the logistic regression function
    def fit(self, X: np.ndarray, y: np.ndarray, epochs=1000, verbose=False):
        for i in range(epochs):
            # Perform gradient descent
            self.update_weights(X, y, i + 1) # plus 1 to avoid divide by zero

            # Report loss if verbose
            if verbose and i % 100 == 0:
                loss = HingeLoss(self.w, y, self.forward(X))
                print(f'[{i}]: Loss: {loss}')
