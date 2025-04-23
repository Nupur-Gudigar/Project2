import numpy as np
import math
import random

class DecisionTree:
    def __init__(self, max_depth=5, feature_indices=None):
        self.max_depth = max_depth
        self.feature_indices = feature_indices
        self.tree = None

    class Node:
        def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
            self.feature_index = feature_index
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

    def _weighted_mse(self, y):
        if len(y) == 0:
            return 0
        mean = np.mean(y)
        return np.mean((y - mean) ** 2)

    def _best_split(self, X, y, features):
        best_score = float('inf')
        best_feature = None
        best_threshold = None
        best_left_indices = None
        best_right_indices = None

        for feature in features:
            thresholds = np.unique(X[:, feature])
            for t in thresholds:
                left_indices = np.where(X[:, feature] <= t)[0]
                right_indices = np.where(X[:, feature] > t)[0]

                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue

                left_mse = self._weighted_mse(y[left_indices])
                right_mse = self._weighted_mse(y[right_indices])
                weighted_mse = (len(left_indices) * left_mse + len(right_indices) * right_mse) / len(y)

                if weighted_mse < best_score:
                    best_score = weighted_mse
                    best_feature = feature
                    best_threshold = t
                    best_left_indices = left_indices
                    best_right_indices = right_indices

        return best_feature, best_threshold, best_left_indices, best_right_indices

    def _build_tree(self, X, y, depth):
        if depth == self.max_depth or len(set(y)) == 1:
            return self.Node(value=np.mean(y))

        n_samples, n_features = X.shape
        features = self.feature_indices if self.feature_indices is not None else range(n_features)

        feature, threshold, left_indices, right_indices = self._best_split(X, y, features)

        if feature is None:
            return self.Node(value=np.mean(y))

        left_child = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return self.Node(feature_index=feature, threshold=threshold, left=left_child, right=right_child)

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, 0)

    def _predict_row(self, row, node):
        if node.value is not None:
            return node.value
        if row[node.feature_index] <= node.threshold:
            return self._predict_row(row, node.left)
        else:
            return self._predict_row(row, node.right)

    def predict(self, X):
        return np.array([self._predict_row(row, self.tree) for row in X])


class GradientBoostingClassifier:
    def __init__(self, n_estimators=150, learning_rate=0.05, max_depth=5, normalize=True, verbose=False, early_stopping_rounds=None, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.normalize = normalize
        self.verbose = verbose
        self.early_stopping_rounds = early_stopping_rounds
        self.max_features = max_features

        self.trees = []
        self.init_prediction = None
        self.mean = None
        self.std = None

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _log_loss_grad_hess(self, y_true, y_pred):
        p = self._sigmoid(y_pred)
        grad = p - y_true
        hess = p * (1 - p)
        hess = np.clip(hess, 1e-6, 1.0)
        return grad, hess

    def _normalize_features(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-8
        return (X - self.mean) / self.std

    def _apply_normalization(self, X):
        return (X - self.mean) / self.std

    def _get_feature_subset(self, n_features):
        if self.max_features == 'sqrt':
            k = max(1, int(math.sqrt(n_features)))
        elif self.max_features == 'log2':
            k = max(1, int(math.log2(n_features)))
        elif isinstance(self.max_features, int):
            k = min(n_features, self.max_features)
        else:
            k = n_features
        return sorted(random.sample(range(n_features), k))

    def fit(self, X, y):
        random.seed(42)

        if self.normalize:
            X = self._normalize_features(X)

        self.init_prediction = np.full(y.shape, 0.0)
        y_pred = self.init_prediction.copy()

        best_acc = 0
        rounds_since_improvement = 0
        tolerance = 1e-4

        for i in range(self.n_estimators):
            grad, hess = self._log_loss_grad_hess(y, y_pred)
            residual = -grad / hess

            feature_subset = self._get_feature_subset(X.shape[1])
            tree = DecisionTree(max_depth=self.max_depth, feature_indices=feature_subset)
            tree.fit(X, residual)
            update = tree.predict(X)

            y_pred += self.learning_rate * update
            self.trees.append(tree)

            acc = ((self._sigmoid(y_pred) >= 0.5) == y).mean()
            if self.verbose:
                print(f"[Round {i+1}] Accuracy: {acc:.4f}")

            if self.early_stopping_rounds:
                if acc > best_acc + tolerance:
                    best_acc = acc
                    rounds_since_improvement = 0
                else:
                    rounds_since_improvement += 1
                    if rounds_since_improvement >= self.early_stopping_rounds:
                        if self.verbose:
                            print(f"Early stopping at round {i+1} (best acc: {best_acc:.4f})")
                        break

    def predict_proba(self, X):
        if self.normalize:
            X = self._apply_normalization(X)

        y_pred = self.init_prediction.copy()
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        proba = self._sigmoid(y_pred)
        return np.vstack((1 - proba, proba)).T

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)
