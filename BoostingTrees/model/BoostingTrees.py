import numpy as np
import math
import random

class DecisionTree:
    def __init__(self, max_depth=5, feature_indices=None):
        self.max_depth = max_depth
        self.feature_index = None
        self.threshold = None
        self.left_value = None
        self.right_value = None
        self.feature_indices = feature_indices

    def fit(self, X, y):
        best_score = float('inf')
        n_samples, n_features = X.shape
        features = self.feature_indices if self.feature_indices is not None else range(n_features)

        for feature in features:
            thresholds = np.unique(X[:, feature])
            for t in thresholds:
                left_mask = X[:, feature] <= t
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                left_prob = y[left_mask].mean()
                right_prob = y[right_mask].mean()

                gini_left = 1 - (left_prob ** 2 + (1 - left_prob) ** 2)
                gini_right = 1 - (right_prob ** 2 + (1 - right_prob) ** 2)

                gini = (left_mask.sum() * gini_left + right_mask.sum() * gini_right) / n_samples

                if gini < best_score:
                    best_score = gini
                    self.feature_index = feature
                    self.threshold = t
                    self.left_value = left_prob
                    self.right_value = right_prob

    def predict(self, X):
        feature = self.feature_index
        return np.where(X[:, feature] <= self.threshold, self.left_value, self.right_value)


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

    def _log_loss_gradient(self, y_true, y_pred):
        return y_true - self._sigmoid(y_pred)

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
            residual = self._log_loss_gradient(y, y_pred)
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
