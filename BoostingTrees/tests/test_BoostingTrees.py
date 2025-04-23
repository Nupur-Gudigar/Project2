import numpy as np
import pandas as pd
import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.BoostingTrees import GradientBoostingClassifier

def load_data(file_name, label_col):
    file_path = os.path.join(os.path.dirname(__file__), file_name)
    df = pd.read_csv(file_path)
    X = df.drop(columns=[label_col]).values
    y = df[label_col].values
    return X, y

def accuracy_score(y_true, y_pred):
    return (y_true == y_pred).sum() / len(y_true)

def manual_precision(y_true, y_pred):
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    return tp / (tp + fp + 1e-8)

def manual_recall(y_true, y_pred):
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    return tp / (tp + fn + 1e-8)

def manual_f1(precision, recall):
    return 2 * (precision * recall) / (precision + recall + 1e-8)

def manual_log_loss(y_true, y_proba):
    eps = 1e-15
    y_proba = np.clip(y_proba, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_proba) + (1 - y_true) * np.log(1 - y_proba))

def manual_mse(y_true, y_proba):
    return np.mean((y_true - y_proba) ** 2)

def manual_mae(y_true, y_proba):
    return np.mean(np.abs(y_true - y_proba))

def manual_r2(y_true, y_proba):
    ss_res = np.sum((y_true - y_proba) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

def print_classification_metrics(y_true, y_pred, name=""):
    acc = accuracy_score(y_true, y_pred)
    correct = (y_true == y_pred).sum()
    total = len(y_true)
    print(f"\n{name} Metrics:")
    print(f"Accuracy: {acc:.4f} ({correct}/{total})")
    print(f"Sample Predictions: {y_pred[:10]}")
    print(f"True Labels:        {y_true[:10]}")

def test_classification_data():
    X, y = load_data("classification_data.csv", label_col="label")
    model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=2, normalize=False)
    model.fit(X, y)
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    precision = manual_precision(y, y_pred)
    recall = manual_recall(y, y_pred)
    f1 = manual_f1(precision, recall)
    logloss = manual_log_loss(y, y_proba)
    mse = manual_mse(y, y_proba)
    mae = manual_mae(y, y_proba)
    r2 = manual_r2(y, y_proba)

    print_classification_metrics(y, y_pred, name="Classification Test")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Log Loss: {logloss:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")

    assert y_pred.shape == y.shape
    assert accuracy_score(y, y_pred) > 0.5

def test_idempotency():
    X, y = load_data("classification_data.csv", label_col="label")
    model = GradientBoostingClassifier(n_estimators=10, learning_rate=0.1, max_depth=1)
    model.fit(X, y)
    y_pred1 = model.predict(X)

    model.fit(X, y)
    y_pred2 = model.predict(X)

    assert np.allclose(y_pred1, y_pred2), "Model should behave deterministically"

def test_tiny_synthetic():
    np.random.seed(42)
    X = np.random.randn(20, 4)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)

    model = GradientBoostingClassifier(n_estimators=20, learning_rate=0.2, max_depth=1)
    model.fit(X, y)
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)

    print("\nTiny Synthetic Test Accuracy:", acc)
    assert acc > 0.8

def test_few_estimators():
    X, y = load_data("classification_data.csv", label_col="label")
    model = GradientBoostingClassifier(n_estimators=2, learning_rate=0.1, max_depth=1)
    model.fit(X, y)
    y_pred = model.predict(X)
    print_classification_metrics(y, y_pred, name="Few Estimators Test")
    assert accuracy_score(y, y_pred) < 0.9

def test_random_labels():
    X, y = load_data("classification_data.csv", label_col="label")
    np.random.seed(0)
    y_noise = np.random.randint(0, 2, size=len(y))
    model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=2)
    model.fit(X, y_noise)
    y_pred = model.predict(X)
    print_classification_metrics(y_noise, y_pred, name="Random Label Test")
    assert accuracy_score(y_noise, y_pred) < 0.8

def test_normalized_features():
    X, y = load_data("classification_data.csv", label_col="label")
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=2)
    model.fit(X, y)
    y_pred = model.predict(X)
    print_classification_metrics(y, y_pred, name="Normalized Input Test")
    assert accuracy_score(y, y_pred) > 0.7

def test_unbalanced_classes():
    np.random.seed(42)
    X = np.random.randn(1000, 10)
    y = np.zeros(1000, dtype=int)
    y[:100] = 1 
    np.random.shuffle(y)
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=2)
    model.fit(X, y)
    y_pred = model.predict(X)
    print_classification_metrics(y, y_pred, name="Unbalanced Class Test")
    assert accuracy_score(y, y_pred) > 0.8

def test_single_feature():
    np.random.seed(42)
    X = np.random.randn(500, 1)
    y = (X[:, 0] > 0).astype(int)
    model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=1)
    model.fit(X, y)
    y_pred = model.predict(X)
    print_classification_metrics(y, y_pred, name="Single Feature Test")
    assert accuracy_score(y, y_pred) > 0.8

def test_early_stopping_behavior():
    X, y = load_data("classification_data.csv", label_col="label")
    model = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=2, verbose=True)
    model.fit(X, y)
    y_pred = model.predict(X)
    print_classification_metrics(y, y_pred, name="Early Stopping Simulation")
    assert accuracy_score(y, y_pred) > 0.75

def test_ibm_attrition_dataset():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "ibm_attrition.csv"))
    y = df["Attrition"].values
    X = df.select_dtypes(include=[np.number]).drop("EmployeeNumber", axis=1, errors="ignore").values

    # Normalize features
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=2, normalize=False)
    model.fit(X, y)
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    precision = manual_precision(y, y_pred)
    recall = manual_recall(y, y_pred)
    f1 = manual_f1(precision, recall)
    logloss = manual_log_loss(y, y_proba)
    mse = manual_mse(y, y_proba)
    mae = manual_mae(y, y_proba)
    r2 = manual_r2(y, y_proba)

    print_classification_metrics(y, y_pred, name="IBM Attrition Test")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Log Loss: {logloss:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")

    assert y_pred.shape == y.shape
    assert accuracy_score(y, y_pred) > 0.5

def test_moon_data_full_metrics():
    X, y = load_data("moon_classification_data.csv", label_col="label")
    model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=2)
    model.fit(X, y)
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    precision = manual_precision(y, y_pred)
    recall = manual_recall(y, y_pred)
    f1 = manual_f1(precision, recall)
    logloss = manual_log_loss(y, y_proba)
    mse = manual_mse(y, y_proba)
    mae = manual_mae(y, y_proba)
    r2 = manual_r2(y, y_proba)

    print_classification_metrics(y, y_pred, name="Moon Dataset Test")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Log Loss: {logloss:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")

    assert y_pred.shape == y.shape
    assert accuracy_score(y, y_pred) > 0.8


def test_circle_data_full_metrics():
    X, y = load_data("circle_classification_data.csv", label_col="label")
    model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=2)
    model.fit(X, y)
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    precision = manual_precision(y, y_pred)
    recall = manual_recall(y, y_pred)
    f1 = manual_f1(precision, recall)
    logloss = manual_log_loss(y, y_proba)
    mse = manual_mse(y, y_proba)
    mae = manual_mae(y, y_proba)
    r2 = manual_r2(y, y_proba)

    print_classification_metrics(y, y_pred, name="Circle Dataset Test")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Log Loss: {logloss:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")

    assert y_pred.shape == y.shape
    assert accuracy_score(y, y_pred) > 0.8

