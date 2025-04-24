
# Project 2: Gradient Boosting Trees

---

## Table of Contents

1. [Introduction](#introduction)
2. [Overview](#overview)
3. [How the Model Works](#how-the-model-works)
4. [Getting Started](#getting-started)
5. [Test Coverage](#test-coverage)
6. [Summary of Test Scenarios](#summary-of-test-scenarios)
7. [Visual Output Samples](#visual-output-samples)
8. [Adjustable Parameters](#adjustable-parameters)
9. [Limitations & Future Work](#limitations--future-work)
10. [Q&A](#qa)
11. [Team Members](#team-members)

---

## Introduction

This repository presents a from-scratch implementation of a **Gradient Boosting Classifier**, based on logistic loss minimization via additive tree models. Inspired by Sections 10.9–10.10 of *The Elements of Statistical Learning*, this model builds gradient boosting trees step-by-step for classification.

---

## Overview

The model was designed to classify binary outcomes using iterative refinement through weak learners (shallow trees). This repo includes:

- A full gradient boosting classifier
- Utilities for metric calculations and visualization
- Data generation for synthetic datasets
- Evaluation on real-world (IBM Attrition) and synthetic datasets

---

## How the Model Works

1. Start with a constant prediction (log-odds of class 1).
2. Compute the negative gradient of the logistic loss.
3. Train a regression tree on this gradient.
4. Add this new model to the ensemble.
5. Repeat for a fixed number of boosting rounds (`n_estimators`).

Each decision tree splits based on variance reduction (as a proxy for classification gain), and outputs a real-valued score added to the logits.

---

## Getting Started

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Nupur-Gudigar/Project2.git
cd Project2/Boosting Trees
```

### 2️⃣ Install dependencies

```bash
pip install numpy matplotlib pandas seaborn
```

### 3️⃣ Run the model

```bash
python tests/test_BoostingTrees.py
```

### 4️⃣ Generate synthetic datasets (optional)

```bash
python generate_data.py
```

---

## Test Coverage

This repo includes tests using both real-world and synthetic data.

### Included datasets:

- ✅ `ibm_attrition.csv`
- ✅ `classification_data.csv`
- ✅ `circle_classification_data.csv`
- ✅ `moon_classification_data.csv`

Each dataset helps validate model behavior on linearly separable, non-linear, and imbalanced cases.

---

## Summary of Test Scenarios

| Test Scenario          | Dataset                   | Description                                    |
|------------------------|---------------------------|------------------------------------------------|
| Real-world data        | `ibm_attrition.csv`       | HR Attrition classifier using actual company data |
| Synthetic circle data  | `circle_classification_data.csv` | Non-linear separation test                      |
| Synthetic moon data    | `moon_classification_data.csv`   | Interleaved half-moons pattern                 |
| Manual split strategy  | Balanced class training   | Ensures classes appear in train/test properly |
| Visual confirmation    | All datasets              | Saves `confusion_matrix.png` and `roc_curve.png` |

---

## Visual Output Samples

After successful model run, these are generated automatically:

- 📊 `confusion_matrix.png`: Heatmap of predicted vs actual labels
- 📈 `roc_curve.png`: ROC performance curve with thresholds

---

## Adjustable Parameters

All configurable in `GradientBoostingClassifier`:

- `n_estimators`: Number of boosting rounds
- `learning_rate`: Learning rate (shrinkage)
- `max_depth`: Depth of each weak learner
- `min_samples_split`: Optional split threshold
- `metrics.py`: Includes all scoring logic (manually implemented)

---

## Limitations & Future Work

### Known Limitations

- Manual oversampling instead of using class weights
- No support for multi-class targets
- No pruning or early stopping
- No cross-validation or automatic parameter search

### Possible Improvements

- Implement pruning for better generalization
- Add early stopping using validation loss
- Introduce automatic hyperparameter tuning
- Add `feature_importance_` tracking
- Export tree structures in a visual format

---

## Q&A

**What does the model do?**  
→ Predicts binary outcomes using gradient-boosted decision trees.

**How did you test it?**  
→ With synthetic and real data, visual and numeric outputs, stratified sampling, and multiple metric evaluations.

**What can I customize?**  
→ You can change the depth, number of trees, learning rate, and dataset.

**Any problems you faced?**  
→ Class imbalance needed manual upsampling. ROC was initially negative due to improper FPR ordering. All fixed now.

---

## Team Members

| Name              | A-Number    | Email                        |
|-------------------|-------------|------------------------------|
| Nupur Gudigar     | A20549865   | ngudigar@hawk.iit.edu        |
| Zaigham Shaikh    | A20554429   | zshaikh4@hawk.iit.edu        |
| Nehil Joshi       | A20554381   | njoshi20@hawk.iit.edu        |
| Riddhi Das        | A20582829   | rdas8@hawk.iit.edu           |
