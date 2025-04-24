
# Gradient Boosting Classifier from Scratch

This project implements a gradient boosting classifier from first principles, based on Sections 10.9–10.10 of "The Elements of Statistical Learning (2nd Edition)".

## What does the model do and when should it be used?

The model is a binary classifier that uses decision trees as weak learners and fits them sequentially to correct errors of previous models using gradient descent on a logistic loss function. It should be used for structured data with complex relationships where ensemble methods outperform single estimators.

## How did you test your model?

We tested the implementation on the real-world IBM employee attrition dataset. We manually computed evaluation metrics such as:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC
- Log Loss
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R² Score

We also used 80/20 train-test splitting and ensured robustness through reproducible shuffling.

## What parameters have you exposed for tuning?

- `n_estimators`: Number of boosting iterations
- `learning_rate`: Shrinkage factor for updates
- `max_depth`: Depth of individual decision trees

### Example usage:
```bash
python tests/test_model.py
```

## Are there specific inputs your model has trouble with?

- Highly imbalanced data may reduce recall without tuning.
- Noisy or irrelevant features may lead to overfitting if trees are deep.
- Missing values or non-numeric categorical columns need preprocessing.

Given more time, feature encoding, early stopping, and hyperparameter tuning would be incorporated.

## Files & Structure
```
Project2/
├── data/
│   ├── generate_data.py
│   ├── ibm_attrition.csv
│   └── synthetic_data.csv
├── models/
│   ├── gradient_boost.py
│   └── regression_tree.py
├── tests/
│   ├── __init__.py
│   └── test_model.py
├── utils/
│   └── metrics.py
├── confusion_matrix.png
├── README.md
└── roc_curve.png
```

## Credits

This project was completed as part of **CS 584 – Machine Learning** coursework.

| Name              | A-Number      | Email                          |
|-------------------|---------------|--------------------------------|
| Nupur Gudigar     | A20549865     | ngudigar@hawk.iit.edu          |
| Zaigham Shaikh    | A20554429     | zshaikh4@hawk.iit.edu          |
| Nehil Joshi       | A20554381     | njoshi20@hawk.iit.edu          |
| Riddhi Das        | A20582829     | rdas8@hawk.iit.edu             |
