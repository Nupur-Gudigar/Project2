import numpy as np
import pandas as pd

np.random.seed(42)
n_samples = 1000
n_features = 10

X = np.random.randn(n_samples, n_features)

weights = np.random.randn(n_features)
linear_combination = X @ weights
probabilities = 1 / (1 + np.exp(-linear_combination)) 
y = (probabilities > 0.5).astype(int)

data = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
data["label"] = y
data.to_csv("BoostingTrees/tests/classification_data.csv", index=False)
print("classification_data.csv generated successfully.")
