# train_and_save_model.py
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load a small demo dataset
X, y = load_diabetes(return_X_y=True, as_frame=True)

# Train/test split (just to be clean; we only need a model file)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Fit a simple model
model = LinearRegression().fit(X_train, y_train)

# Save for the Flask app
joblib.dump(model, "model.pkl")
print("Saved model.pkl")
