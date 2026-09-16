#!/usr/bin/env python3
"""Minimal demo: train sklearn logistic regression on synthetic tabular data
and compute permutation importance—no pandas, matplotlib, or torch required.
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score

def main():
    print("Generating synthetic tabular data...")
    X, y = make_classification(n_samples=2000, n_features=10, n_informative=6, random_state=42)
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    
    print("Training logistic regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")
    
    print("\nComputing permutation importance...")
    r = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=0, n_jobs=1)
    
    print("\nFeature Importances (permutation):")
    for fname, imp in zip(feature_names, r.importances_mean):
        print(f"  {fname}: {imp:.6f}")
    
    print("\nCoefficients (linear model):")
    for fname, coef in zip(feature_names, model.coef_[0]):
        print(f"  {fname}: {coef:.6f}")
    
    print("\n✓ Demo completed successfully!")
    print("For full feature attribution (SHAP, Integrated Gradients),")
    print("install with: pip install pandas matplotlib shap")
    print("Then run: python3 feature_attribution/attribution_demo.py")

if __name__ == "__main__":
    main()
