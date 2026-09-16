#!/usr/bin/env python3
"""Small demo: train a PyTorch MLP on synthetic tabular data and compute
feature attributions using Captum (Integrated Gradients), SHAP (KernelExplainer),
and permutation importance.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score

# try importing torch and captum; if unavailable we'll fall back to scikit-learn
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader
    from captum.attr import IntegratedGradients
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    IntegratedGradients = None
    torch = None

# We'll define a simple sklearn model if torch isn't present
from sklearn.linear_model import LogisticRegression

try:
    import shap
except Exception:
    shap = None


if TORCH_AVAILABLE:
    class MLP(nn.Module):
        def __init__(self, n_in, hidden=32):
            super().__init__()
            self.fc1 = nn.Linear(n_in, hidden)
            self.fc2 = nn.Linear(hidden, hidden)
            self.out = nn.Linear(hidden, 2)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.out(x)


def train_model(X_train, y_train, n_epochs=50):
    if TORCH_AVAILABLE:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MLP(X_train.shape[1], hidden=64).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        ds = TensorDataset(torch.from_numpy(X_train.astype(np.float32)), torch.from_numpy(y_train.astype(np.int64)))
        dl = DataLoader(ds, batch_size=64, shuffle=True)
        loss_fn = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(n_epochs):
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
        return model
    else:
        # sklearn logistic regression
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        return clf


class TorchSklearnWrapper:
    """Minimal wrapper exposing predict / predict_proba for sklearn utilities.
    Works with either a PyTorch model (if TORCH_AVAILABLE) or a sklearn estimator.
    """

    def __init__(self, model):
        self.model = model
        self.is_torch = TORCH_AVAILABLE and hasattr(model, "eval")
        if self.is_torch:
            self.device = next(model.parameters()).device

    def predict_proba(self, X):
        if self.is_torch:
            self.model.eval()
            with torch.no_grad():
                xb = torch.from_numpy(X.astype(np.float32)).to(self.device)
                logits = self.model(xb)
                probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            return np.vstack([1 - probs, probs]).T
        else:
            return self.model.predict_proba(X)

    def predict(self, X):
        if self.is_torch:
            probs = self.predict_proba(X)[:, 1]
            return (probs >= 0.5).astype(int)
        else:
            return self.model.predict(X)


def integrated_gradients_attribution(model, X_ref, X_target, feature_names):
    if IntegratedGradients is None:
        print("Captum not available; skipping Integrated Gradients")
        return None
    device = next(model.parameters()).device
    ig = IntegratedGradients(model)
    model.eval()
    baseline = torch.from_numpy(X_ref.mean(axis=0).astype(np.float32)).unsqueeze(0).to(device)
    inputs = torch.from_numpy(X_target.astype(np.float32)).to(device)
    attrs, _ = ig.attribute(inputs, baselines=baseline, target=1, return_convergence_delta=True)
    attributions = attrs.detach().cpu().numpy()
    importance = np.mean(np.abs(attributions), axis=0)
    return pd.Series(importance, index=feature_names)


def shap_kernel_attribution(predict_fn, X_background, X_explain, feature_names):
    if shap is None:
        print("shap not available; skipping SHAP")
        return None
    explainer = shap.KernelExplainer(predict_fn, X_background)
    # explain a small batch for speed
    shap_vals = explainer.shap_values(X_explain, nsamples=100)
    # shap_vals is list per class; take class 1
    sv = np.array(shap_vals)
    if sv.ndim == 3:
        class1 = sv[1]
    else:
        class1 = sv
    importance = np.mean(np.abs(class1), axis=0)
    return pd.Series(importance, index=feature_names)


def main():
    out_dir = "feature_attribution_output"
    os.makedirs(out_dir, exist_ok=True)

    X, y = make_classification(n_samples=2000, n_features=10, n_informative=6, random_state=42)
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    model = train_model(X_train, y_train, n_epochs=100)
    wrapper = TorchSklearnWrapper(model)
    if TORCH_AVAILABLE:
        print("Using PyTorch model for attribution")
    else:
        print("Torch unavailable; using sklearn logistic regression")

    # Evaluate
    y_pred = wrapper.predict(X_test)
    print("Test accuracy:", accuracy_score(y_test, y_pred))

    # Permutation importance
    print("Computing permutation importance (may take a moment)...")
    r = permutation_importance(wrapper, X_test, y_test, n_repeats=10, random_state=0, n_jobs=1)
    perm_imp = pd.Series(r.importances_mean, index=feature_names)

    ig_imp = None
    if TORCH_AVAILABLE:
        # Integrated Gradients (on a few instances)
        ig_imp = integrated_gradients_attribution(model, X_train[:50], X_test[:50], feature_names)

    # SHAP Kernel (on a small subset)
    def predict_prob_class1(x):
        return wrapper.predict_proba(np.atleast_2d(x))[:, 1]

    shap_imp = shap_kernel_attribution(predict_prob_class1, X_train[:50], X_test[:20], feature_names)

    # Combine and plot
    df = pd.DataFrame({"permutation": perm_imp})
    if ig_imp is not None:
        df["integrated_gradients"] = ig_imp
    if shap_imp is not None:
        df["shap_kernel"] = shap_imp

    print(df)
    ax = df.plot.bar(title="Feature importance (higher = more important)")
    fig = ax.get_figure()
    fig.tight_layout()
    out_path = os.path.join(out_dir, "feature_importances.png")
    fig.savefig(out_path)
    print("Saved plot to", out_path)


if __name__ == "__main__":
    main()
