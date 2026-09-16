# Feature attribution demo

This folder contains a minimal demo that trains a small PyTorch MLP on synthetic tabular data and computes feature attributions using:

- Captum (Integrated Gradients)
- SHAP (KernelExplainer)
- scikit-learn permutation importance

Run:

```bash
python3 feature_attribution/attribution_demo.py
```

Install dependencies first:

```bash
pip install -r feature_attribution/requirements.txt
```

Output is stored in `feature_attribution_output/` (includes a PNG with feature importance comparisons).
