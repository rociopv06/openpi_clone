"""
Deep analysis of SAE interpretation results.

Goes beyond visualize_sae.py by digging into:
1. Feature-proprioception correlations (what robot states activate each feature?)
2. Joint activation profiles (for action-correlated features, which joint angles trigger them?)
3. Action prediction profiles (what actions are predicted at top-activating examples?)
4. Feature clustering (which features are semantically related?)
5. Per-feature summary cards for the most interesting features

Usage:
    python deep_analyze_sae.py

Then scp results back locally:
    tar czf deep_analysis.tar.gz viz_deep_*.png deep_analysis_report.txt
    scp rvaldes6@login-ice.pace.gatech.edu:.../deep_analysis.tar.gz .
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

# ── CONFIG ────────────────────────────────────────────────────────────────
DEFAULT_OUTPUT_DIR = "/storage/project/r-agarg35-0/rvaldes6/sae_interpretations/pi05_libero_expert_layer_0_k32_exp4"
# ──────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
args = parser.parse_args()

out = Path(args.output_dir)
assert out.exists(), f"Output dir not found: {out}"

# ── Load data ─────────────────────────────────────────────────────────────
print("Loading data...")
with open(out / "health_metrics.json") as f:
    health = json.load(f)
with open(out / "feature_statistics.json") as f:
    feat_stats = json.load(f)
with open(out / "feature_action_correlations.json") as f:
    corr_summary = json.load(f)

feat_action_corr = torch.load(out / "feature_action_correlation.pt", weights_only=True).float().numpy()
dec_weights      = torch.load(out / "decoder_weights.pt", weights_only=True).float().numpy()
fire_counts      = torch.load(out / "feature_fire_counts.pt", weights_only=True).float().numpy()
top_vals         = torch.load(out / "top_activation_values.pt", weights_only=True).float().numpy()

dict_size      = feat_action_corr.shape[0]
action_dim     = feat_action_corr.shape[1]
activation_dim = dec_weights.shape[0]
freq           = np.array(feat_stats["activation_frequency"])
cond_mean      = np.array(feat_stats["conditional_mean"])

# Determine which action dims are "real" (have any correlation > threshold)
action_max_abs = np.abs(feat_action_corr).max(axis=0)
real_action_dims = np.where(action_max_abs > 0.05)[0]
print(f"Real action dims (max|r| > 0.05): {real_action_dims.tolist()}")

# Load the big file
print(f"Loading top_activating_examples.json ({(out / 'top_activating_examples.json').stat().st_size/1e6:.0f} MB)...")
with open(out / "top_activating_examples.json") as f:
    top_examples = json.load(f)
print("Loaded.")


# ── Helper: extract arrays from top examples for a feature ───────────────
def get_feature_data(feat_idx):
    """Return arrays of (activation_values, proprio_states, actions) for top examples."""
    examples = top_examples.get(f"feature_{feat_idx}", [])
    acts, states, actions = [], [], []
    for ex in examples:
        if ex.get("activation", -1) <= 0:
            continue
        acts.append(ex["activation"])
        if "proprio_state" in ex:
            states.append(ex["proprio_state"])
        if "actions" in ex:
            a = np.array(ex["actions"])
            # Flatten if [horizon, action_dim]
            if a.ndim == 2:
                a = a.mean(axis=0)
            actions.append(a)
    return (
        np.array(acts) if acts else None,
        np.array(states) if states else None,
        np.array(actions) if actions else None,
    )


# ── 1. Feature-Proprio Correlation ───────────────────────────────────────
print("\n[1/5] Computing feature-proprio correlations from top examples...")

# For each feature, collect all proprio states and activations
# We'll build a correlation matrix [dict_size, state_dim]

# First pass: figure out state_dim and collect data
state_dim = None
feature_proprio_corr = None

sample_feats = [i for i in range(dict_size) if freq[i] > 0.05][:200]
all_feature_vals = []
all_states = []

for fi in sample_feats:
    acts, states, _ = get_feature_data(fi)
    if states is not None and len(states) >= 3:
        if state_dim is None:
            state_dim = states.shape[1] if states.ndim > 1 else len(states[0])
        all_feature_vals.append(acts)
        all_states.append(states)

if state_dim is not None:
    print(f"  State dim: {state_dim}, using {len(sample_feats)} features for proprio analysis")

    # For each feature with data: compute correlation with each state dim
    n_feats_with_data = len(all_feature_vals)
    feat_state_corr = np.zeros((n_feats_with_data, state_dim))

    for i, (act_vals, states) in enumerate(zip(all_feature_vals, all_states)):
        n = min(len(act_vals), len(states))
        if n < 3:
            continue
        a = act_vals[:n]
        s = states[:n]
        a_centered = a - a.mean()
        a_std = a.std()
        if a_std < 1e-8:
            continue
        for d in range(state_dim):
            sd = s[:, d] if s.ndim > 1 else s
            sd_centered = sd - sd.mean()
            sd_std = sd.std()
            if sd_std < 1e-8:
                continue
            feat_state_corr[i, d] = (a_centered * sd_centered).mean() / (a_std * sd_std)

    # Which state dims have the strongest feature correlations?
    state_max_abs = np.abs(feat_state_corr).max(axis=0)
    top_state_dims = np.argsort(state_max_abs)[::-1][:10]
    print(f"  Most predictive state dims: {top_state_dims.tolist()}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Heatmap
    im = axes[0].imshow(feat_state_corr[:50, :], cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=axes[0])
    axes[0].set_xlabel("Proprio state dimension")
    axes[0].set_ylabel("Feature (top 50 by freq)")
    axes[0].set_title("Feature–Proprio State Pearson Correlations")

    # Top state dims bar chart
    axes[1].bar(range(10), state_max_abs[top_state_dims], color="steelblue")
    axes[1].set_xticks(range(10))
    axes[1].set_xticklabels([f"s{d}" for d in top_state_dims], rotation=45)
    axes[1].set_ylabel("Max |r| across features")
    axes[1].set_title("Most Predictive Proprio State Dimensions")

    plt.tight_layout()
    plt.savefig(out / "viz_deep_proprio_corr.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved viz_deep_proprio_corr.png")
else:
    print("  No proprio state data found in top examples.")
    feat_state_corr = None
    top_state_dims = []


# ── 2. Joint Activation Profiles for Action-Correlated Features ──────────
print("\n[2/5] Joint activation profiles for action-correlated features...")

# Focus on the real action dims only
if len(real_action_dims) > 0:
    fig, big_axes = plt.subplots(
        len(real_action_dims), 3, figsize=(15, 4 * len(real_action_dims))
    )
    if len(real_action_dims) == 1:
        big_axes = big_axes[np.newaxis, :]

    for row, a_idx in enumerate(real_action_dims):
        # Top feature correlated with this action dim
        col = feat_action_corr[:, a_idx]
        top_feat = int(np.argmax(np.abs(col)))
        r_val = col[top_feat]

        acts, states, actions = get_feature_data(top_feat)

        # Panel 1: activation value distribution
        if acts is not None:
            big_axes[row, 0].hist(acts, bins=20, color="steelblue", edgecolor="white")
            big_axes[row, 0].set_xlabel("SAE activation value")
            big_axes[row, 0].set_ylabel("Count")
            big_axes[row, 0].set_title(f"Action dim {a_idx}: feature {top_feat} (r={r_val:.3f})\nActivation Distribution")

        # Panel 2: action dim value at top examples
        if actions is not None and len(actions) > 0:
            action_vals = actions[:, a_idx] if actions.ndim > 1 and actions.shape[1] > a_idx else None
            if action_vals is not None and acts is not None:
                n = min(len(acts), len(action_vals))
                sc = big_axes[row, 1].scatter(acts[:n], action_vals[:n], alpha=0.7, s=30, c="coral")
                big_axes[row, 1].set_xlabel("SAE activation")
                big_axes[row, 1].set_ylabel(f"Action dim {a_idx} value")
                big_axes[row, 1].set_title(f"Feature activation vs action dim {a_idx}")
                # Add trend line
                if n > 2:
                    z = np.polyfit(acts[:n], action_vals[:n], 1)
                    p = np.poly1d(z)
                    xr = np.linspace(acts[:n].min(), acts[:n].max(), 50)
                    big_axes[row, 1].plot(xr, p(xr), "k--", linewidth=1.5, alpha=0.7)

        # Panel 3: proprio state dim most correlated with this feature
        if states is not None and state_dim is not None and len(states) > 0:
            if feat_state_corr is not None and top_feat in sample_feats:
                feat_row = sample_feats.index(top_feat) if top_feat in sample_feats else None
                if feat_row is not None:
                    best_state_dim = int(np.argmax(np.abs(feat_state_corr[feat_row])))
                    state_vals = states[:, best_state_dim] if states.ndim > 1 else states
                    if acts is not None:
                        n = min(len(acts), len(state_vals))
                        big_axes[row, 2].scatter(state_vals[:n], acts[:n], alpha=0.7, s=30, c="steelblue")
                        big_axes[row, 2].set_xlabel(f"Proprio dim {best_state_dim}")
                        big_axes[row, 2].set_ylabel("SAE activation")
                        big_axes[row, 2].set_title(f"Proprio dim {best_state_dim} vs feature activation")
                else:
                    big_axes[row, 2].axis("off")
            else:
                big_axes[row, 2].axis("off")
        else:
            big_axes[row, 2].axis("off")

    plt.suptitle("Top Feature per Real Action Dimension", fontsize=13)
    plt.tight_layout()
    plt.savefig(out / "viz_deep_action_profiles.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved viz_deep_action_profiles.png")


# ── 3. Feature Clustering ─────────────────────────────────────────────────
print("\n[3/5] Feature clustering via decoder weights...")

# Cluster features using their decoder weight vectors (feature directions in activation space)
# Use simple cosine-similarity based agglomerative grouping
W = dec_weights.T  # [dict_size, activation_dim]
W_normed = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-8)

# Subsample to most active features for clarity
active_mask = freq > 0.02
active_idx = np.where(active_mask)[0]
n_active = len(active_idx)
print(f"  Active features (freq > 0.02): {n_active}")

W_active = W_normed[active_idx]
cos_sim_active = W_active @ W_active.T  # [n_active, n_active]

# Sort by action correlation to see structure
max_abs_action = np.abs(feat_action_corr[active_idx]).max(axis=1)
sort_order = np.argsort(max_abs_action)[::-1]
cos_sorted = cos_sim_active[np.ix_(sort_order, sort_order)]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Sorted cosine similarity heatmap
n_show = min(200, n_active)
im = axes[0].imshow(cos_sorted[:n_show, :n_show], cmap="RdBu_r", vmin=-0.3, vmax=0.3, aspect="auto")
plt.colorbar(im, ax=axes[0])
axes[0].set_title(f"Active Feature Similarity (sorted by action corr)\ntop {n_show} of {n_active} active features")
axes[0].set_xlabel("Feature rank (by action correlation)")
axes[0].set_ylabel("Feature rank (by action correlation)")

# Scatter: action correlation vs activation frequency, colored by "best action dim"
best_action_dim = np.argmax(np.abs(feat_action_corr), axis=1)
scatter_colors = plt.cm.tab10(best_action_dim[active_idx] % 10)
axes[1].scatter(freq[active_idx], max_abs_action, c=scatter_colors, alpha=0.4, s=15)
axes[1].set_xlabel("Activation Frequency")
axes[1].set_ylabel("Max |r| with any action dim")
axes[1].set_title("Feature Space: Frequency vs Action Correlation\n(color = best-correlated action dim)")
axes[1].axhline(0.1, color="red", linestyle="--", alpha=0.7, label="|r|=0.1")
axes[1].legend()

# Add colorbar legend for action dims
for a_idx in real_action_dims[:7]:
    axes[1].scatter([], [], c=[plt.cm.tab10(a_idx % 10)], label=f"action {a_idx}", s=20)
axes[1].legend(fontsize=7, loc="upper right")

plt.tight_layout()
plt.savefig(out / "viz_deep_clustering.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved viz_deep_clustering.png")


# ── 4. Feature Cards for Most Interesting Features ─────────────────────────
print("\n[4/5] Feature cards for top action-correlated features...")

# Pick top 6 features with highest action correlation
max_abs_per_feat = np.abs(feat_action_corr).max(axis=1)
top6_feat_idx = np.argsort(max_abs_per_feat)[::-1][:6]

fig = plt.figure(figsize=(18, 12))
outer = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4)

for card_idx, fi in enumerate(top6_feat_idx):
    inner = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[card_idx], hspace=0.5, wspace=0.4)
    acts, states, actions = get_feature_data(fi)
    best_action = int(np.argmax(np.abs(feat_action_corr[fi])))
    r_val = feat_action_corr[fi, best_action]

    title = f"Feature {fi}\nfreq={freq[fi]:.3f}  max|r|={max_abs_per_feat[fi]:.3f}  best_action={best_action}  r={r_val:.3f}"

    # Top-left: activation value histogram
    ax00 = fig.add_subplot(inner[0, 0])
    if acts is not None and len(acts) > 0:
        ax00.hist(acts, bins=15, color="steelblue", edgecolor="none")
    ax00.set_title("Activation dist.", fontsize=7)
    ax00.tick_params(labelsize=6)

    # Top-right: action dim scatter
    ax01 = fig.add_subplot(inner[0, 1])
    if actions is not None and acts is not None and len(actions) > 0:
        if actions.ndim > 1 and actions.shape[1] > best_action:
            action_vals = actions[:, best_action]
            n = min(len(acts), len(action_vals))
            ax01.scatter(acts[:n], action_vals[:n], s=15, alpha=0.7, color="coral")
            ax01.set_xlabel("activation", fontsize=6)
            ax01.set_ylabel(f"action[{best_action}]", fontsize=6)
    ax01.set_title(f"vs action {best_action}", fontsize=7)
    ax01.tick_params(labelsize=6)

    # Bottom-left: action correlation bar chart (real dims only)
    ax10 = fig.add_subplot(inner[1, 0])
    if len(real_action_dims) > 0:
        corrs = feat_action_corr[fi, real_action_dims]
        colors = ["steelblue" if c > 0 else "coral" for c in corrs]
        ax10.bar(range(len(real_action_dims)), corrs, color=colors)
        ax10.set_xticks(range(len(real_action_dims)))
        ax10.set_xticklabels([str(d) for d in real_action_dims], fontsize=6)
        ax10.axhline(0, color="black", linewidth=0.5)
        ax10.set_title("Action correlations", fontsize=7)
    ax10.tick_params(labelsize=6)

    # Bottom-right: proprio state scatter (if available)
    ax11 = fig.add_subplot(inner[1, 1])
    if states is not None and len(states) > 0 and feat_state_corr is not None:
        if fi in sample_feats:
            feat_row = sample_feats.index(fi)
            best_state = int(np.argmax(np.abs(feat_state_corr[feat_row])))
            state_r = feat_state_corr[feat_row, best_state]
            state_vals = states[:, best_state] if states.ndim > 1 else states
            if acts is not None:
                n = min(len(acts), len(state_vals))
                ax11.scatter(state_vals[:n], acts[:n], s=15, alpha=0.7, color="green")
                ax11.set_xlabel(f"state[{best_state}]", fontsize=6)
                ax11.set_ylabel("activation", fontsize=6)
                ax11.set_title(f"vs state {best_state} (r={state_r:.2f})", fontsize=7)
    ax11.tick_params(labelsize=6)

    fig.text(
        outer[card_idx].get_position(fig).x0 + 0.01,
        outer[card_idx].get_position(fig).y1,
        title, fontsize=7, va="bottom"
    )

plt.suptitle("Feature Cards: Top 6 Action-Correlated Features", fontsize=13, y=1.01)
plt.savefig(out / "viz_deep_feature_cards.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved viz_deep_feature_cards.png")


# ── 5. Text Report ────────────────────────────────────────────────────────
print("\n[5/5] Writing analysis report...")

lines = []
lines.append("=" * 70)
lines.append("SAE DEEP ANALYSIS REPORT")
lines.append(f"Experiment: {out.name}")
lines.append("=" * 70)

lines.append("\n--- HEALTH ---")
lines.append(f"  Variance explained: {health['frac_variance_explained']:.4f}")
lines.append(f"  Cosine similarity:  {health['cossim']:.4f}")
lines.append(f"  L0 (avg active):    {health['l0']:.2f}  (target k=32)")
lines.append(f"  Fraction alive:     {health['frac_alive']:.4f}  ({health['n_alive']} alive, {health['n_dead']} dead)")
lines.append(f"  L2 ratio:           {health['l2_ratio']:.4f}")

lines.append("\n--- ACTION SPACE ---")
lines.append(f"  Total action dims: {action_dim}")
lines.append(f"  Real action dims (max|r| > 0.05): {real_action_dims.tolist()}")
lines.append(f"  Features with max|r| > 0.1: {(max_abs_per_feat > 0.1).sum()}")
lines.append(f"  Features with max|r| > 0.2: {(max_abs_per_feat > 0.2).sum()}")

lines.append("\n--- TOP ACTION-CORRELATED FEATURES ---")
lines.append(f"  {'Rank':>4} {'FeatID':>6} {'Max|r|':>7} {'BestAct':>8} {'r':>7} {'Freq':>7} {'CondMean':>9}")
lines.append("  " + "-" * 55)
for rank, fi in enumerate(top6_feat_idx):
    best_a = int(np.argmax(np.abs(feat_action_corr[fi])))
    r = feat_action_corr[fi, best_a]
    lines.append(f"  {rank+1:4d} {fi:6d} {max_abs_per_feat[fi]:7.4f} {best_a:8d} {r:7.4f} {freq[fi]:7.4f} {cond_mean[fi]:9.4f}")

lines.append("\n--- PER ACTION DIM: TOP CORRELATED FEATURES ---")
for a_idx in real_action_dims:
    col = feat_action_corr[:, a_idx]
    top_pos_idx = int(np.argmax(col))
    top_neg_idx = int(np.argmin(col))
    lines.append(f"\n  Action dim {a_idx}:")
    lines.append(f"    Most positively correlated: feature {top_pos_idx}  r={col[top_pos_idx]:.4f}  freq={freq[top_pos_idx]:.4f}")
    lines.append(f"    Most negatively correlated: feature {top_neg_idx}  r={col[top_neg_idx]:.4f}  freq={freq[top_neg_idx]:.4f}")
    # Show example proprio states for top feature
    acts, states, actions = get_feature_data(top_pos_idx)
    if states is not None and len(states) > 0:
        mean_state = states.mean(axis=0) if states.ndim > 1 else states
        lines.append(f"    Top feature mean proprio (first 7): {mean_state[:7].round(3).tolist()}")

report_path = out / "deep_analysis_report.txt"
with open(report_path, "w") as f:
    f.write("\n".join(lines))
print(f"  Saved deep_analysis_report.txt")

print(f"\nDone. Output files:")
for f in sorted(out.glob("viz_deep_*.png")):
    print(f"  {f.name}")
print(f"  deep_analysis_report.txt")

print("\nTo copy back locally:")
print(f"  cd {out}")
print(f"  tar czf deep_analysis.tar.gz viz_deep_*.png deep_analysis_report.txt")
print(f"  scp rvaldes6@login-ice.pace.gatech.edu:{out}/deep_analysis.tar.gz .")
