"""
SAE Interpretation Visualization — run headlessly on PACE, saves PNGs to output dir.

Usage:
    python visualize_sae.py
    python visualize_sae.py --output_dir /path/to/sae_interpretations/exp_name

Then scp the PNGs back locally:
    scp 'rvaldes6@login-ice.pace.gatech.edu:/storage/project/r-agarg35-0/rvaldes6/sae_interpretations/pi05_libero_expert_layer_0_k32_exp4/viz_*.png' .
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt
import numpy as np
import torch

# ── CONFIG ────────────────────────────────────────────────────────────────
DEFAULT_OUTPUT_DIR = "/storage/project/r-agarg35-0/rvaldes6/sae_interpretations/pi05_libero_expert_layer_0_k32_exp4"
# ──────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
parser.add_argument("--load_top_examples", action="store_true", help="Load the 528MB top_activating_examples.json")
args = parser.parse_args()

out = Path(args.output_dir)
assert out.exists(), f"Output dir not found: {out}"
print("Files found:")
for f in sorted(out.iterdir()):
    print(f"  {f.name:45s} {f.stat().st_size / 1e6:.1f} MB")

# ── Load data ─────────────────────────────────────────────────────────────
with open(out / "health_metrics.json") as f:
    health = json.load(f)
with open(out / "feature_statistics.json") as f:
    feat_stats = json.load(f)
with open(out / "decoder_analysis.json") as f:
    dec_analysis = json.load(f)
with open(out / "feature_action_correlations.json") as f:
    corr_summary = json.load(f)

fire_counts      = torch.load(out / "feature_fire_counts.pt", weights_only=True).float()
top_vals         = torch.load(out / "top_activation_values.pt", weights_only=True).float()
top_indices      = torch.load(out / "top_activation_indices.pt", weights_only=True)
feat_action_corr = torch.load(out / "feature_action_correlation.pt", weights_only=True).float()
dec_cos_sim      = torch.load(out / "decoder_cosine_sim.pt", weights_only=True).float()
dec_weights      = torch.load(out / "decoder_weights.pt", weights_only=True).float()

dict_size      = feat_action_corr.shape[0]
action_dim     = feat_action_corr.shape[1]
activation_dim = dec_weights.shape[0]
freq           = np.array(feat_stats["activation_frequency"])
fire           = fire_counts.numpy()
corr_np        = feat_action_corr.numpy()
dec_norms      = np.array(dec_analysis["decoder_norms"])
sort_order     = np.argsort(freq)[::-1]

print(f"\nddict_size={dict_size}, activation_dim={activation_dim}, action_dim={action_dim}")

# ── 1. Health Metrics ─────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  SAE HEALTH METRICS")
print("=" * 55)
for name, val in [
    ("L2 reconstruction loss",    f"{health['l2_loss']:.4f}"),
    ("L0 (avg active features)",  f"{health['l0']:.2f}  (target k=32)"),
    ("Variance explained",         f"{health['frac_variance_explained']:.4f}"),
    ("Cosine similarity",          f"{health['cossim']:.4f}"),
    ("L2 ratio",                   f"{health['l2_ratio']:.4f}"),
    ("Fraction alive features",    f"{health['frac_alive']:.4f}  ({health['n_alive']} alive / {health['n_dead']} dead)"),
    ("Total samples evaluated",    f"{health['total_samples']:,}"),
]:
    print(f"  {name:<35s} {val}")
print("=" * 55)

# ── 2. Feature Utilization ────────────────────────────────────────────────
print("\n[1/5] Feature utilization...")
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].hist(freq, bins=50, color="steelblue", edgecolor="white", linewidth=0.5)
axes[0].axvline(freq.mean(), color="red", linestyle="--", label=f"mean={freq.mean():.3f}")
axes[0].set_xlabel("Activation Frequency")
axes[0].set_ylabel("# Features")
axes[0].set_title("Feature Activation Frequency Distribution")
axes[0].legend()

sorted_freq = np.sort(freq)[::-1]
axes[1].plot(sorted_freq, color="steelblue", linewidth=1.5)
axes[1].axhline(0.01, color="red", linestyle="--", alpha=0.7, label="1% threshold")
axes[1].set_xlabel("Feature rank (sorted by freq)")
axes[1].set_ylabel("Activation Frequency")
axes[1].set_title("Sorted Feature Frequencies")
axes[1].legend()

nonzero_fire = fire[fire > 0]
axes[2].hist(np.log10(nonzero_fire + 1), bins=50, color="coral", edgecolor="white", linewidth=0.5)
axes[2].set_xlabel("log10(fire count + 1)")
axes[2].set_ylabel("# Features")
axes[2].set_title(f"Fire Count Distribution (dead={health['n_dead']})")

plt.tight_layout()
plt.savefig(out / "viz_feature_utilization.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"  freq > 0.5:  {(freq > 0.5).sum()}")
print(f"  freq > 0.1:  {(freq > 0.1).sum()}")
print(f"  freq < 0.01: {(freq < 0.01).sum()}")
print(f"  dead (0 fires): {(fire == 0).sum()}")

# ── 3. Decoder Structure ──────────────────────────────────────────────────
print("\n[2/5] Decoder structure...")
print(f"  Mean pairwise cosine sim: {dec_analysis['mean_pairwise_cosine']:.4f}")
print(f"  Max pairwise cosine sim:  {dec_analysis['max_pairwise_cosine']:.4f}")
print(f"  Mean enc-dec alignment:   {dec_analysis['mean_enc_dec_alignment']:.4f}")
print(f"  Decoder norm range:       [{dec_norms.min():.4f}, {dec_norms.max():.4f}]")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

cos_np = dec_cos_sim.numpy()
step = max(1, dict_size // 128)
cos_sub = cos_np[::step, ::step]
im = axes[0].imshow(cos_sub, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, ax=axes[0])
axes[0].set_title(f"Decoder Cosine Similarity\n(sampled {cos_sub.shape[0]}x{cos_sub.shape[1]})")
axes[0].set_xlabel("Feature index")
axes[0].set_ylabel("Feature index")

upper_tri = cos_np[np.triu_indices(dict_size, k=1)]
axes[1].hist(upper_tri, bins=100, color="steelblue", edgecolor="none")
axes[1].axvline(upper_tri.mean(), color="red", linestyle="--", label=f"mean={upper_tri.mean():.3f}")
axes[1].set_xlabel("Pairwise cosine similarity")
axes[1].set_ylabel("Count")
axes[1].set_title("Off-diagonal Cosine Similarities")
axes[1].legend()

axes[2].hist(dec_norms, bins=50, color="coral", edgecolor="white", linewidth=0.5)
axes[2].axvline(dec_norms.mean(), color="red", linestyle="--", label=f"mean={dec_norms.mean():.3f}")
axes[2].set_xlabel("Decoder column norm")
axes[2].set_ylabel("# Features")
axes[2].set_title("Decoder Column Norms")
axes[2].legend()

plt.tight_layout()
plt.savefig(out / "viz_decoder_structure.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 4. Feature-Action Correlations ────────────────────────────────────────
print("\n[3/5] Feature-action correlations...")
max_abs = np.abs(corr_np).max(axis=1)
sorted_feat_idx = np.argsort(max_abs)[::-1]

print(f"  Mean |correlation|: {np.abs(corr_np).mean():.4f}")
print(f"  max|r| > 0.1: {(max_abs > 0.1).sum()}")
print(f"  max|r| > 0.2: {(max_abs > 0.2).sum()}")
print(f"  max|r| > 0.3: {(max_abs > 0.3).sum()}")

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

n_show = min(100, dict_size)
corr_show = corr_np[sorted_feat_idx[:n_show], :]
im = axes[0].imshow(corr_show, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, ax=axes[0])
axes[0].set_xlabel("Action dimension")
axes[0].set_ylabel(f"Top {n_show} features (sorted by max |r|)")
axes[0].set_title("Feature-Action Pearson Correlations")
axes[0].set_xticks(range(action_dim))

axes[1].hist(max_abs, bins=50, color="steelblue", edgecolor="white", linewidth=0.5)
axes[1].axvline(0.1, color="orange", linestyle="--", label="|r|=0.1")
axes[1].axvline(0.2, color="red", linestyle="--", label="|r|=0.2")
axes[1].set_xlabel("Max |correlation| with any action dim")
axes[1].set_ylabel("# Features")
axes[1].set_title("Distribution of Feature-Action Correlation Strength")
axes[1].legend()

plt.tight_layout()
plt.savefig(out / "viz_feature_action_corr.png", dpi=150, bbox_inches="tight")
plt.close()

# Per-action-dim bar chart
n_cols = min(action_dim, 7)
n_rows = (action_dim + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
axes = np.array(axes).flatten()

for a_idx in range(action_dim):
    col = corr_np[:, a_idx]
    top5_pos = np.argsort(col)[-5:][::-1]
    top5_neg = np.argsort(col)[:5]
    feat_ids = list(top5_pos) + list(top5_neg)
    vals = col[feat_ids]
    colors = ["steelblue" if v > 0 else "coral" for v in vals]
    axes[a_idx].barh([f"f{fi}" for fi in feat_ids], vals, color=colors)
    axes[a_idx].axvline(0, color="black", linewidth=0.8)
    axes[a_idx].set_title(f"Action dim {a_idx}")
    axes[a_idx].set_xlabel("Pearson r")

for i in range(action_dim, len(axes)):
    axes[i].set_visible(False)

plt.suptitle("Top ±5 Features per Action Dimension", fontsize=13)
plt.tight_layout()
plt.savefig(out / "viz_per_action_dim.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 5. Feature Overview Table ─────────────────────────────────────────────
print("\n[4/5] Feature overview table...")
cond_mean    = np.array(feat_stats["conditional_mean"])
feat_max     = np.array(feat_stats["max"])
max_abs_corr = np.abs(corr_np).max(axis=1)
best_action  = np.argmax(np.abs(corr_np), axis=1)

print(f"\n{'Rank':>4} {'FeatID':>6} {'Freq':>7} {'CondMean':>9} {'MaxAct':>8} {'Max|r|':>7} {'BestAct':>8}")
print("-" * 60)
for rank, fi in enumerate(sort_order[:50]):
    print(f"{rank+1:4d} {fi:6d} {freq[fi]:7.4f} {cond_mean[fi]:9.4f} "
          f"{feat_max[fi]:8.4f} {max_abs_corr[fi]:7.4f} {best_action[fi]:8d}")

# ── 6. Top-Activating Examples (optional) ────────────────────────────────
print("\n[5/5] Top-activating examples...")
top_ex_path = out / "top_activating_examples.json"
if args.load_top_examples:
    print(f"  Loading {top_ex_path.stat().st_size/1e6:.0f} MB file...")
    with open(top_ex_path) as f:
        top_examples = json.load(f)

    top4_features = sort_order[:4]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax_idx, fi in enumerate(top4_features):
        examples = top_examples.get(f"feature_{fi}", [])
        states = [ex["proprio_state"] for ex in examples if "proprio_state" in ex]
        acts   = [ex["activation"] for ex in examples if "proprio_state" in ex]

        if not states:
            axes[ax_idx].text(0.5, 0.5, "No proprio data", ha="center", va="center")
            axes[ax_idx].set_title(f"Feature {fi}")
            continue

        states_arr = np.array(states)
        acts_arr   = np.array(acts)
        sc = axes[ax_idx].scatter(
            states_arr[:, 0], states_arr[:, 1],
            c=acts_arr, cmap="viridis", s=60, edgecolors="none", alpha=0.8
        )
        plt.colorbar(sc, ax=axes[ax_idx], label="SAE activation")
        axes[ax_idx].set_xlabel("proprio dim 0")
        axes[ax_idx].set_ylabel("proprio dim 1")
        axes[ax_idx].set_title(f"Feature {fi}  (freq={freq[fi]:.3f}, max|r|={max_abs_corr[fi]:.3f})")

    plt.suptitle("Top-Activating Examples: Proprio State (dims 0 & 1)", fontsize=13)
    plt.tight_layout()
    plt.savefig(out / "viz_top_examples_proprio.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved viz_top_examples_proprio.png")
else:
    print(f"  Skipped (pass --load_top_examples to include, file is {top_ex_path.stat().st_size/1e6:.0f} MB)")

print(f"\nDone. PNGs saved to: {out}")
print("To copy locally:")
print(f"  scp 'rvaldes6@login-ice.pace.gatech.edu:{out}/viz_*.png' .")
