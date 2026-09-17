"""
Feature interpretation script for trained SAEs on PI0 model activations.

This script loads a trained SAE checkpoint and analyzes what each learned feature
represents by:
1. Computing health metrics (frac_alive, variance explained, L0, etc.)
2. Finding top-activating data points for each feature
3. Correlating feature activations with observable properties (task, proprioception)
4. Analyzing decoder weight structure (cosine similarities, norms)

Usage:
    python -m openpi.interpretability.interpret_sae \
        --config pi05_libero \
        --hook_point expert_layer_0 \
        --sae_checkpoint /path/to/sae_checkpoints/exp_name/trainer_0/ae.pt \
        --output_dir ./sae_interpretations/exp_name
"""

import argparse
import json
import logging
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import safetensors.torch

# Stub out nnsight before any dictionary_learning imports, since the package __init__.py
# imports buffer.py which requires nnsight -> transformers -> regex.
# We don't need nnsight for interpretation — only BatchTopKSAE (pure PyTorch).
import sys
import types
if "nnsight" not in sys.modules:
    _nnsight_stub = types.ModuleType("nnsight")
    _nnsight_stub.LanguageModel = None
    sys.modules["nnsight"] = _nnsight_stub

from openpi.interpretability.dictionary_learning.dictionary_learning.trainers.batch_top_k import (
    BatchTopKSAE,
)

import openpi.models.pi0_config as _pi0_config
import openpi.training.config as _config
import openpi.training.data_loader as _data
from openpi.interpretability.activation_hooks import (
    PI0ActivationBuffer,
    PI0ActivationCollector,
    PI0_HOOK_POINTS,
    to_device,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Interpret trained SAE features")

    # Model and data config
    parser.add_argument("--config", type=str, required=True, help="Training config name (e.g., pi05_libero)")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to PI0 model checkpoint")
    parser.add_argument("--hook_point", type=str, required=True, help="Hook point the SAE was trained on")

    # SAE checkpoint
    parser.add_argument("--sae_checkpoint", type=str, required=True, help="Path to trained SAE checkpoint (ae.pt)")

    # Analysis parameters
    parser.add_argument("--n_batches", type=int, default=200, help="Number of data batches to collect activations from")
    parser.add_argument("--data_batch_size", type=int, default=16, help="Batch size for model forward passes")
    parser.add_argument("--top_k", type=int, default=20, help="Number of top-activating examples per feature")
    parser.add_argument("--eval_batches", type=int, default=50, help="Number of batches for SAE eval metrics")

    # Output
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save interpretation results")

    # Hardware
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"])

    return parser.parse_args()


def get_dtype(dtype_str: str) -> torch.dtype:
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_str]


def load_pi0_model(config, checkpoint_path, device):
    from openpi.models_pytorch import pi0_pytorch

    logger.info(f"Loading PI0 model with config: {config.model}")
    model = pi0_pytorch.PI0Pytorch(config=config.model)
    if checkpoint_path:
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        safetensors.torch.load_model(model, checkpoint_path)
    else:
        logger.warning("No checkpoint provided - using randomly initialized model!")
    model = model.to(device)
    model.eval()
    model.sample_actions = model.sample_actions.__wrapped__
    return model


def load_sae(checkpoint_path, device):
    """Load a trained BatchTopKSAE from checkpoint."""
    logger.info(f"Loading SAE from: {checkpoint_path}")
    sae = BatchTopKSAE.from_pretrained(checkpoint_path, device=device)
    sae.eval()
    logger.info(f"  activation_dim={sae.activation_dim}, dict_size={sae.dict_size}, k={sae.k.item()}")
    logger.info(f"  threshold={sae.threshold.item():.6f}")
    return sae


# =====================================================================
# 1. Health Metrics
# =====================================================================

@torch.no_grad()
def compute_health_metrics(sae, activation_buffer, n_batches=50, device="cuda"):
    """Compute SAE reconstruction quality and feature utilization metrics."""
    logger.info(f"Computing health metrics over {n_batches} batches...")

    total_l2 = 0.0
    total_l1 = 0.0
    total_l0 = 0.0
    total_var_explained = 0.0
    total_cossim = 0.0
    total_l2_ratio = 0.0
    feature_fire_counts = torch.zeros(sae.dict_size, device=device)
    total_samples = 0
    count = 0

    for x in activation_buffer:
        x = x.to(device)
        x_hat, f = sae(x, output_features=True)

        batch_size = x.shape[0]
        total_samples += batch_size

        # Reconstruction metrics
        total_l2 += F.mse_loss(x_hat, x, reduction="mean").item()
        total_l1 += f.norm(p=1, dim=-1).mean().item()
        total_l0 += (f != 0).float().sum(dim=-1).mean().item()

        # Cosine similarity
        x_normed = F.normalize(x, dim=-1)
        x_hat_normed = F.normalize(x_hat, dim=-1)
        total_cossim += (x_normed * x_hat_normed).sum(dim=-1).mean().item()

        # L2 ratio
        total_l2_ratio += (x_hat.norm(dim=-1) / x.norm(dim=-1).clamp(min=1e-8)).mean().item()

        # Variance explained
        total_var = torch.var(x, dim=0).sum()
        residual_var = torch.var(x - x_hat, dim=0).sum()
        total_var_explained += (1 - residual_var / total_var.clamp(min=1e-8)).item()

        # Feature utilization
        feature_fire_counts += (f != 0).float().sum(dim=0)

        count += 1
        if count >= n_batches:
            break

    metrics = {
        "l2_loss": total_l2 / count,
        "l1_loss": total_l1 / count,
        "l0": total_l0 / count,
        "frac_variance_explained": total_var_explained / count,
        "cossim": total_cossim / count,
        "l2_ratio": total_l2_ratio / count,
        "frac_alive": (feature_fire_counts > 0).float().mean().item(),
        "n_alive": int((feature_fire_counts > 0).sum().item()),
        "n_dead": int((feature_fire_counts == 0).sum().item()),
        "total_samples": total_samples,
        "feature_fire_counts": feature_fire_counts.cpu().tolist(),
    }

    logger.info(f"  L2 loss:         {metrics['l2_loss']:.4f}")
    logger.info(f"  L0 (avg active): {metrics['l0']:.1f}")
    logger.info(f"  Var explained:   {metrics['frac_variance_explained']:.4f}")
    logger.info(f"  Cosine sim:      {metrics['cossim']:.4f}")
    logger.info(f"  L2 ratio:        {metrics['l2_ratio']:.4f}")
    logger.info(f"  Frac alive:      {metrics['frac_alive']:.4f} ({metrics['n_alive']}/{metrics['n_alive']+metrics['n_dead']})")

    return metrics


# =====================================================================
# 2. Top-Activating Examples Per Feature
# =====================================================================

@torch.no_grad()
def find_top_activating_examples(
    sae,
    model,
    data_loader,
    hook_point,
    n_batches=200,
    top_k=20,
    device="cuda",
    dtype=torch.bfloat16,
):
    """
    Collect activations and track which data points maximally activate each feature.

    Returns per-feature info:
        - top_k activation values
        - corresponding batch/sample indices
        - corresponding proprioceptive states and task descriptions
    """
    logger.info(f"Finding top-{top_k} activating examples per feature over {n_batches} batches...")

    collector = PI0ActivationCollector(model)
    collector.register_hooks([hook_point])

    dict_size = sae.dict_size

    # Track top-k per feature using a running min-heap approach
    # Store (activation_value, global_sample_idx) per feature
    top_vals = torch.full((dict_size, top_k), -float("inf"), device=device)
    top_indices = torch.full((dict_size, top_k), -1, dtype=torch.long, device=device)

    # Store metadata for each global sample
    sample_metadata = {}
    global_idx = 0

    param = model.action_in_proj.weight
    model_device, model_dtype = param.device, param.dtype

    data_iter = iter(data_loader)
    for batch_idx in range(n_batches):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            batch = next(data_iter)

        observation, actions = batch if isinstance(batch, tuple) else (
            batch["observation"], batch["actions"]
        )

        # Extract metadata before moving to device
        batch_metadata = _extract_batch_metadata(observation, actions)

        observation = to_device(observation, model_device, model_dtype)
        actions = to_device(actions, model_device, model_dtype)

        with torch.no_grad():
            _ = model(observation, actions)

        # Get unflattened activations first to determine seq_len for correct sample mapping
        raw_acts = collector.get_activations(flatten=False)[hook_point]  # [B, seq_len, D]
        seq_len = raw_acts.shape[1] if raw_acts.ndim >= 3 else 1
        acts = raw_acts.reshape(-1, raw_acts.shape[-1])  # [B*seq_len, D]
        acts = acts.to(device, dtype=torch.float32)
        collector.clear_activations()

        n_image_tokens = batch_metadata.get("n_image_tokens", 0)
        text_len_val = batch_metadata.get("text_len", 0)
        is_lang_hook = hook_point.startswith("lang_layer_")

        # Encode through SAE
        features = sae.encode(acts.to(device))  # [n_tokens, dict_size]

        n_tokens = features.shape[0]

        for token_idx in range(n_tokens):
            feat_vals = features[token_idx]  # [dict_size]

            # For each feature, check if this activation is in the top-k
            min_vals, min_positions = top_vals.min(dim=1)  # [dict_size]
            replace_mask = feat_vals > min_vals  # [dict_size]

            if replace_mask.any():
                replace_features = replace_mask.nonzero(as_tuple=True)[0]
                for fi in replace_features:
                    fi = fi.item()
                    pos = min_positions[fi].item()
                    top_vals[fi, pos] = feat_vals[fi]
                    top_indices[fi, pos] = global_idx

            # Correct sample mapping: token_idx // seq_len gives the sample index within the batch
            sample_idx = token_idx // seq_len
            token_pos = token_idx % seq_len

            if is_lang_hook:
                if token_pos < n_image_tokens:
                    token_type = "image"
                elif token_pos < n_image_tokens + text_len_val:
                    token_type = "text"
                else:
                    token_type = "padding"
            else:
                token_type = "action_state"

            sample_metadata[global_idx] = {
                "batch_idx": batch_idx,
                "token_idx": token_idx,
                "token_pos": token_pos,
                "token_type": token_type,
                **batch_metadata.get(sample_idx, {}),
            }
            global_idx += 1

        if (batch_idx + 1) % 50 == 0:
            logger.info(f"  Processed {batch_idx + 1}/{n_batches} batches ({global_idx} total tokens)")

    collector.clear_hooks()

    # Sort top-k by activation value (descending) for each feature
    sorted_order = top_vals.argsort(dim=1, descending=True)
    top_vals = top_vals.gather(1, sorted_order)
    top_indices = top_indices.gather(1, sorted_order)

    logger.info(f"  Total tokens processed: {global_idx}")

    return {
        "top_vals": top_vals.cpu(),
        "top_indices": top_indices.cpu(),
        "sample_metadata": sample_metadata,
    }


def _extract_batch_metadata(observation, actions):
    """Extract interpretable metadata from a batch (task descriptions, proprioceptive state, etc.)."""
    metadata = {}
    batch_size = 1

    # Try to extract task/prompt info
    if hasattr(observation, "tokenized_prompt") and observation.tokenized_prompt is not None:
        tp = observation.tokenized_prompt
        batch_size = tp.shape[0] if hasattr(tp, "shape") else 1
        # Store per-batch text length for token type classification
        metadata["text_len"] = tp.shape[1] if hasattr(tp, "shape") and tp.ndim >= 2 else 0

    # Number of image tokens: SigLIP produces 256 tokens per image per camera
    if hasattr(observation, "images") and observation.images is not None:
        n_cameras = len(observation.images) if hasattr(observation.images, "__len__") else 1
        metadata["n_image_tokens"] = 256 * n_cameras

    # Extract proprioceptive state
    if hasattr(observation, "state") and observation.state is not None:
        state = observation.state
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu().numpy()
        elif isinstance(state, np.ndarray):
            pass
        else:
            state = None

        if state is not None:
            for i in range(min(state.shape[0], batch_size) if state.ndim > 1 else 1):
                if i not in metadata:
                    metadata[i] = {}
                metadata[i]["proprio_state"] = state[i].tolist() if state.ndim > 1 else state.tolist()

    # NOTE: actions are intentionally NOT stored per-token to avoid RAM exhaustion.
    # With n_batches=1000 x batch_size=16 x seq_len=800 = 12.8M tokens, storing
    # actions (50x7 floats each) per token would consume ~18GB of CPU RAM.

    metadata["batch_size"] = batch_size
    return metadata


# =====================================================================
# 3. Feature Activation Statistics
# =====================================================================

@torch.no_grad()
def compute_feature_statistics(sae, activation_buffer, n_batches=100, device="cuda"):
    """Compute per-feature activation statistics."""
    logger.info(f"Computing per-feature activation statistics over {n_batches} batches...")

    dict_size = sae.dict_size

    # Running statistics
    feat_sum = torch.zeros(dict_size, device=device)
    feat_sq_sum = torch.zeros(dict_size, device=device)
    feat_max = torch.full((dict_size,), -float("inf"), device=device)
    feat_count = torch.zeros(dict_size, device=device)  # nonzero activations count
    total_count = 0
    count = 0

    for x in activation_buffer:
        x = x.to(device)
        features = sae.encode(x)  # [batch, dict_size]

        nonzero_mask = features != 0
        feat_sum += features.sum(dim=0)
        feat_sq_sum += (features ** 2).sum(dim=0)
        feat_max = torch.max(feat_max, features.max(dim=0).values)
        feat_count += nonzero_mask.float().sum(dim=0)
        total_count += features.shape[0]

        count += 1
        if count >= n_batches:
            break

    # Compute stats
    feat_mean = feat_sum / total_count
    feat_var = (feat_sq_sum / total_count) - feat_mean ** 2
    feat_std = feat_var.clamp(min=0).sqrt()

    # Conditional mean (mean when active)
    feat_cond_mean = feat_sum / feat_count.clamp(min=1)

    # Activation frequency
    feat_freq = feat_count / total_count

    stats = {
        "mean": feat_mean.cpu().tolist(),
        "std": feat_std.cpu().tolist(),
        "max": feat_max.cpu().tolist(),
        "conditional_mean": feat_cond_mean.cpu().tolist(),
        "activation_frequency": feat_freq.cpu().tolist(),
        "total_samples": total_count,
    }

    # Log summary
    freq = torch.tensor(stats["activation_frequency"])
    logger.info(f"  Feature frequency range: [{freq.min():.4f}, {freq.max():.4f}]")
    logger.info(f"  Mean feature frequency: {freq.mean():.4f}")
    logger.info(f"  Features with freq > 0.5: {(freq > 0.5).sum().item()}")
    logger.info(f"  Features with freq < 0.01: {(freq < 0.01).sum().item()}")

    return stats


# =====================================================================
# 4. Decoder Weight Analysis
# =====================================================================

@torch.no_grad()
def analyze_decoder_weights(sae, output_dir):
    """Analyze the decoder weight matrix for structure."""
    logger.info("Analyzing decoder weight structure...")

    # Decoder columns are the feature directions in activation space
    # Shape: [activation_dim, dict_size]
    W_dec = sae.decoder.weight.data.float().cpu()

    dict_size = W_dec.shape[1]
    activation_dim = W_dec.shape[0]

    # Cosine similarity between all pairs of decoder columns
    W_dec_normed = F.normalize(W_dec, dim=0)  # [activation_dim, dict_size]
    cos_sim = W_dec_normed.T @ W_dec_normed  # [dict_size, dict_size]

    # Zero out diagonal
    cos_sim_offdiag = cos_sim.clone()
    cos_sim_offdiag.fill_diagonal_(0)

    # Find most similar/dissimilar feature pairs
    upper_tri = torch.triu(cos_sim_offdiag, diagonal=1)
    flat = upper_tri.flatten()
    nonzero_mask = flat != 0

    if nonzero_mask.sum() > 0:
        nonzero_sims = flat[nonzero_mask]
        top_similar_vals, top_similar_flat = nonzero_sims.topk(min(10, nonzero_sims.numel()))
        # Map back to full flat indices
        nonzero_indices = nonzero_mask.nonzero(as_tuple=True)[0]
        top_similar_flat_full = nonzero_indices[top_similar_flat]
        top_similar_pairs = [(int(idx // dict_size), int(idx % dict_size)) for idx in top_similar_flat_full]

        bottom_similar_vals, bottom_similar_flat = nonzero_sims.topk(min(10, nonzero_sims.numel()), largest=False)
        bottom_similar_flat_full = nonzero_indices[bottom_similar_flat]
        bottom_similar_pairs = [(int(idx // dict_size), int(idx % dict_size)) for idx in bottom_similar_flat_full]
    else:
        top_similar_pairs = []
        top_similar_vals = torch.tensor([])
        bottom_similar_pairs = []
        bottom_similar_vals = torch.tensor([])

    # Decoder column norms (should be ~1 if normalized)
    dec_norms = W_dec.norm(dim=0)

    # Encoder-decoder alignment
    W_enc = sae.encoder.weight.data.float().cpu()  # [dict_size, activation_dim]
    enc_dec_alignment = (F.normalize(W_enc, dim=1) * F.normalize(W_dec.T, dim=1)).sum(dim=1)

    results = {
        "cosine_sim_matrix": cos_sim.numpy().tolist(),
        "mean_pairwise_cosine": cos_sim_offdiag.abs().mean().item(),
        "max_pairwise_cosine": cos_sim_offdiag.abs().max().item(),
        "most_similar_pairs": [
            {"features": pair, "cosine_sim": val.item()}
            for pair, val in zip(top_similar_pairs, top_similar_vals)
        ],
        "most_dissimilar_pairs": [
            {"features": pair, "cosine_sim": val.item()}
            for pair, val in zip(bottom_similar_pairs, bottom_similar_vals)
        ],
        "decoder_norms": dec_norms.tolist(),
        "encoder_decoder_alignment": enc_dec_alignment.tolist(),
        "mean_enc_dec_alignment": enc_dec_alignment.mean().item(),
        "activation_dim": activation_dim,
        "dict_size": dict_size,
    }

    logger.info(f"  Mean pairwise cosine sim: {results['mean_pairwise_cosine']:.4f}")
    logger.info(f"  Max pairwise cosine sim:  {results['max_pairwise_cosine']:.4f}")
    logger.info(f"  Mean enc-dec alignment:   {results['mean_enc_dec_alignment']:.4f}")
    logger.info(f"  Decoder norm range:       [{dec_norms.min():.4f}, {dec_norms.max():.4f}]")

    # Save decoder weights and cosine sim matrix as tensors for visualization
    torch.save(W_dec, os.path.join(output_dir, "decoder_weights.pt"))
    torch.save(cos_sim, os.path.join(output_dir, "decoder_cosine_sim.pt"))

    return results


# =====================================================================
# 5. Feature-Action Correlation Analysis
# =====================================================================

@torch.no_grad()
def feature_action_correlation(
    sae,
    model,
    data_loader,
    hook_point,
    n_batches=100,
    device="cuda",
    dtype=torch.bfloat16,
):
    """
    Correlate feature activations with action dimensions.
    Helps understand if features encode action-relevant information.
    """
    logger.info(f"Computing feature-action correlations over {n_batches} batches...")

    collector = PI0ActivationCollector(model)
    collector.register_hooks([hook_point])

    param = model.action_in_proj.weight
    model_device, model_dtype = param.device, param.dtype

    all_features = []
    all_actions = []

    data_iter = iter(data_loader)
    for batch_idx in range(n_batches):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            batch = next(data_iter)

        observation, actions = batch if isinstance(batch, tuple) else (
            batch["observation"], batch["actions"]
        )

        # Keep actions for correlation
        if isinstance(actions, torch.Tensor):
            batch_actions = actions.detach().cpu().float()
        else:
            batch_actions = torch.tensor(actions, dtype=torch.float32)

        observation = to_device(observation, model_device, model_dtype)
        actions_dev = to_device(actions, model_device, model_dtype)

        with torch.no_grad():
            _ = model(observation, actions_dev)

        acts = collector.get_activations(flatten=False)[hook_point]
        collector.clear_activations()

        # Average feature activations over the sequence dimension for each sample
        if acts.ndim == 3:
            # [batch, seq, d_model] -> [batch, d_model]
            acts_mean = acts.mean(dim=1)
        else:
            acts_mean = acts

        features = sae.encode(acts_mean.to(device).float())  # [batch, dict_size]
        all_features.append(features.cpu())

        # Handle action shape: [batch, action_horizon, action_dim] -> [batch, action_dim] (mean over horizon)
        if batch_actions.ndim == 3:
            batch_actions = batch_actions.mean(dim=1)
        elif batch_actions.ndim == 1:
            batch_actions = batch_actions.unsqueeze(0)
        all_actions.append(batch_actions)

        if (batch_idx + 1) % 50 == 0:
            logger.info(f"  Processed {batch_idx + 1}/{n_batches} batches")

    collector.clear_hooks()

    all_features = torch.cat(all_features, dim=0)  # [N, dict_size]
    all_actions = torch.cat(all_actions, dim=0)  # [N, action_dim]

    # Truncate to minimum length
    min_n = min(all_features.shape[0], all_actions.shape[0])
    all_features = all_features[:min_n]
    all_actions = all_actions[:min_n]

    logger.info(f"  Features shape: {all_features.shape}, Actions shape: {all_actions.shape}")

    # Compute Pearson correlation between each feature and each action dimension
    # Normalize features and actions
    f_centered = all_features - all_features.mean(dim=0, keepdim=True)
    a_centered = all_actions - all_actions.mean(dim=0, keepdim=True)

    f_std = f_centered.std(dim=0, keepdim=True).clamp(min=1e-8)
    a_std = a_centered.std(dim=0, keepdim=True).clamp(min=1e-8)

    f_normed = f_centered / f_std
    a_normed = a_centered / a_std

    # Correlation matrix: [dict_size, action_dim]
    corr = (f_normed.T @ a_normed) / min_n

    # Find features most correlated with each action dimension
    action_dim = all_actions.shape[1]
    dict_size = all_features.shape[1]

    action_correlations = {}
    for a_idx in range(action_dim):
        col = corr[:, a_idx]
        top_pos = col.topk(min(5, dict_size))
        top_neg = col.topk(min(5, dict_size), largest=False)
        action_correlations[f"action_dim_{a_idx}"] = {
            "top_positive": [
                {"feature": int(idx), "correlation": float(val)}
                for idx, val in zip(top_pos.indices, top_pos.values)
            ],
            "top_negative": [
                {"feature": int(idx), "correlation": float(val)}
                for idx, val in zip(top_neg.indices, top_neg.values)
            ],
        }

    # Find features with highest absolute correlation to any action
    max_abs_corr_per_feature = corr.abs().max(dim=1).values
    top_correlated_features = max_abs_corr_per_feature.topk(min(10, dict_size))

    results = {
        "correlation_matrix": corr.numpy().tolist(),
        "action_dim": action_dim,
        "dict_size": dict_size,
        "n_samples": min_n,
        "action_correlations": action_correlations,
        "top_correlated_features": [
            {"feature": int(idx), "max_abs_correlation": float(val)}
            for idx, val in zip(top_correlated_features.indices, top_correlated_features.values)
        ],
        "mean_abs_correlation": corr.abs().mean().item(),
    }

    logger.info(f"  Mean |correlation|: {results['mean_abs_correlation']:.4f}")
    logger.info(f"  Top correlated features:")
    for entry in results["top_correlated_features"][:5]:
        logger.info(f"    Feature {entry['feature']}: max |r| = {entry['max_abs_correlation']:.4f}")

    # Save correlation matrix as tensor
    torch.save(corr, os.path.join(output_dir_global, "feature_action_correlation.pt"))

    return results


# Global for passing output_dir to correlation function
output_dir_global = ""


# =====================================================================
# Main
# =====================================================================

def main():
    global output_dir_global
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir_global = str(output_dir)

    # Save args
    with open(output_dir / "interpret_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    dtype = get_dtype(args.dtype)

    # Load model
    train_config = _config.get_config(args.config)
    model = load_pi0_model(train_config, args.checkpoint_path, args.device)

    # Load SAE
    sae = load_sae(args.sae_checkpoint, args.device)

    # Create data loader
    logger.info("Creating data loader...")
    data_loader = _data.create_data_loader(train_config, shuffle=True, framework="pytorch")

    # ---- 1. Health Metrics ----
    logger.info("=" * 60)
    logger.info("STEP 1: Computing SAE health metrics")
    logger.info("=" * 60)

    buffer = PI0ActivationBuffer(
        data_loader=data_loader,
        model=model,
        hook_point=args.hook_point,
        n_ctxs=5000,
        out_batch_size=4096,
        device=args.device,
        dtype=dtype,
    )

    health_metrics = compute_health_metrics(
        sae, buffer, n_batches=args.eval_batches, device=args.device
    )

    # Save (remove large list for JSON)
    fire_counts = health_metrics.pop("feature_fire_counts")
    with open(output_dir / "health_metrics.json", "w") as f:
        json.dump(health_metrics, f, indent=2)
    torch.save(torch.tensor(fire_counts), output_dir / "feature_fire_counts.pt")
    health_metrics["feature_fire_counts"] = fire_counts

    buffer.close()

    # ---- 2. Feature Statistics ----
    logger.info("=" * 60)
    logger.info("STEP 2: Computing per-feature activation statistics")
    logger.info("=" * 60)

    buffer2 = PI0ActivationBuffer(
        data_loader=data_loader,
        model=model,
        hook_point=args.hook_point,
        n_ctxs=5000,
        out_batch_size=4096,
        device=args.device,
        dtype=dtype,
    )

    feat_stats = compute_feature_statistics(
        sae, buffer2, n_batches=args.eval_batches, device=args.device
    )

    with open(output_dir / "feature_statistics.json", "w") as f:
        json.dump(feat_stats, f, indent=2)

    buffer2.close()

    # ---- 3. Decoder Weight Analysis ----
    logger.info("=" * 60)
    logger.info("STEP 3: Analyzing decoder weights")
    logger.info("=" * 60)

    decoder_analysis = analyze_decoder_weights(sae, str(output_dir))

    # Remove large matrix for JSON summary
    cos_sim_matrix = decoder_analysis.pop("cosine_sim_matrix")
    with open(output_dir / "decoder_analysis.json", "w") as f:
        json.dump(decoder_analysis, f, indent=2)

    # ---- 4. Top-Activating Examples ----
    logger.info("=" * 60)
    logger.info("STEP 4: Finding top-activating examples per feature")
    logger.info("=" * 60)

    top_examples = find_top_activating_examples(
        sae, model, data_loader, args.hook_point,
        n_batches=args.n_batches, top_k=args.top_k,
        device=args.device, dtype=dtype,
    )

    torch.save(top_examples["top_vals"], output_dir / "top_activation_values.pt")
    torch.save(top_examples["top_indices"], output_dir / "top_activation_indices.pt")

    # Save metadata (convert to serializable format)
    meta_summary = {}
    for feat_idx in range(sae.dict_size):
        feat_top = []
        for rank in range(args.top_k):
            val = top_examples["top_vals"][feat_idx, rank].item()
            idx = top_examples["top_indices"][feat_idx, rank].item()
            if idx >= 0 and idx in top_examples["sample_metadata"]:
                entry = {"activation": val, "global_idx": idx}
                meta = top_examples["sample_metadata"][idx]
                if "proprio_state" in meta:
                    entry["proprio_state"] = meta["proprio_state"]
                entry["token_pos"] = meta.get("token_pos", -1)
                entry["token_type"] = meta.get("token_type", "unknown")
                entry["batch_idx"] = meta.get("batch_idx", -1)
                feat_top.append(entry)
        meta_summary[f"feature_{feat_idx}"] = feat_top

    with open(output_dir / "top_activating_examples.json", "w") as f:
        json.dump(meta_summary, f, indent=2)

    # ---- 5. Feature-Action Correlations ----
    logger.info("=" * 60)
    logger.info("STEP 5: Computing feature-action correlations")
    logger.info("=" * 60)

    corr_results = feature_action_correlation(
        sae, model, data_loader, args.hook_point,
        n_batches=min(args.n_batches, 100),
        device=args.device, dtype=dtype,
    )

    # Remove large matrix for JSON summary
    corr_matrix = corr_results.pop("correlation_matrix")
    with open(output_dir / "feature_action_correlations.json", "w") as f:
        json.dump(corr_results, f, indent=2)

    # ---- Summary ----
    logger.info("=" * 60)
    logger.info("INTERPRETATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Files:")
    for f in sorted(output_dir.iterdir()):
        size_mb = f.stat().st_size / (1024 * 1024)
        logger.info(f"  {f.name:40s} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
