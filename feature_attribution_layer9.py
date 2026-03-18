"""
Feature attribution for SAE features at lang_layer_9 using gradient × activation.

For each data point, computes:
  GxA attribution_i = (∂L/∂h · d_i) * f_i(h)

where:
  h    = lang_layer_9 activations [batch, seq, d_model]
  d_i  = i-th SAE decoder direction (column of decoder weight matrix [d_model])
  f_i  = SAE feature activation for feature i (from BatchTopKSAE.encode())
  L    = sum of per-element MSE training loss

This is the Gradient × Activation (GxA) attribution, which estimates how much
each SAE feature causally influences the training objective.

Aggregates |attribution_i| across the dataset to get a per-feature importance score.

Outputs:
  feature_attribution_scores.pt      [dict_size]  mean |GxA attr| per feature
  feature_grad_projection.pt         [dict_size]  mean |grad · d_i| per feature
  feature_activation_magnitude.pt    [dict_size]  mean |f_i| per feature
  feature_attribution_summary.json   human-readable top-feature rankings

Usage:
    python feature_attribution_layer9.py \\
        --config pi05_libero \\
        --checkpoint_path /path/to/model.safetensors \\
        --sae_checkpoint /path/to/ae.pt \\
        --output_dir ./sae_interpretations/pi05_libero_lang_layer_9_k32_exp4 \\
        --n_batches 100
"""

import argparse
import json
import logging
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import safetensors.torch

# Stub out nnsight before dictionary_learning imports
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
from openpi.interpretability.activation_hooks import to_device


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Gradient × Activation feature attribution for SAE at lang_layer_9")
    parser.add_argument("--config", type=str, required=True, help="Training config name (e.g., pi05_libero)")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to PI0 model checkpoint")
    parser.add_argument("--hook_point", type=str, default="lang_layer_9", help="Hook point to attribute at")
    parser.add_argument("--sae_checkpoint", type=str, required=True, help="Path to trained SAE checkpoint (ae.pt)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--n_batches", type=int, default=100, help="Number of batches for attribution")
    parser.add_argument("--data_batch_size", type=int, default=8, help="Batch size (smaller needed for gradients)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"])
    return parser.parse_args()


def get_dtype(dtype_str):
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_str]


def load_pi0_model(config, checkpoint_path, device):
    from openpi.models_pytorch import pi0_pytorch
    logger.info(f"Loading PI0 model...")
    model = pi0_pytorch.PI0Pytorch(config=config.model)
    if checkpoint_path:
        logger.info(f"  Loading checkpoint from: {checkpoint_path}")
        safetensors.torch.load_model(model, checkpoint_path)
    else:
        logger.warning("No checkpoint path provided - using random weights!")
    model = model.to(device)
    model.eval()
    model.sample_actions = model.sample_actions.__wrapped__
    return model


def load_sae(checkpoint_path, device):
    logger.info(f"Loading SAE from: {checkpoint_path}")
    sae = BatchTopKSAE.from_pretrained(checkpoint_path, device=device)
    sae.eval()
    logger.info(f"  activation_dim={sae.activation_dim}, dict_size={sae.dict_size}, k={sae.k.item()}")
    return sae


def get_layer_idx_from_hook(hook_point: str) -> int:
    """Extract numeric layer index from e.g. 'lang_layer_9' -> 9."""
    return int(hook_point.split("_")[-1])


def is_expert_hook(hook_point: str) -> bool:
    return hook_point.startswith("expert_")


def compute_feature_attribution(
    model,
    sae,
    data_loader,
    hook_point: str = "lang_layer_9",
    n_batches: int = 100,
    device: str = "cuda",
):
    """
    Compute per-feature gradient × activation attribution at the given hook_point.

    The attribution for SAE feature i at a single token position t is:
        attr_i(t) = (grad_h(t) · d_i) * f_i(t)

    where grad_h = ∂L/∂h (gradient of summed loss w.r.t. layer activations).

    We aggregate |attr_i| across all tokens and batches to get a single
    importance score per feature.

    Returns dict with:
        attribution            [dict_size]  mean |GxA| per feature
        gradient_projection    [dict_size]  mean |grad · d_i| per feature
        activation_magnitude   [dict_size]  mean |f_i| per feature
        signed_attribution     [dict_size]  mean signed GxA (sign = direction of influence)
    """
    layer_idx = get_layer_idx_from_hook(hook_point)
    pg_model = model.paligemma_with_expert

    # Enable attribution mode: cache activations WITHOUT detach so gradients flow back
    if is_expert_hook(hook_point):
        pg_model.enable_activation_cache(expert_layers=[layer_idx], attribution_mode=True)
    else:
        pg_model.enable_activation_cache(lang_layers=[layer_idx], attribution_mode=True)

    dict_size = sae.dict_size
    activation_dim = sae.activation_dim

    # Decoder weight matrix [activation_dim, dict_size] — each column is a feature direction
    W_dec = sae.decoder.weight.data.float().cpu()  # [activation_dim, dict_size]

    # Running accumulators (on CPU to save GPU memory)
    attr_abs_sum = torch.zeros(dict_size)      # sum of |grad_proj * feat|
    attr_signed_sum = torch.zeros(dict_size)   # sum of (grad_proj * feat)  — signed
    grad_proj_abs_sum = torch.zeros(dict_size) # sum of |grad · d_i|
    feat_abs_sum = torch.zeros(dict_size)      # sum of |f_i|
    total_tokens = 0
    count = 0

    param = model.action_in_proj.weight
    model_device, model_dtype = param.device, param.dtype

    # --- Detach SigLIP from the gradient graph ---
    # SigLIP (ViT-So400M, ~400M params) builds a large forward graph that causes OOM.
    # Since we only need ∂loss/∂acts_ref where acts_ref is at a language-transformer layer,
    # gradients do NOT need to flow back through SigLIP. We patch embed_image to run under
    # torch.no_grad() and return a detached tensor, so only the language transformer layers
    # (after the image embeddings are computed) are in the gradient graph.
    _orig_embed_image = None
    if hasattr(pg_model, "embed_image"):
        _orig_embed_image = pg_model.embed_image

        def _embed_image_no_grad(img):
            with torch.no_grad():
                return _orig_embed_image(img).detach()

        pg_model.embed_image = _embed_image_no_grad

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

        observation = to_device(observation, model_device, model_dtype)
        actions = to_device(actions, model_device, model_dtype)

        # Clear any previous cached activations
        pg_model.clear_cached_activations()

        # --- Forward pass WITH gradients (SigLIP runs without grad to save memory) ---
        loss = model.forward(observation, actions)   # [batch, horizon, action_dim]
        scalar_loss = loss.sum()

        # Retrieve the cached activation tensor (still in computation graph)
        acts_ref = pg_model._cached_activations.get(hook_point)
        if acts_ref is None:
            logger.warning(f"No cached activation for {hook_point} at batch {batch_idx}; skipping")
            pg_model.clear_cached_activations()
            continue

        if acts_ref.grad_fn is None:
            logger.warning(
                f"acts_ref has no grad_fn at batch {batch_idx} — "
                "attribution mode may not be enabled correctly; skipping"
            )
            pg_model.clear_cached_activations()
            continue

        # --- Compute gradient of loss w.r.t. lang_layer_9 activations ---
        # torch.autograd.grad avoids accumulating parameter gradients (more memory-efficient
        # than .backward() which would set .grad on all model parameters).
        try:
            (grad,) = torch.autograd.grad(
                outputs=scalar_loss,
                inputs=acts_ref,
                retain_graph=False,
                create_graph=False,
            )
        except RuntimeError as e:
            logger.warning(f"Gradient computation failed at batch {batch_idx}: {e}; skipping")
            pg_model.clear_cached_activations()
            torch.cuda.empty_cache()
            continue

        # acts_ref: [batch, seq, d_model], grad: [batch, seq, d_model]
        acts_flat = acts_ref.detach().reshape(-1, activation_dim).float().cpu()  # [n_tokens, d_model]
        grad_flat = grad.reshape(-1, activation_dim).float().cpu()               # [n_tokens, d_model]
        n_tokens = acts_flat.shape[0]

        # SAE encode activations -> feature values [n_tokens, dict_size]
        with torch.no_grad():
            feats = sae.encode(acts_flat.to(device)).float().cpu()  # [n_tokens, dict_size]

        # Project gradient onto each decoder direction: [n_tokens, dict_size]
        #   grad_proj[t, i] = grad_flat[t] · W_dec[:, i]
        grad_proj = grad_flat @ W_dec  # [n_tokens, dict_size]

        # GxA attribution: (grad · d_i) * f_i
        attr = grad_proj * feats  # [n_tokens, dict_size]

        # Accumulate
        attr_abs_sum    += attr.abs().sum(dim=0)
        attr_signed_sum += attr.sum(dim=0)
        grad_proj_abs_sum += grad_proj.abs().sum(dim=0)
        feat_abs_sum    += feats.abs().sum(dim=0)
        total_tokens    += n_tokens
        count           += 1

        # Free GPU memory
        del loss, scalar_loss, grad, acts_flat, grad_flat, feats, grad_proj, attr
        torch.cuda.empty_cache()

        if (batch_idx + 1) % 10 == 0:
            logger.info(
                f"  Processed {batch_idx + 1}/{n_batches} batches "
                f"({total_tokens} tokens, ~{total_tokens/max(count,1):.0f} tokens/batch)"
            )

    # Restore original embed_image
    if _orig_embed_image is not None:
        pg_model.embed_image = _orig_embed_image

    # Disable attribution mode
    pg_model.disable_activation_cache()

    if total_tokens == 0:
        raise RuntimeError(
            "No tokens were processed! Check that the hook_point is correct and "
            "that the model is running in the right mode."
        )

    logger.info(f"  Done: {count} batches, {total_tokens} total tokens")

    # Normalize by total tokens to get per-token mean scores
    attr_mean        = attr_abs_sum    / total_tokens
    attr_signed_mean = attr_signed_sum / total_tokens
    grad_proj_mean   = grad_proj_abs_sum / total_tokens
    feat_mean        = feat_abs_sum    / total_tokens

    logger.info(f"  Mean |attribution|:         {attr_mean.mean():.6f}")
    logger.info(f"  Max  |attribution|:         {attr_mean.max():.6f}  "
                f"(feature {attr_mean.argmax().item()})")
    logger.info(f"  Mean |grad projection|:     {grad_proj_mean.mean():.6f}")
    logger.info(f"  Max  |grad projection|:     {grad_proj_mean.max():.6f}  "
                f"(feature {grad_proj_mean.argmax().item()})")

    return {
        "attribution":          attr_mean,        # [dict_size] mean |GxA|
        "signed_attribution":   attr_signed_mean, # [dict_size] mean signed GxA
        "gradient_projection":  grad_proj_mean,   # [dict_size] mean |grad · d_i|
        "activation_magnitude": feat_mean,        # [dict_size] mean |f_i|
        "total_tokens":         total_tokens,
        "n_batches":            count,
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = get_dtype(args.dtype)

    # Load model
    train_config = _config.get_config(args.config)
    model = load_pi0_model(train_config, args.checkpoint_path, args.device)

    # Load SAE
    sae = load_sae(args.sae_checkpoint, args.device)

    # Create data loader (uses config.batch_size)
    logger.info(f"Creating data loader (batch_size from config: {train_config.batch_size})...")
    data_loader = _data.create_data_loader(train_config, shuffle=True, framework="pytorch")

    # Save args
    with open(output_dir / "attribution_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── Run attribution ────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"Computing gradient × activation attribution at {args.hook_point}")
    logger.info(f"  n_batches={args.n_batches}")
    logger.info("=" * 60)

    results = compute_feature_attribution(
        model=model,
        sae=sae,
        data_loader=data_loader,
        hook_point=args.hook_point,
        n_batches=args.n_batches,
        device=args.device,
    )

    # ── Save tensors ───────────────────────────────────────────────────────
    torch.save(results["attribution"],          output_dir / "feature_attribution_scores.pt")
    torch.save(results["signed_attribution"],   output_dir / "feature_attribution_signed.pt")
    torch.save(results["gradient_projection"],  output_dir / "feature_grad_projection.pt")
    torch.save(results["activation_magnitude"], output_dir / "feature_activation_magnitude.pt")

    # ── Build JSON summary ─────────────────────────────────────────────────
    attr      = results["attribution"]
    attr_sign = results["signed_attribution"]
    grad_proj = results["gradient_projection"]
    feat_mag  = results["activation_magnitude"]

    # Top 30 features by absolute GxA attribution
    top_n = min(30, sae.dict_size)
    top_attr = attr.topk(top_n)

    # Top 30 by gradient projection (direction importance regardless of current activation)
    top_grad = grad_proj.topk(top_n)

    summary = {
        "hook_point":     args.hook_point,
        "dict_size":      sae.dict_size,
        "activation_dim": sae.activation_dim,
        "k":              int(sae.k.item()),
        "total_tokens":   results["total_tokens"],
        "n_batches":      results["n_batches"],
        "method":         "gradient_times_activation (GxA)",
        "description": (
            "attribution_i = mean_t |( dL/dh_t · d_i ) * f_i(h_t)| "
            "summed over all token positions t, averaged over dataset. "
            "d_i = i-th SAE decoder column, f_i = SAE feature activation, "
            "h = lang_layer_9 activations, L = sum of MSE training loss."
        ),
        "attribution_stats": {
            "mean":       float(attr.mean()),
            "std":        float(attr.std()),
            "max":        float(attr.max()),
            "max_feature": int(attr.argmax()),
            "min":        float(attr.min()),
        },
        "gradient_projection_stats": {
            "mean":       float(grad_proj.mean()),
            "std":        float(grad_proj.std()),
            "max":        float(grad_proj.max()),
            "max_feature": int(grad_proj.argmax()),
        },
        "top_features_by_gxa_attribution": [
            {
                "rank":               i + 1,
                "feature":            int(idx),
                "gxa_attribution":    float(val),
                "signed_attribution": float(attr_sign[idx]),
                "grad_projection":    float(grad_proj[idx]),
                "activation_magnitude": float(feat_mag[idx]),
            }
            for i, (idx, val) in enumerate(zip(top_attr.indices, top_attr.values))
        ],
        "top_features_by_gradient_projection": [
            {
                "rank":               i + 1,
                "feature":            int(idx),
                "grad_projection":    float(val),
                "gxa_attribution":    float(attr[idx]),
                "signed_attribution": float(attr_sign[idx]),
                "activation_magnitude": float(feat_mag[idx]),
            }
            for i, (idx, val) in enumerate(zip(top_grad.indices, top_grad.values))
        ],
    }

    with open(output_dir / "feature_attribution_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── Final log ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("ATTRIBUTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"")
    logger.info(f"Top 10 features by gradient × activation attribution (|GxA|):")
    logger.info(f"  {'Rank':>4}  {'Feature':>7}  {'|GxA|':>10}  {'signed':>10}  {'|grad·d|':>10}  {'|f_i|':>8}")
    logger.info(f"  {'-'*60}")
    for entry in summary["top_features_by_gxa_attribution"][:10]:
        logger.info(
            f"  {entry['rank']:4d}  {entry['feature']:7d}  "
            f"{entry['gxa_attribution']:10.6f}  "
            f"{entry['signed_attribution']:10.6f}  "
            f"{entry['grad_projection']:10.6f}  "
            f"{entry['activation_magnitude']:8.6f}"
        )
    logger.info(f"")
    logger.info(f"Output files:")
    for p in sorted(output_dir.glob("feature_attribution*")):
        size_mb = p.stat().st_size / 1024 / 1024
        logger.info(f"  {p.name:50s} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
