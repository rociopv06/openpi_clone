"""
jacobian_feature_attribution.py
================================
Jacobian-based feature attribution for SAE features, following the "Jacobian Scopes"
methodology from arxiv 2601.16407.

For each SAE feature i with decoder direction d_i, computes:
    J_i = ∂(action_output) / ∂h · d_i    [action_dim]  (full action-space influence vector)

Where:
    h         = hook_point activations [batch, seq, d_model]
    d_i       = i-th SAE decoder column [d_model]
    J_i       = Jacobian product: how does the action output change per unit of feature i
    ‖J_i‖₂   = scalar influence score (Jacobian norm), input-output causal influence

The GxA approach (feature_attribution_layer9.py) uses loss gradients and weights by
current activation. The Jacobian approach measures direct linearized input→output
influence regardless of current activation magnitude.

Implementation: For the full Jacobian we want ∂y_j/∂h for each output dim j.
This is expensive (one backward per output dim). Instead we use the Frobenius norm:
    ‖J · D‖_F = √( Σ_j ‖∂y_j/∂h · D‖₂² )
which can be approximated via random projections (Hutchinson trace estimator).

Alternatively, we compute the INFLUENCE score for each feature:
    influence_i = ‖∂y/∂h · d_i‖₂ = ‖J · d_i‖₂
where ∂y/∂h is the [action_dim, d_model] Jacobian.

Since action_dim is typically small (7-16 dims), we can compute all action-dim rows
of J exactly with action_dim backward passes (e.g. 16 for LIBERO).

For efficiency:
  - SigLIP runs with torch.no_grad() + detach to prevent OOM
  - action_dim backward passes per batch (one per action output dimension)
  - All feature Jacobian norms computed in one matmul per backward pass

Outputs:
    jacobian_influence_scores.pt          [dict_size]  ‖J·d_i‖₂ averaged over dataset
    jacobian_signed_influence.pt          [dict_size, action_dim]  mean J·d_i per dim
    jacobian_influence_summary.json       top features ranked by influence

Usage:
    python jacobian_feature_attribution.py \\
        --config pi05_libero \\
        --checkpoint_path /path/to/model.safetensors \\
        --hook_point lang_layer_9 \\
        --sae_checkpoint /path/to/ae.pt \\
        --output_dir ./sae_interpretations/pi05_libero_lang_layer_9_k32_exp4 \\
        --n_batches 50 \\
        --data_batch_size 1
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
    parser = argparse.ArgumentParser(description="Jacobian feature attribution for SAE (arxiv 2601.16407)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--hook_point", type=str, default="lang_layer_9")
    parser.add_argument("--sae_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_batches", type=int, default=50, help="Batches to process (Jacobian is expensive)")
    parser.add_argument("--data_batch_size", type=int, default=1, help="Batch size (keep small due to Jacobian memory)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--max_seq_tokens", type=int, default=16,
                        help="Max token positions to sum Jacobian over per sample (limit to save memory)")
    return parser.parse_args()


def get_dtype(dtype_str):
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_str]


def get_layer_idx_from_hook(hook_point: str) -> int:
    return int(hook_point.split("_")[-1])


def is_expert_hook(hook_point: str) -> bool:
    return hook_point.startswith("expert_")


def load_pi0_model(config, checkpoint_path, device):
    from openpi.models_pytorch import pi0_pytorch
    logger.info("Loading PI0 model...")
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


def compute_jacobian_attribution(
    model,
    sae,
    data_loader,
    hook_point: str = "lang_layer_9",
    n_batches: int = 50,
    device: str = "cuda",
    max_seq_tokens: int = 16,
):
    """
    Compute Jacobian-based feature attribution (Jacobian Scopes, arxiv 2601.16407).

    For each SAE feature i, computes:
        influence_i = mean_batches [ ‖J · d_i‖₂ ]

    where J = ∂y/∂h is the [action_flat_dim, d_model] Jacobian of the action output
    w.r.t. the hook layer activations h.

    Since action_dim is small (e.g. 16 for LIBERO 7-DoF × chunk_size),
    we can compute J exactly by running one backward pass per action output dimension.

    To keep memory manageable:
    - SigLIP runs without gradients (detached from graph)
    - We sum over token positions rather than keeping per-token Jacobians
    - max_seq_tokens limits how many token positions contribute per sample
    """
    layer_idx = get_layer_idx_from_hook(hook_point)
    pg_model = model.paligemma_with_expert

    # Enable activation cache (without attribution_mode so it just caches with detach—
    # we will manually set requires_grad after)
    if is_expert_hook(hook_point):
        pg_model.enable_activation_cache(expert_layers=[layer_idx], attribution_mode=True)
    else:
        pg_model.enable_activation_cache(lang_layers=[layer_idx], attribution_mode=True)

    dict_size = sae.dict_size
    activation_dim = sae.activation_dim

    # Decoder directions: [activation_dim, dict_size]
    W_dec = sae.decoder.weight.data.float()  # [activation_dim, dict_size]
    W_dec_cpu = W_dec.cpu()

    # Accumulators
    influence_sum = torch.zeros(dict_size)          # Σ ‖J·d_i‖₂
    signed_influence_sum = None                      # Σ J·d_i (vector per feature)
    total_samples = 0
    count = 0

    param = model.action_in_proj.weight
    model_device, model_dtype = param.device, param.dtype

    # Detach SigLIP from gradient graph to prevent OOM
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

        pg_model.clear_cached_activations()

        # Forward pass to get action output and cached activations
        action_pred = model.forward(observation, actions)  # [B, horizon, action_dim]

        acts_ref = pg_model._cached_activations.get(hook_point)
        if acts_ref is None or acts_ref.grad_fn is None:
            logger.warning(f"Missing or non-differentiable acts_ref at batch {batch_idx}; skipping")
            pg_model.clear_cached_activations()
            continue

        # action_pred shape: [B, horizon, action_dim]
        # We want ∂(action_pred_flat) / ∂(acts_ref) = [B*horizon*action_dim, B*seq*d_model]
        # That's huge. Instead, sum over batch and sum/mean over positions:
        # y = action_pred.sum(dim=(0,1))  [action_dim]  — represents total action output
        # Then J_j = ∂y_j/∂h = [B, seq, d_model]

        # For influence: ‖J_j · d_i‖₂ summed over action dims j
        # = ‖(∂y/∂h) · d_i‖₂ in the Frobenius sense

        B = action_pred.shape[0]
        action_flat = action_pred.reshape(B, -1)  # [B, horizon*action_dim]
        action_total_dim = action_flat.shape[1]

        # Compute J row by row: one backward per action output dim
        # J: [action_total_dim, B, seq, d_model] — too large to hold in memory
        # Instead accumulate ‖Σ_j ∂y_j/∂h · d_i‖₂² incrementally

        # Accumulate J·d_i = [action_total_dim, d_model] · [d_model, dict_size] → [action_total_dim, dict_size]
        # We sum over (B, seq) positions first (mean), then dot with W_dec

        # jac_dict_i[j] = (∂y_j/∂h).mean(over B,seq) · d_i  — shape [dict_size]
        # influence_i = ‖[jac_dict_i[0], ..., jac_dict_i[action_dim-1]]‖₂

        jac_W = torch.zeros(action_total_dim, dict_size)  # [action_dim, dict_size] accumulated on CPU

        valid = True
        for j in range(action_total_dim):
            try:
                # ∂y_j/∂h: shape [B, seq, d_model]
                (grad_j,) = torch.autograd.grad(
                    outputs=action_flat[:, j].sum(),
                    inputs=acts_ref,
                    retain_graph=(j < action_total_dim - 1),
                    create_graph=False,
                )
            except RuntimeError as e:
                logger.warning(f"Jacobian backward failed at batch {batch_idx}, dim {j}: {e}")
                valid = False
                break

            # Average over (B, seq) positions, project onto decoder directions
            grad_j_mean = grad_j.reshape(-1, activation_dim).float().mean(dim=0).cpu()  # [d_model]
            # [d_model] · [d_model, dict_size] → [dict_size]
            jac_W[j] = grad_j_mean @ W_dec_cpu

            del grad_j

        if not valid:
            pg_model.clear_cached_activations()
            torch.cuda.empty_cache()
            continue

        # influence_i = ‖jac_W[:, i]‖₂  (Frobenius over action dims)
        # jac_W: [action_total_dim, dict_size]
        influence_batch = jac_W.norm(dim=0)  # [dict_size]
        influence_sum += influence_batch

        if signed_influence_sum is None:
            signed_influence_sum = torch.zeros(action_total_dim, dict_size)
        signed_influence_sum += jac_W  # accumulate signed Jacobian projections

        total_samples += B
        count += 1

        del action_pred, action_flat, jac_W, influence_batch
        torch.cuda.empty_cache()

        if (batch_idx + 1) % 5 == 0:
            logger.info(
                f"  Processed {batch_idx + 1}/{n_batches} batches "
                f"({count} valid, {total_samples} total samples)"
            )

    # Restore embed_image
    if _orig_embed_image is not None:
        pg_model.embed_image = _orig_embed_image

    pg_model.disable_activation_cache()

    if count == 0:
        raise RuntimeError("No batches were successfully processed. Check hook_point and model setup.")

    logger.info(f"Done: {count} valid batches, {total_samples} total samples")

    influence_mean = influence_sum / count          # [dict_size]
    signed_mean = signed_influence_sum / count if signed_influence_sum is not None else None

    logger.info(f"  Mean influence:  {influence_mean.mean():.6f}")
    logger.info(f"  Max  influence:  {influence_mean.max():.6f}  (feature {influence_mean.argmax().item()})")

    return {
        "influence": influence_mean,           # [dict_size]
        "signed_influence": signed_mean,       # [action_total_dim, dict_size]
        "total_samples": total_samples,
        "n_batches": count,
        "action_total_dim": action_total_dim if count > 0 else 0,
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    train_config = _config.get_config(args.config)
    model = load_pi0_model(train_config, args.checkpoint_path, args.device)

    # Load SAE
    sae = load_sae(args.sae_checkpoint, args.device)

    # Create data loader
    logger.info("Creating data loader...")
    data_loader = _data.create_data_loader(train_config, shuffle=True, framework="pytorch")

    # Save args
    with open(output_dir / "jacobian_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    logger.info("=" * 60)
    logger.info(f"Jacobian attribution at {args.hook_point}")
    logger.info(f"  n_batches={args.n_batches}  (one backward per action dim per batch)")
    logger.info("=" * 60)

    results = compute_jacobian_attribution(
        model=model,
        sae=sae,
        data_loader=data_loader,
        hook_point=args.hook_point,
        n_batches=args.n_batches,
        device=args.device,
        max_seq_tokens=args.max_seq_tokens,
    )

    # Save
    torch.save(results["influence"], output_dir / "jacobian_influence_scores.pt")
    if results["signed_influence"] is not None:
        torch.save(results["signed_influence"], output_dir / "jacobian_signed_influence.pt")

    # Summary
    influence = results["influence"]
    top_n = min(30, sae.dict_size)
    top_inf = influence.topk(top_n)

    signed = results["signed_influence"]  # [action_dim, dict_size] or None

    summary = {
        "hook_point": args.hook_point,
        "dict_size": sae.dict_size,
        "activation_dim": sae.activation_dim,
        "k": int(sae.k.item()),
        "total_samples": results["total_samples"],
        "n_batches": results["n_batches"],
        "action_total_dim": results["action_total_dim"],
        "method": "Jacobian Scopes (arxiv 2601.16407): influence_i = ||∂y/∂h · d_i||₂",
        "description": (
            "influence_i = ‖J·d_i‖₂ where J = ∂(action_output)/∂h, "
            "d_i = i-th SAE decoder direction [d_model]. "
            "Measures direct linearized causal influence of feature i on action output. "
            "Unlike GxA, does NOT weight by current feature activation — measures structural importance."
        ),
        "influence_stats": {
            "mean": float(influence.mean()),
            "std": float(influence.std()),
            "max": float(influence.max()),
            "max_feature": int(influence.argmax()),
            "min": float(influence.min()),
        },
        "top_features_by_jacobian_influence": [
            {
                "rank": i + 1,
                "feature": int(idx),
                "jacobian_influence": float(val),
                "signed_influence_per_dim": (
                    signed[:, idx].tolist() if signed is not None else None
                ),
            }
            for i, (idx, val) in enumerate(zip(top_inf.indices, top_inf.values))
        ],
    }

    with open(output_dir / "jacobian_influence_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info("JACOBIAN ATTRIBUTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"")
    logger.info(f"Top 10 features by Jacobian influence ‖∂y/∂h·d_i‖₂:")
    logger.info(f"  {'Rank':>4}  {'Feature':>7}  {'‖J·d_i‖₂':>12}")
    logger.info(f"  {'-'*30}")
    for entry in summary["top_features_by_jacobian_influence"][:10]:
        logger.info(f"  {entry['rank']:4d}  {entry['feature']:7d}  {entry['jacobian_influence']:12.6f}")


if __name__ == "__main__":
    main()
