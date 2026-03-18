"""
verify_top_examples.py
======================
Causal verification that the top-activating examples in top_activating_examples.json
actually correspond to the correct frames.

For each of N_FEATURES features, takes the top K examples, loads their matched
parquet frame, runs it through PI0 → SAE, and checks:
  1. State match: does the parquet frame's normalized state equal the stored proprio_state?
  2. Activation match: does running the frame through PI0+SAE produce the stored activation?

A high correlation between stored and recomputed activations is the causal proof
that the metadata fix was correct.

Usage:
    python verify_top_examples.py \\
        --config pi05_libero \\
        --checkpoint_path /path/to/model.safetensors \\
        --sae_checkpoint /path/to/ae.pt \\
        --interp_dir /path/to/sae_interpretations/pi05_libero_lang_layer_9_k32_exp4 \\
        --hook_point lang_layer_9 \\
        --n_features 20 \\
        --top_k 5
"""

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import safetensors.torch

if "nnsight" not in sys.modules:
    stub = types.ModuleType("nnsight")
    stub.LanguageModel = None
    sys.modules["nnsight"] = stub

from openpi.interpretability.dictionary_learning.dictionary_learning.trainers.batch_top_k import BatchTopKSAE
from openpi.interpretability.activation_hooks import PI0ActivationCollector, to_device
import openpi.training.config as _config
import openpi.training.data_loader as _data

LIBERO_CACHE = Path("/storage/project/r-agarg35-0/rvaldes6/.cache/huggingface/lerobot/physical-intelligence/libero")
NORM_STATS_PATH = Path("/storage/project/r-agarg35-0/rvaldes6/openpi_clone/assets/pi05_libero/physical-intelligence/libero/norm_stats.json")
MATCH_DECIMALS = 4


def load_quantile_stats():
    with open(NORM_STATS_PATH) as f:
        data = json.load(f)
    stats = data["norm_stats"]["state"]
    return np.array(stats["q01"], dtype=np.float32), np.array(stats["q99"], dtype=np.float32)


def quantile_normalize(state, q01, q99):
    return (state - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


def state_key(norm_state, decimals=MATCH_DECIMALS):
    return tuple(np.round(np.array(norm_state[:8], dtype=np.float64), decimals).tolist())


def build_parquet_lookup(q01, q99):
    lookup = {}
    data_dir = LIBERO_CACHE / "data"
    for chunk_dir in sorted(data_dir.iterdir()):
        for pf in sorted(chunk_dir.glob("episode_*.parquet")):
            table = pq.read_table(pf, columns=["state", "episode_index", "frame_index", "task_index"])
            states = np.array(table["state"].to_pylist(), dtype=np.float32)
            ep_idx = table["episode_index"].to_pylist()
            fr_idx = table["frame_index"].to_pylist()
            tk_idx = table["task_index"].to_pylist()
            norm_states = quantile_normalize(states, q01, q99)
            for i in range(len(table)):
                k = state_key(norm_states[i])
                lookup[k] = (int(ep_idx[i]), int(fr_idx[i]), int(tk_idx[i]), pf, i)
    print(f"Built lookup: {len(lookup)} frames", flush=True)
    return lookup


def load_frame_row(pf, row_idx):
    """Load a single row's observation image bytes and state."""
    table = pq.read_table(pf, columns=["state", "observation.images.cam_high", "observation.images.cam_low"])
    row = {col: table[col][row_idx] for col in table.schema.names}
    state = np.array(table["state"][row_idx].as_py(), dtype=np.float32)
    return row, state


def read_features_from_json(json_path, feature_ids):
    """Stream JSON to extract only the requested feature entries."""
    results = {}
    target_keys = {f'"feature_{i}"' for i in feature_ids}
    collecting = False
    current_id = None
    depth = 0
    buf = []

    with open(json_path) as fh:
        for line in fh:
            s = line.strip()
            if not collecting:
                for tk in list(target_keys):
                    if tk in s:
                        current_id = int(tk.strip('"').split("_")[1])
                        collecting = True
                        bp = s.find("[")
                        if bp != -1:
                            rest = s[bp:]
                            buf = [rest]
                            depth = rest.count("[") - rest.count("]")
                        break
            else:
                buf.append(line)
                depth += line.count("[") - line.count("]")
                if depth <= 0:
                    raw = "".join(buf).strip().rstrip(",")
                    try:
                        results[current_id] = json.loads(raw)
                    except Exception:
                        pass
                    collecting = False
                    buf = []
                    depth = 0
                    target_keys.discard(f'"feature_{current_id}"')
                    if not target_keys:
                        break
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="pi05_libero")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--sae_checkpoint", required=True)
    parser.add_argument("--interp_dir", required=True)
    parser.add_argument("--hook_point", default="lang_layer_9")
    parser.add_argument("--n_features", type=int, default=20, help="Number of features to verify")
    parser.add_argument("--top_k", type=int, default=5, help="Examples per feature to verify")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    interp_dir = Path(args.interp_dir)
    device = args.device

    # ── Load normalization stats & parquet lookup ──
    print("Loading norm stats and parquet lookup...", flush=True)
    q01, q99 = load_quantile_stats()
    lookup = build_parquet_lookup(q01, q99)

    # ── Choose features to verify (top N by peak activation) ──
    top_vals = torch.load(interp_dir / "top_activation_values.pt")   # [dict_size, top_k]
    peak = top_vals[:, 0]  # highest activation per feature
    top_feature_ids = peak.argsort(descending=True)[:args.n_features].tolist()
    print(f"\nVerifying features (top {args.n_features} by peak activation): {top_feature_ids[:10]}...", flush=True)

    # ── Stream JSON for those features ──
    json_path = interp_dir / "top_activating_examples.json"
    print(f"Streaming {json_path.name} ...", flush=True)
    feature_data = read_features_from_json(json_path, set(top_feature_ids))
    print(f"Loaded {len(feature_data)} features from JSON", flush=True)

    # ── Load PI0 model ──
    print("\nLoading PI0 model...", flush=True)
    from openpi.models_pytorch import pi0_pytorch
    train_config = _config.get_config(args.config)
    model = pi0_pytorch.PI0Pytorch(config=train_config.model)
    safetensors.torch.load_model(model, args.checkpoint_path)
    model = model.to(device)
    model.eval()
    model.sample_actions = model.sample_actions.__wrapped__

    # ── Load SAE ──
    sae = BatchTopKSAE.from_pretrained(args.sae_checkpoint, device=device)
    sae.eval()
    print(f"SAE: activation_dim={sae.activation_dim}, dict_size={sae.dict_size}", flush=True)

    # ── Setup activation collector ──
    collector = PI0ActivationCollector(model)
    collector.register_hooks([args.hook_point])

    # ── Load data loader to get observation structure for forward pass ──
    data_loader = _data.create_data_loader(train_config, shuffle=False, framework="pytorch")
    param = model.action_in_proj.weight
    model_device, model_dtype = param.device, param.dtype

    # ── Verify each feature ──
    results = []
    stored_activations = []
    recomputed_activations = []
    state_match_errors = []

    # We'll run a batch from the data loader, map frames to examples
    # For each top example: find frame in parquet, verify state matches stored proprio
    print("\n=== STATE MATCH VERIFICATION (no model needed) ===", flush=True)
    n_matched = 0
    n_state_correct = 0
    n_total = 0

    for feat_id in top_feature_ids:
        if feat_id not in feature_data:
            continue
        examples = feature_data[feat_id][:args.top_k]
        for ex in examples:
            n_total += 1
            stored_proprio = ex.get("proprio_state", [])
            stored_activation = ex.get("activation", 0)
            token_type = ex.get("token_type", "unknown")
            token_pos = ex.get("token_pos", -1)

            if not stored_proprio:
                continue

            k = state_key(stored_proprio[:8])
            if k in lookup:
                n_matched += 1
                ep, fr, tk, pf, row_idx = lookup[k]
                # Reload and re-normalize to verify state actually matches
                table = pq.read_table(pf, columns=["state"])
                raw_state = np.array(table["state"][row_idx].as_py(), dtype=np.float32)
                renorm = quantile_normalize(raw_state, q01, q99)
                diff = np.abs(renorm[:8] - np.array(stored_proprio[:8]))
                state_match_errors.append(diff.max())
                if diff.max() < 1e-3:
                    n_state_correct += 1
            else:
                state_match_errors.append(float("nan"))

    print(f"Examples checked:  {n_total}", flush=True)
    print(f"Parquet matched:   {n_matched}/{n_total}  ({100*n_matched/max(n_total,1):.1f}%)", flush=True)
    print(f"State exact match: {n_state_correct}/{n_matched}  (diff < 1e-3)", flush=True)
    if state_match_errors:
        errs = [e for e in state_match_errors if not np.isnan(e)]
        if errs:
            print(f"State diff stats:  mean={np.mean(errs):.6f}  max={np.max(errs):.6f}", flush=True)

    # ── Now run model to verify activations ──
    print("\n=== ACTIVATION MATCH VERIFICATION (runs model) ===", flush=True)
    print("Processing batches from data loader...", flush=True)

    # Build a lookup: state_key -> (stored_feature_id, stored_activation)
    expected = {}  # state_key -> [(feat_id, stored_activation, token_pos)]
    for feat_id in top_feature_ids:
        if feat_id not in feature_data:
            continue
        for ex in feature_data[feat_id][:args.top_k]:
            sp = ex.get("proprio_state", [])
            if sp:
                k = state_key(sp[:8])
                expected.setdefault(k, []).append((feat_id, ex.get("activation", 0), ex.get("token_pos", -1)))

    verified = []
    n_batches_to_run = 50

    data_iter = iter(data_loader)
    for batch_idx in range(n_batches_to_run):
        if not expected:
            break
        try:
            batch = next(data_iter)
        except StopIteration:
            break

        observation, actions = batch if isinstance(batch, tuple) else (batch["observation"], batch["actions"])

        # Check if any frame in this batch matches an expected example
        if hasattr(observation, "state") and observation.state is not None:
            states_np = observation.state.cpu().numpy() if isinstance(observation.state, torch.Tensor) else observation.state
            batch_keys = [state_key(quantile_normalize(states_np[i], q01, q99)[:8]) for i in range(states_np.shape[0])]
            hit_indices = [i for i, k in enumerate(batch_keys) if k in expected]
        else:
            hit_indices = []

        if not hit_indices:
            continue

        # Run model on this batch
        observation_dev = to_device(observation, model_device, model_dtype)
        actions_dev = to_device(actions, model_device, model_dtype)
        with torch.no_grad():
            _ = model(observation_dev, actions_dev)

        raw_acts = collector.get_activations(flatten=False)[args.hook_point]  # [B, seq, D]
        collector.clear_activations()
        seq_len = raw_acts.shape[1]

        for sample_i in hit_indices:
            k = batch_keys[sample_i]
            if k not in expected:
                continue
            acts_sample = raw_acts[sample_i]  # [seq, D]
            with torch.no_grad():
                feats = sae.encode(acts_sample.float().to(device))  # [seq, dict_size]

            for feat_id, stored_act, token_pos in expected[k]:
                if 0 <= token_pos < seq_len:
                    recomputed = feats[token_pos, feat_id].item()
                else:
                    # Try all token positions, take max
                    recomputed = feats[:, feat_id].max().item()

                verified.append({
                    "feat_id": feat_id,
                    "stored_activation": stored_act,
                    "recomputed_activation": recomputed,
                    "token_pos": token_pos,
                    "ratio": recomputed / max(stored_act, 1e-8),
                })
                print(f"  feature_{feat_id:5d}  token_pos={token_pos:4d}  "
                      f"stored={stored_act:8.3f}  recomputed={recomputed:8.3f}  "
                      f"ratio={recomputed/max(stored_act,1e-8):.3f}", flush=True)

            del expected[k]

        if (batch_idx + 1) % 10 == 0:
            print(f"  batch {batch_idx+1}/{n_batches_to_run}, verified so far: {len(verified)}", flush=True)

    collector.clear_hooks()

    # ── Summary ──
    print("\n" + "=" * 60, flush=True)
    print("VERIFICATION SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"State match:  {n_state_correct}/{n_matched} exact matches  "
          f"({100*n_state_correct/max(n_matched,1):.1f}%)", flush=True)
    if verified:
        ratios = [v["ratio"] for v in verified]
        corr = np.corrcoef(
            [v["stored_activation"] for v in verified],
            [v["recomputed_activation"] for v in verified]
        )[0, 1] if len(verified) > 1 else float("nan")
        close = sum(1 for v in verified if abs(v["ratio"] - 1.0) < 0.1)
        print(f"Activation verification: {len(verified)} examples checked", flush=True)
        print(f"  Pearson r (stored vs recomputed): {corr:.4f}", flush=True)
        print(f"  Within 10% of stored value:       {close}/{len(verified)}", flush=True)
        print(f"  Mean ratio (recomputed/stored):   {np.mean(ratios):.4f}", flush=True)
        print(f"\nConclusion:", flush=True)
        if corr > 0.9:
            print("  ✓ STRONG MATCH — images are causally correct.", flush=True)
        elif corr > 0.5:
            print("  ~ MODERATE MATCH — mostly correct, some noise.", flush=True)
        else:
            print("  ✗ WEAK MATCH — images may still be mismatched.", flush=True)
    else:
        print("No activation verifications completed (no batch hits in 50 batches).", flush=True)
        print("Try increasing --n_features or running with a shuffled loader.", flush=True)

    # Save results
    out = interp_dir / "verification_results.json"
    with open(out, "w") as f:
        json.dump({
            "state_matches": n_state_correct,
            "state_checked": n_matched,
            "activation_verified": verified,
        }, f, indent=2)
    print(f"\nResults saved to {out}", flush=True)


if __name__ == "__main__":
    main()
