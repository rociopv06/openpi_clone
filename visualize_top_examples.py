"""
visualize_top_examples.py
=========================
Maps SAE top-activating examples back to their source images & language prompts
and renders an HTML dashboard.

Strategy
--------
1. Build a parquet lookup table:
   - Load all episode parquet files (state 8-dim + task_index + episode_index + frame_index)
   - Apply quantile-normalization (same as training pipeline)
   - Index by rounded normalized state → (episode_idx, frame_idx, task_idx)

2. Stream the top_activating_examples.json (1.1 GB) feature-by-feature:
   - Only parse entries for the N most-active features (ranked by peak activation in .pt file)
   - Extract the 8-dim normalized state from each example

3. Match every example to a parquet frame, load its PNG bytes & prompt

4. Write an HTML file with image cards per feature
"""

import argparse
import base64
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

# ─── Config ──────────────────────────────────────────────────────────────────
LIBERO_CACHE = Path(
    "/storage/project/r-agarg35-0/rvaldes6/.cache/huggingface/lerobot/physical-intelligence/libero"
)
NORM_STATS_PATH = Path(
    "/storage/project/r-agarg35-0/rvaldes6/openpi_clone/assets/pi05_libero/"
    "physical-intelligence/libero/norm_stats.json"
)
_INTERP_BASE = Path("/storage/project/r-agarg35-0/rvaldes6/sae_interpretations")
SAE_DIRS = {
    "vision_proj":    _INTERP_BASE / "pi05_libero_vision_proj_k32_exp4",
    "lang_layer_3":   _INTERP_BASE / "pi05_libero_lang_layer_3_k32_exp4",
    "lang_layer_6":   _INTERP_BASE / "pi05_libero_lang_layer_6_k32_exp4",
    "lang_layer_9":   _INTERP_BASE / "pi05_libero_lang_layer_9_k32_exp4",
    "lang_layer_12":  _INTERP_BASE / "pi05_libero_lang_layer_12_k32_exp4",
    "lang_layer_17":  _INTERP_BASE / "pi05_libero_lang_layer_17_k32_exp4",
    "expert_layer_0": _INTERP_BASE / "pi05_libero_expert_layer_0_k32_exp4",
}
OUT_DIR = Path("/storage/project/r-agarg35-0/rvaldes6/sae_interpretations/visualizations")
TOP_N_FEATURES = 30   # how many features to visualise
TOP_K_EXAMPLES = 5    # how many top examples per feature to show
MATCH_DECIMALS = 4    # rounding precision for state fingerprint lookup
# ─────────────────────────────────────────────────────────────────────────────


# ── 1. Normalization helpers ──────────────────────────────────────────────────

def load_quantile_stats(path: Path):
    """Return q01 and q99 for the 'state' key as numpy arrays."""
    with open(path) as f:
        data = json.load(f)
    stats = data["norm_stats"]["state"]
    return np.array(stats["q01"], dtype=np.float32), np.array(stats["q99"], dtype=np.float32)


def quantile_normalize(state: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    """Apply the same quantile norm used by the training pipeline."""
    return (state - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


def state_key(norm_state: np.ndarray, decimals: int = MATCH_DECIMALS) -> tuple:
    return tuple(np.round(norm_state[:8].astype(np.float64), decimals).tolist())


# ── 2. Build parquet lookup ───────────────────────────────────────────────────

def build_parquet_lookup(cache_dir: Path, q01: np.ndarray, q99: np.ndarray):
    """
    Returns
    -------
    lookup : dict  state_key -> (episode_idx, frame_idx, task_idx)
    tasks  : dict  task_idx  -> str
    """
    print("Loading tasks …", flush=True)
    tasks = {}
    with open(cache_dir / "meta" / "tasks.jsonl") as f:
        for line in f:
            obj = json.loads(line)
            tasks[obj["task_index"]] = obj["task"]

    lookup = {}
    data_dir = cache_dir / "data"
    chunks = sorted(data_dir.iterdir())
    total = 0
    for chunk_dir in chunks:
        parquet_files = sorted(chunk_dir.glob("episode_*.parquet"))
        print(f"  {chunk_dir.name}: {len(parquet_files)} episodes", flush=True)
        for pf in parquet_files:
            table = pq.read_table(pf, columns=["state", "episode_index", "frame_index", "task_index"])
            n = len(table)
            states = np.array(table["state"].to_pylist(), dtype=np.float32)   # (n, 8)
            ep_idx = table["episode_index"].to_pylist()
            fr_idx = table["frame_index"].to_pylist()
            tk_idx = table["task_index"].to_pylist()
            norm_states = quantile_normalize(states, q01, q99)
            for i in range(n):
                k = state_key(norm_states[i])
                lookup[k] = (int(ep_idx[i]), int(fr_idx[i]), int(tk_idx[i]))
            total += n
    print(f"  Built lookup for {total} frames.", flush=True)
    return lookup, tasks


# ── 3. Streaming JSON parser ──────────────────────────────────────────────────

def _read_feature_block(path: Path, feature_ids: set):
    """
    Stream a top_activating_examples.json and yield (feature_id, examples_list)
    one at a time, parsing only the requested feature_ids.

    The file has the structure:
        {
          "feature_0": [ {…}, … ],
          "feature_1": [ {…}, … ],
          …
        }
    We detect feature key lines, accumulate the array text, then json.loads it.
    """
    found = {}
    collecting = False
    current_id = None
    depth = 0
    buffer_lines = []

    target_keys = {f'"feature_{i}"' for i in feature_ids}

    with open(path, "r") as fh:
        for line in fh:
            stripped = line.strip()

            if not collecting:
                # Look for a line like   "feature_3523": [
                for tk in target_keys:
                    if tk in stripped:
                        current_id = int(tk.strip('"').split("_")[1])
                        collecting = True
                        # The '[' may be on the same line
                        bracket_pos = stripped.find("[")
                        if bracket_pos != -1:
                            rest = stripped[bracket_pos:]
                            buffer_lines = [rest]
                            depth = rest.count("[") - rest.count("]")
                        break
            else:
                buffer_lines.append(line)
                depth += line.count("[") - line.count("]")
                if depth <= 0:
                    # We have the full array
                    raw = "".join(buffer_lines).strip().rstrip(",")
                    try:
                        parsed = json.loads(raw)
                        yield current_id, parsed
                    except json.JSONDecodeError as e:
                        print(f"  Warning: failed to parse feature_{current_id}: {e}", flush=True)
                    collecting = False
                    buffer_lines = []
                    depth = 0
                    target_keys.discard(f'"feature_{current_id}"')
                    if not target_keys:
                        return  # all done


# ── 4. Image helpers ──────────────────────────────────────────────────────────

def load_frame(cache_dir: Path, episode_idx: int, frame_idx: int):
    """Return (base_img_b64, wrist_img_b64) PNG images as base64 strings."""
    chunk = episode_idx // 1000
    pf = cache_dir / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_idx:06d}.parquet"
    table = pq.read_table(pf, columns=["image", "wrist_image"])
    row = table.slice(frame_idx, 1)
    base_bytes  = row["image"][0]["bytes"].as_py()
    wrist_bytes = row["wrist_image"][0]["bytes"].as_py()

    def to_b64(raw_bytes, size=(160, 160)):
        img = Image.open(io.BytesIO(raw_bytes)).resize(size, Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    return to_b64(base_bytes), to_b64(wrist_bytes)


# ── 5. HTML generation ────────────────────────────────────────────────────────

HTML_HEADER = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SAE Top-Activating Examples – {exp_name}</title>
<style>
  body {{ font-family: sans-serif; background: #111; color: #eee; margin: 20px; }}
  h1 {{ color: #7cf; }}
  h2 {{ color: #fa0; border-bottom: 1px solid #444; padding-bottom: 4px; }}
  .feature-card {{ background: #1e1e2e; border-radius: 10px; padding: 16px; margin: 20px 0; }}
  .example-row {{ display: flex; flex-wrap: wrap; gap: 14px; margin-top: 10px; }}
  .example-cell {{ background: #2a2a3e; border-radius: 8px; padding: 10px; width: 340px; }}
  .example-cell img {{ border-radius: 4px; }}
  .prompt {{ font-style: italic; color: #adf; margin: 6px 0; font-size: 0.9em; }}
  .act {{ color: #f90; font-size: 0.85em; }}
  .label {{ color: #888; font-size: 0.75em; }}
  .nomatch {{ color: #f66; font-size: 0.85em; }}
  .ep-info {{ color: #8a8; font-size: 0.75em; }}
</style>
</head>
<body>
<h1>SAE Top-Activating Examples &mdash; {exp_name}</h1>
<p>{n_features} features shown &nbsp;|&nbsp; up to {top_k} examples each</p>
"""

HTML_FOOTER = "</body></html>"

FEATURE_CARD_TMPL = """
<div class="feature-card">
  <h2>Feature {feat_id} &nbsp;<span style="color:#8cf;font-size:0.75em">rank #{rank}</span>
      &nbsp;<span class="act">peak activation = {peak:.2f}</span></h2>
  <div class="example-row">
    {examples_html}
  </div>
</div>
"""

EXAMPLE_CELL_TMPL = """
<div class="example-cell">
  <div class="label">Rank #{ex_rank} &nbsp; <span class="act">activation = {activation:.2f}</span></div>
  <div class="prompt">"{prompt}"</div>
  <div style="display:flex;gap:6px;margin-top:6px;">
    <div><div class="label">Base camera</div><img src="data:image/png;base64,{base_b64}" width="155"/></div>
    <div><div class="label">Wrist camera</div><img src="data:image/png;base64,{wrist_b64}" width="155"/></div>
  </div>
  <div class="ep-info">Episode {ep_idx} · Frame {fr_idx}</div>
</div>
"""

NO_MATCH_CELL_TMPL = """
<div class="example-cell">
  <div class="label">Rank #{ex_rank} &nbsp; <span class="act">activation = {activation:.2f}</span></div>
  <div class="nomatch">Could not match this example to a parquet frame.</div>
  <div class="ep-info">State: {state_preview}</div>
</div>
"""


# ── Main ──────────────────────────────────────────────────────────────────────

def process_experiment(exp_name: str, sae_dir: Path, lookup: dict, tasks: dict,
                       out_dir: Path, top_n: int = TOP_N_FEATURES, top_k: int = TOP_K_EXAMPLES):
    print(f"\n{'='*60}", flush=True)
    print(f"Processing {exp_name}", flush=True)
    print(f"{'='*60}", flush=True)

    # --- pick top N features by peak activation ---
    vals_path = sae_dir / "top_activation_values.pt"
    if not vals_path.exists():
        print(f"  Missing {vals_path}, skipping.", flush=True)
        return

    top_vals = torch.load(vals_path, map_location="cpu", weights_only=True)  # [dict_size, top_k]
    peak_per_feature = top_vals.max(dim=1).values  # [dict_size]
    ranked = peak_per_feature.argsort(descending=True)[:top_n]
    selected_features = sorted(ranked.tolist())
    peak_map = {int(fi): float(peak_per_feature[fi]) for fi in selected_features}
    rank_map  = {int(fi): r for r, fi in enumerate(ranked.tolist()[:top_n])}

    print(f"  Top {top_n} features: {selected_features[:10]} …", flush=True)

    json_path = sae_dir / "top_activating_examples.json"
    if not json_path.exists():
        print(f"  Missing {json_path}, skipping.", flush=True)
        return

    print(f"  Streaming {json_path} ({json_path.stat().st_size / 1e6:.0f} MB) …", flush=True)

    feature_data = {}
    n_found = 0
    for feat_id, examples in _read_feature_block(json_path, set(selected_features)):
        feature_data[feat_id] = examples[:top_k]
        n_found += 1
        if n_found % 10 == 0:
            print(f"    Parsed {n_found}/{top_n} features …", flush=True)
    print(f"  Parsed {n_found} features total.", flush=True)

    # --- render HTML ---
    all_cards_html = []
    match_count = 0
    miss_count = 0

    for rank_idx, feat_id in enumerate(ranked.tolist()[:top_n]):
        feat_id = int(feat_id)
        examples = feature_data.get(feat_id, [])
        peak = peak_map[feat_id]
        rank = rank_map[feat_id]

        examples_html_parts = []
        for ex_rank, ex in enumerate(examples):
            activation = ex.get("activation", 0.0)
            norm_state = ex.get("proprio_state", [])
            norm_state_arr = np.array(norm_state[:8], dtype=np.float32)
            k = state_key(norm_state_arr)

            if k in lookup:
                ep_idx, fr_idx, tk_idx = lookup[k]
                prompt = tasks.get(tk_idx, f"[task {tk_idx}]")
                try:
                    base_b64, wrist_b64 = load_frame(LIBERO_CACHE, ep_idx, fr_idx)
                    examples_html_parts.append(EXAMPLE_CELL_TMPL.format(
                        ex_rank=ex_rank + 1,
                        activation=activation,
                        prompt=prompt,
                        base_b64=base_b64,
                        wrist_b64=wrist_b64,
                        ep_idx=ep_idx,
                        fr_idx=fr_idx,
                    ))
                    match_count += 1
                except Exception as exc:
                    examples_html_parts.append(NO_MATCH_CELL_TMPL.format(
                        ex_rank=ex_rank + 1,
                        activation=activation,
                        state_preview=str(norm_state_arr.round(3)),
                    ))
                    print(f"    Img load error feat {feat_id} ex {ex_rank}: {exc}", flush=True)
                    miss_count += 1
            else:
                examples_html_parts.append(NO_MATCH_CELL_TMPL.format(
                    ex_rank=ex_rank + 1,
                    activation=activation,
                    state_preview=str(norm_state_arr.round(3)),
                ))
                miss_count += 1

        all_cards_html.append(FEATURE_CARD_TMPL.format(
            feat_id=feat_id,
            rank=rank + 1,
            peak=peak,
            examples_html="\n".join(examples_html_parts),
        ))

    html = (
        HTML_HEADER.format(exp_name=exp_name, n_features=top_n, top_k=top_k)
        + "\n".join(all_cards_html)
        + HTML_FOOTER
    )

    out_path = out_dir / f"{exp_name}_top_examples.html"
    out_path.write_text(html)
    print(f"  Wrote {out_path}", flush=True)
    print(f"  Matched {match_count} / {match_count + miss_count} examples", flush=True)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Visualize SAE top-activating examples with images and prompts")
    parser.add_argument("--experiments", nargs="+", default=list(SAE_DIRS.keys()),
                        choices=list(SAE_DIRS.keys()), help="Which experiments to process")
    parser.add_argument("--top_n_features", type=int, default=TOP_N_FEATURES)
    parser.add_argument("--top_k_examples", type=int, default=TOP_K_EXAMPLES)
    parser.add_argument("--out_dir", type=str, default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load normalization stats
    print("Loading normalization stats …", flush=True)
    q01, q99 = load_quantile_stats(NORM_STATS_PATH)
    print(f"  state q01: {q01}")
    print(f"  state q99: {q99}")

    # Build parquet lookup (shared across experiments)
    print("\nBuilding parquet state lookup (this may take a few minutes) …", flush=True)
    lookup, tasks = build_parquet_lookup(LIBERO_CACHE, q01, q99)
    print(f"Total unique state keys in lookup: {len(lookup)}", flush=True)
    print(f"Total tasks: {len(tasks)}", flush=True)

    # Process each experiment
    generated = []
    for exp_name in args.experiments:
        sae_dir = SAE_DIRS[exp_name]
        path = process_experiment(
            exp_name, sae_dir, lookup, tasks, out_dir,
            top_n=args.top_n_features, top_k=args.top_k_examples,
        )
        if path:
            generated.append(path)

    print("\n" + "="*60)
    print("DONE. Generated files:")
    for p in generated:
        print(f"  {p}")


if __name__ == "__main__":
    main()
