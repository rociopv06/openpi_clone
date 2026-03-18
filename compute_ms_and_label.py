"""
compute_ms_and_label.py
========================
Two-phase pipeline for lang_layer_9 (generalises to any SAE checkpoint).

Phase 1 — MonoSemanticity Score (MS)
    For every alive SAE feature:
      • Pull top-K activating frames from top_activating_examples.json
      • Look up images via state-fingerprint matching (same as visualise_top_examples.py)
      • Compute DINOv2 activation-weighted pairwise cosine similarity → MS score
      • Save: ms_scores.json  {feature_id: ms_score}

Phase 2 — LLM Feature Labeling
    For features with MS >= MS_THRESHOLD:
      • Build a prompt containing:
          - top-3 images (base64)
          - task name for each example
          - action pattern summary (which dims dominate)
          - proprio state fingerprint
      • Write prompts to a JSONL file ready to send to any LLM API
        (claude, gpt-4o, gemini — your choice)
      • Save: label_prompts.jsonl  {feature_id, ms_score, prompt, images_b64}
      • If LABEL_WITH_CLAUDE=1 env var is set, also calls Claude API and saves labels

Usage:
    # Phase 1 + 2 (write prompts only, no API calls):
    python compute_ms_and_label.py

    # Phase 1 + 2 + actual labeling (set your API key first):
    ANTHROPIC_API_KEY=sk-... LABEL_WITH_CLAUDE=1 python compute_ms_and_label.py

    # Specific SAE:
    python compute_ms_and_label.py --sae_name pi05_libero_lang_layer_9_k32_exp4
"""

import argparse
import base64
import io
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

# ─── Config ──────────────────────────────────────────────────────────────────
PROJECT_DIR  = Path("/storage/project/r-agarg35-0/rvaldes6")
OPENPI_DIR   = PROJECT_DIR / "openpi_clone"
LIBERO_CACHE = PROJECT_DIR / ".cache/huggingface/lerobot/physical-intelligence/libero"
NORM_STATS   = OPENPI_DIR / "assets/pi05_libero/physical-intelligence/libero/norm_stats.json"
INTERP_DIR   = PROJECT_DIR / "sae_interpretations"

TOP_JSON_EXAMPLES = 10   # how many examples to pull per feature for MS
MS_EXAMPLES       = 5    # how many to use for MS computation (top N by activation)
LABEL_EXAMPLES    = 3    # how many images to send to LLM
MS_THRESHOLD      = 0.45 # only label features above this
ACTION_NAMES      = ["X", "Y", "Z", "Roll", "Pitch", "Yaw", "Gripper"]
MATCH_DECIMALS    = 4

# ─── Args ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sae_name", default="pi05_libero_lang_layer_9_k32_exp4")
    p.add_argument("--ms_threshold", type=float, default=MS_THRESHOLD)
    p.add_argument("--top_n", type=int, default=TOP_JSON_EXAMPLES,
                   help="Examples to pull from JSON per feature")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--skip_ms", action="store_true", help="Skip MS computation, load existing ms_scores.json")
    return p.parse_args()

args = parse_args()
SAE_DIR = INTERP_DIR / args.sae_name
OUT_DIR = SAE_DIR
print(f"SAE: {args.sae_name}")
print(f"Device: {args.device}")

# ─── Norm stats ───────────────────────────────────────────────────────────────
with open(NORM_STATS) as f:
    ns = json.load(f)["norm_stats"]["state"]
q01 = np.array(ns["q01"], dtype=np.float32)
q99 = np.array(ns["q99"], dtype=np.float32)

def quantile_norm(state):
    return (state - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0

def state_key(norm_state):
    return tuple(np.round(np.array(norm_state[:8], dtype=np.float64), MATCH_DECIMALS).tolist())

# ─── Build parquet lookup (state -> episode/frame/task) ───────────────────────
print("\nBuilding parquet lookup...", flush=True)
with open(LIBERO_CACHE / "meta" / "tasks.jsonl") as f:
    tasks = {obj["task_index"]: obj["task"] for obj in (json.loads(l) for l in f)}

state_lookup = {}   # state_key -> (ep, fr, task_idx)
ep_to_pf     = {}   # ep -> parquet path

data_dir = LIBERO_CACHE / "data"
for chunk_dir in sorted(data_dir.iterdir()):
    for pf in sorted(chunk_dir.glob("episode_*.parquet")):
        table = pq.read_table(pf, columns=["state","episode_index","frame_index","task_index"])
        states  = np.array(table["state"].to_pylist(), dtype=np.float32)
        ep_idxs = table["episode_index"].to_pylist()
        fr_idxs = table["frame_index"].to_pylist()
        tk_idxs = table["task_index"].to_pylist()
        norm_s  = quantile_norm(states)
        for i in range(len(table)):
            k  = state_key(norm_s[i])
            ep = int(ep_idxs[i])
            state_lookup[k] = (ep, int(fr_idxs[i]), int(tk_idxs[i]))
            ep_to_pf[ep] = pf

print(f"  Indexed {len(state_lookup):,} frames", flush=True)

# ─── Image loading ─────────────────────────────────────────────────────────────
_pf_cache = {}  # pf_str -> {fr -> img_bytes}

def load_frame_image(ep: int, fr: int) -> bytes | None:
    pf = ep_to_pf.get(ep)
    if pf is None:
        return None
    key = str(pf)
    if key not in _pf_cache:
        table = pq.read_table(pf, columns=["frame_index", "image"])
        _pf_cache[key] = {
            int(table["frame_index"][i].as_py()): table["image"][i].as_py()
            for i in range(len(table))
        }
    raw = _pf_cache[key].get(fr)
    if isinstance(raw, dict):
        raw = raw.get("bytes")
    return raw

def bytes_to_pil(raw: bytes) -> Image.Image | None:
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return None

def img_to_b64(img: Image.Image, size=(224, 224)) -> str:
    img = img.resize(size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()

def clear_img_cache():
    _pf_cache.clear()

def lookup_example(ex: dict):
    """Returns (ep, fr, task_str, pil_image) or None."""
    prop = ex.get("proprio_state", [])
    k = state_key(prop[:8])
    hit = state_lookup.get(k)
    if hit is None:
        return None
    ep, fr, tk = hit
    raw = load_frame_image(ep, fr)
    if raw is None:
        return None
    img = bytes_to_pil(raw)
    return ep, fr, tasks.get(tk, "unknown task"), img

# ─── Load fire counts + alive features ────────────────────────────────────────
fire_counts   = torch.load(SAE_DIR / "feature_fire_counts.pt", map_location="cpu", weights_only=True)
alive_features = fire_counts.nonzero().squeeze().tolist()
if isinstance(alive_features, int):
    alive_features = [alive_features]
print(f"\nAlive features: {len(alive_features)}")

# ─── Load action correlations ─────────────────────────────────────────────────
with open(SAE_DIR / "feature_action_correlations.json") as f:
    corr_data = json.load(f)["action_correlations"]

feature_action_summary = {}  # feat_id -> "best sign+dim (r=0.XXX)" or "no action corr"
for dim_i, name in enumerate(ACTION_NAMES):
    key = f"action_dim_{dim_i}"
    if key not in corr_data:
        continue
    for e in corr_data[key].get("top_positive", []):
        fid, c = e["feature"], e["correlation"]
        if fid not in feature_action_summary or abs(c) > abs(feature_action_summary[fid][1]):
            feature_action_summary[fid] = (f"+{name}", c)
    for e in corr_data[key].get("top_negative", []):
        fid, c = e["feature"], e["correlation"]
        if fid not in feature_action_summary or abs(c) > abs(feature_action_summary[fid][1]):
            feature_action_summary[fid] = (f"-{name}", c)

# ─── Stream top_activating_examples.json ─────────────────────────────────────
print(f"\nStreaming top_activating_examples.json for {len(alive_features)} features...", flush=True)

target_set = set(alive_features)
feature_examples = {}   # feat_id -> list of example dicts

json_path = SAE_DIR / "top_activating_examples.json"
current_feature = None
bracket_depth   = 0
collecting      = False
buffer          = ""

with open(json_path) as f:
    for line in tqdm(f, desc="Streaming JSON", unit=" lines", mininterval=5.0):
        if not collecting:
            for feat_id in target_set:
                key = f'"feature_{feat_id}"'
                if key in line:
                    stripped = line.replace(' ', '').replace('\n', '')
                    if f'"{feat_id}"' + ':' in stripped or key + ':' in stripped:
                        current_feature = feat_id
                        idx = line.find('[')
                        if idx >= 0:
                            collecting = True
                            buffer = line[idx:]
                            bracket_depth = buffer.count('[') - buffer.count(']')
                        break
        else:
            buffer += line
            bracket_depth += line.count('[') - line.count(']')
            if bracket_depth <= 0:
                try:
                    data = json.loads(buffer.strip().rstrip(','))
                    feature_examples[current_feature] = data[:args.top_n]
                except Exception:
                    pass
                collecting = False
                current_feature = None
                buffer = ""
        if len(feature_examples) == len(target_set):
            break

print(f"  Extracted examples for {len(feature_examples)} features")

# ─── Phase 1: DINOv2 MS Scores ────────────────────────────────────────────────
ms_scores = {}

if args.skip_ms and (OUT_DIR / "ms_scores.json").exists():
    print("\nLoading existing ms_scores.json...")
    with open(OUT_DIR / "ms_scores.json") as f:
        ms_scores = {int(k): v for k, v in json.load(f).items()}
else:
    print("\nLoading DINOv2...", flush=True)
    from transformers import AutoImageProcessor, AutoModel

    dinov2_processor = AutoImageProcessor.from_pretrained(
        "facebook/dinov2-base",
        cache_dir=str(PROJECT_DIR / ".cache/huggingface")
    )
    dinov2 = AutoModel.from_pretrained(
        "facebook/dinov2-base",
        cache_dir=str(PROJECT_DIR / ".cache/huggingface")
    ).to(args.device)
    dinov2.eval()
    print("  DINOv2 loaded", flush=True)

    @torch.no_grad()
    def dinov2_embed(pil_imgs: list) -> torch.Tensor:
        inputs = dinov2_processor(images=pil_imgs, return_tensors="pt").to(args.device)
        out    = dinov2(**inputs).last_hidden_state[:, 0]  # CLS token
        return F.normalize(out, dim=-1)

    def compute_ms(examples: list) -> float:
        """Compute activation-weighted pairwise cosine sim (MS score)."""
        resolved = []
        for ex in examples[:MS_EXAMPLES]:
            hit = lookup_example(ex)
            if hit and hit[3] is not None:
                resolved.append((hit[3], float(ex.get("activation", 1.0))))
        if len(resolved) < 2:
            return 0.0

        imgs  = [r[0] for r in resolved]
        acts  = torch.tensor([r[1] for r in resolved], dtype=torch.float32)
        # min-max normalize activations
        acts  = (acts - acts.min()) / (acts.max() - acts.min() + 1e-8)

        embs  = dinov2_embed(imgs)                   # (N, d)
        S     = embs @ embs.T                        # (N, N) cosine sim
        R     = acts.unsqueeze(0) * acts.unsqueeze(1) # (N, N) relevance

        mask  = torch.triu(torch.ones_like(S, dtype=torch.bool), diagonal=1)
        denom = R[mask].sum()
        if denom < 1e-8:
            return 0.0
        return (R[mask] * S[mask].to(R.device)).sum().item() / denom.item()

    print(f"\nComputing MS scores for {len(alive_features)} features...", flush=True)
    for feat_id in tqdm(alive_features, desc="MS scores"):
        examples = feature_examples.get(feat_id, [])
        if not examples:
            ms_scores[feat_id] = 0.0
            continue
        ms_scores[feat_id] = compute_ms(examples)
        clear_img_cache()

    with open(OUT_DIR / "ms_scores.json", "w") as f:
        json.dump(ms_scores, f, indent=2)
    print(f"  Saved ms_scores.json")

# ─── MS summary ───────────────────────────────────────────────────────────────
ms_vals  = list(ms_scores.values())
above    = sum(1 for v in ms_vals if v >= args.ms_threshold)
print(f"\nMS score summary:")
print(f"  mean={np.mean(ms_vals):.3f}  median={np.median(ms_vals):.3f}  "
      f"max={max(ms_vals):.3f}  min={min(ms_vals):.3f}")
print(f"  above threshold ({args.ms_threshold}): {above} / {len(ms_vals)} features")

# Save ranked list
ranked_features = sorted(ms_scores.items(), key=lambda x: -x[1])
with open(OUT_DIR / "ms_ranked_features.json", "w") as f:
    json.dump([{"feature_id": int(fid), "ms": float(ms),
                "fires": int(fire_counts[fid].item()),
                "action_corr": feature_action_summary.get(fid, ("none", 0.0))[0],
                "action_corr_r": round(feature_action_summary.get(fid, ("none", 0.0))[1], 4)}
               for fid, ms in ranked_features], f, indent=2)
print(f"  Saved ms_ranked_features.json")

# ─── Phase 2: Build LLM label prompts ─────────────────────────────────────────
print(f"\nBuilding LLM label prompts for features with MS >= {args.ms_threshold}...", flush=True)

to_label = [(fid, ms) for fid, ms in ranked_features if ms >= args.ms_threshold]
print(f"  Features to label: {len(to_label)}")

SYSTEM_PROMPT = """You are labeling internal features of a Sparse Autoencoder (SAE) trained on PI0.5,
a robot manipulation policy. The SAE decomposes the model's internal activations at a middle vision-language layer
into sparse, interpretable features. Each feature may encode a visual scene, robot state, task context, or motion primitive.

Your job: given a few images where this feature fired most strongly, plus metadata about the robot state and predicted action,
write a SHORT label (5-10 words) for what this feature detects.

Focus on what the images have in common — object present, scene type, robot posture, or action phase.
Be specific. Prefer concrete descriptions over vague ones.
Examples of good labels:
  "arm reaching toward object on table"
  "gripper fully closed, grasping phase"
  "cup visible in upper-left scene area"
  "task: stacking, pre-grasp positioning"
  "horizontal surface with small object"
  "wrist rotating counterclockwise"
Also output a confidence: high / medium / low, and whether the feature seems to encode:
  visual_scene | robot_state | action_primitive | task_context | mixed | unclear

Respond ONLY as JSON: {"label": "...", "confidence": "high|medium|low", "type": "..."}"""

def build_user_prompt(feat_id: int, examples: list, ms: float) -> tuple[str, list]:
    """Returns (text_prompt, list_of_b64_images)."""
    fires = int(fire_counts[feat_id].item())
    ac    = feature_action_summary.get(feat_id)
    ac_str = f"{ac[0]} (r={ac[1]:.3f})" if ac else "no strong action correlation"

    images_b64 = []
    example_strs = []

    for i, ex in enumerate(examples[:LABEL_EXAMPLES]):
        hit = lookup_example(ex)
        act = float(ex.get("activation", 0))
        prop = ex.get("proprio_state", [])[:8]
        act_seq = ex.get("actions", [[]])[0][:7] if ex.get("actions") else []

        prop_str = ", ".join(f"J{j+1}={v:.2f}" for j, v in enumerate(prop[:6]))
        act_str  = ", ".join(f"{n}={v:+.2f}" for n, v in zip(ACTION_NAMES, act_seq))

        if hit and hit[3]:
            ep, fr, task, img = hit
            b64 = img_to_b64(img)
            images_b64.append(b64)
            example_strs.append(
                f"Example {i+1} (activation={act:.1f}):\n"
                f"  Task: {task}\n"
                f"  Robot joints: {prop_str}\n"
                f"  Predicted action: {act_str}"
            )
        else:
            example_strs.append(
                f"Example {i+1} (activation={act:.1f}, no image matched):\n"
                f"  Robot joints: {prop_str}\n"
                f"  Predicted action: {act_str}"
            )

    text = (
        f"Feature {feat_id}\n"
        f"  Fires: {fires:,} times across LIBERO dataset\n"
        f"  MS score: {ms:.3f} (monosemanticity — 1.0 = all activating scenes look identical)\n"
        f"  Strongest action correlation: {ac_str}\n\n"
        + "\n\n".join(example_strs)
        + "\n\nThe images above show the top robot camera views where this feature fired most strongly.\n"
          "What single concept do they have in common?"
    )
    return text, images_b64

print("  Writing label_prompts.jsonl...", flush=True)
prompts_path = OUT_DIR / "label_prompts.jsonl"
with open(prompts_path, "w") as out_f:
    for feat_id, ms in tqdm(to_label, desc="Building prompts"):
        examples = feature_examples.get(feat_id, [])
        if not examples:
            continue
        text_prompt, images_b64 = build_user_prompt(feat_id, examples, ms)
        record = {
            "feature_id":   int(feat_id),
            "ms_score":     float(ms),
            "fires":        int(fire_counts[feat_id].item()),
            "action_corr":  feature_action_summary.get(feat_id, ("none", 0.0))[0],
            "system":       SYSTEM_PROMPT,
            "user_text":    text_prompt,
            "images_b64":   images_b64,  # list of JPEG base64 strings
        }
        out_f.write(json.dumps(record) + "\n")
        clear_img_cache()

print(f"  Saved {len(to_label)} prompts to label_prompts.jsonl")

# ─── Optional: call Claude API directly ───────────────────────────────────────
LABEL_WITH_CLAUDE = os.environ.get("LABEL_WITH_CLAUDE", "0") == "1"
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

if LABEL_WITH_CLAUDE and ANTHROPIC_API_KEY:
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    labels = {}
    with open(prompts_path) as f:
        records = [json.loads(l) for l in f]

    for rec in tqdm(records, desc="LLM labeling"):
        content = []
        for b64 in rec["images_b64"]:
            content.append({"type": "image", "source": {"type": "base64",
                            "media_type": "image/jpeg", "data": b64}})
        content.append({"type": "text", "text": rec["user_text"]})

        try:
            resp = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=200,
                system=rec["system"],
                messages=[{"role": "user", "content": content}]
            )
            raw = resp.content[0].text.strip()
            result = json.loads(raw)
        except Exception as e:
            result = {"label": f"ERROR: {e}", "confidence": "low", "type": "unclear"}

        labels[rec["feature_id"]] = {
            "ms_score": rec["ms_score"],
            "label":    result.get("label", ""),
            "confidence": result.get("confidence", "low"),
            "type":     result.get("type", "unclear"),
            "action_corr": rec["action_corr"],
            "fires":    rec["fires"],
        }

    with open(OUT_DIR / "feature_labels.json", "w") as f:
        json.dump(labels, f, indent=2)
    print(f"\nSaved {len(labels)} labels to feature_labels.json")
else:
    print("\nLLM labeling not run (set LABEL_WITH_CLAUDE=1 and ANTHROPIC_API_KEY to enable).")
    print("Prompts are ready in label_prompts.jsonl — pipe to any LLM API.")

print("\n=== Done ===")
print(f"Outputs in: {OUT_DIR}")
print(f"  ms_scores.json          — MS score per feature")
print(f"  ms_ranked_features.json — features ranked by MS score")
print(f"  label_prompts.jsonl     — LLM-ready prompts (one per line)")
if LABEL_WITH_CLAUDE:
    print(f"  feature_labels.json     — LLM-generated labels")
