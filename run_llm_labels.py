"""
run_llm_labels.py
=================
Read label_prompts.jsonl produced by compute_ms_and_label.py and call
the Anthropic Messages API via plain HTTP (requests only, no SDK).

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python run_llm_labels.py

    # Specific SAE or path:
    python run_llm_labels.py --sae_name pi05_libero_lang_layer_9_k32_exp4
    python run_llm_labels.py --prompts_path /path/to/label_prompts.jsonl

    # Dry-run (print first prompt, no API calls):
    python run_llm_labels.py --dry_run

Options:
    --sae_name      SAE checkpoint name (default: pi05_libero_lang_layer_9_k32_exp4)
    --prompts_path  Override path to label_prompts.jsonl
    --output_path   Override path for feature_labels.json
    --model         Claude model to use (default: claude-opus-4-6)
    --max_tokens    Max tokens per response (default: 300)
    --dry_run       Print the first prompt and exit, no API calls
    --no_resume     Re-label all features even if feature_labels.json exists
    --delay         Seconds between requests (default: 0.5, avoid rate limits)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path("/storage/project/r-agarg35-0/rvaldes6")
INTERP_DIR  = PROJECT_DIR / "sae_interpretations"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sae_name",     default="pi05_libero_lang_layer_9_k32_exp4")
    p.add_argument("--prompts_path", default=None)
    p.add_argument("--output_path",  default=None)
    p.add_argument("--model",        default="claude-opus-4-6")
    p.add_argument("--max_tokens",   type=int, default=300)
    p.add_argument("--dry_run",      action="store_true")
    p.add_argument("--no_resume",    action="store_true")
    p.add_argument("--delay",        type=float, default=0.5)
    return p.parse_args()

args = parse_args()

SAE_DIR       = INTERP_DIR / args.sae_name
prompts_path  = Path(args.prompts_path) if args.prompts_path else SAE_DIR / "label_prompts.jsonl"
output_path   = Path(args.output_path)  if args.output_path  else SAE_DIR / "feature_labels.json"

# ─── Validate ─────────────────────────────────────────────────────────────────
if not prompts_path.exists():
    print(f"ERROR: prompts file not found: {prompts_path}")
    print("Wait for compute_ms_and_label.py (SLURM job) to finish first.")
    sys.exit(1)

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
if not API_KEY and not args.dry_run:
    print("ERROR: ANTHROPIC_API_KEY env var not set.")
    print("  export ANTHROPIC_API_KEY=sk-ant-...")
    sys.exit(1)

# ─── Load existing labels (resume) ────────────────────────────────────────────
existing_labels = {}
if output_path.exists() and not args.no_resume:
    with open(output_path) as f:
        existing_labels = json.load(f)
    # Keys may be strings or ints
    existing_labels = {int(k): v for k, v in existing_labels.items()}
    print(f"Resuming: {len(existing_labels)} features already labeled.")

# ─── Load prompts ─────────────────────────────────────────────────────────────
records = []
with open(prompts_path) as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

print(f"Prompts: {len(records)} features in {prompts_path.name}")
print(f"Model:   {args.model}")

to_do = [r for r in records if int(r["feature_id"]) not in existing_labels]
print(f"To label: {len(to_do)} (skipping {len(existing_labels)} already done)")

# ─── Dry run ──────────────────────────────────────────────────────────────────
if args.dry_run:
    rec = records[0]
    print(f"\n{'='*60}")
    print(f"FEATURE {rec['feature_id']}  MS={rec['ms_score']:.3f}  fires={rec['fires']}")
    print(f"SYSTEM:\n{rec['system'][:300]}...")
    print(f"\nUSER TEXT:\n{rec['user_text']}")
    print(f"\nImages: {len(rec['images_b64'])} base64 JPEGs ({sum(len(b) for b in rec['images_b64'])//1024} KB total)")
    print(f"{'='*60}")
    print("Dry run — no API calls made.")
    sys.exit(0)

# ─── API call (raw HTTP, no SDK) ───────────────────────────────────────────────
API_URL = "https://api.anthropic.com/v1/messages"
HEADERS = {
    "x-api-key":         API_KEY,
    "anthropic-version": "2023-06-01",
    "content-type":      "application/json",
}

def call_claude(record: dict) -> dict:
    """Call Anthropic Messages API; return parsed JSON label dict."""
    content = []
    for b64 in record["images_b64"]:
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}
        })
    content.append({"type": "text", "text": record["user_text"]})

    payload = {
        "model":      args.model,
        "max_tokens": args.max_tokens,
        "system":     record["system"],
        "messages":   [{"role": "user", "content": content}],
    }

    for attempt in range(3):
        try:
            resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
            if resp.status_code == 529:  # overloaded
                wait = 30 * (attempt + 1)
                print(f"  API overloaded, waiting {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code == 429:  # rate limit
                wait = 60 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            raw = resp.json()["content"][0]["text"].strip()
            # strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"label": resp.json()["content"][0]["text"], "confidence": "low", "type": "unclear"}
        except Exception as e:
            if attempt == 2:
                return {"label": f"ERROR: {e}", "confidence": "low", "type": "unclear"}
            time.sleep(5)
    return {"label": "ERROR: max retries", "confidence": "low", "type": "unclear"}

# ─── Main loop ────────────────────────────────────────────────────────────────
labels = dict(existing_labels)

def save():
    with open(output_path, "w") as f:
        json.dump({str(k): v for k, v in labels.items()}, f, indent=2)

try:
    for rec in tqdm(to_do, desc="LLM labeling"):
        feat_id = int(rec["feature_id"])
        result  = call_claude(rec)
        labels[feat_id] = {
            "ms_score":    rec["ms_score"],
            "fires":       rec["fires"],
            "action_corr": rec["action_corr"],
            "label":       result.get("label", ""),
            "confidence":  result.get("confidence", "low"),
            "type":        result.get("type", "unclear"),
        }
        if len(labels) % 20 == 0:
            save()
        time.sleep(args.delay)
except KeyboardInterrupt:
    print("\nInterrupted — saving progress...")

save()
print(f"\nDone. {len(labels)} labels saved to {output_path}")

# ─── Summary ──────────────────────────────────────────────────────────────────
type_counts = {}
conf_counts = {}
for v in labels.values():
    type_counts[v.get("type", "?")] = type_counts.get(v.get("type", "?"), 0) + 1
    conf_counts[v.get("confidence", "?")] = conf_counts.get(v.get("confidence", "?"), 0) + 1

print("\nFeature type breakdown:")
for t, n in sorted(type_counts.items(), key=lambda x: -x[1]):
    print(f"  {t:20s} {n:4d}")

print("\nConfidence breakdown:")
for c, n in sorted(conf_counts.items(), key=lambda x: -x[1]):
    print(f"  {c:10s} {n:4d}")

print(f"\nTop labels (by MS score):")
top = sorted(labels.items(), key=lambda x: -x[1].get("ms_score", 0))[:10]
for fid, v in top:
    print(f"  F{fid:5d}  MS={v['ms_score']:.3f}  [{v['confidence']:6s}]  {v['label']}")
