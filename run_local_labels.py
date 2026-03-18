"""
run_local_labels.py
===================
Label SAE features using Llama-3.2-11B-Vision-Instruct (local, no API key needed).
Reads label_prompts.jsonl produced by compute_ms_and_label.py.
Outputs feature_labels.json in the same format as run_llm_labels.py.

Usage:
    python run_local_labels.py --sae_name pi05_libero_lang_layer_9_k32_exp4
    python run_local_labels.py --sae_name pi05_libero_lang_layer_9_k32_exp4 --dry_run
"""

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

PROJECT_DIR = Path("/storage/project/r-agarg35-0/rvaldes6")
INTERP_DIR  = PROJECT_DIR / "sae_interpretations"
HF_CACHE    = PROJECT_DIR / ".cache/huggingface/hub"
DEFAULT_MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sae_name",      default="pi05_libero_lang_layer_9_k32_exp4")
    p.add_argument("--prompts_path",  default=None)
    p.add_argument("--output_path",   default=None)
    p.add_argument("--model",         default=DEFAULT_MODEL)
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--no_resume",     action="store_true")
    p.add_argument("--dry_run",       action="store_true")
    return p.parse_args()


args = parse_args()

SAE_DIR      = INTERP_DIR / args.sae_name
prompts_path = Path(args.prompts_path) if args.prompts_path else SAE_DIR / "label_prompts.jsonl"
output_path  = Path(args.output_path)  if args.output_path  else SAE_DIR / "feature_labels.json"

if not prompts_path.exists():
    print(f"ERROR: {prompts_path} not found. Run compute_ms_and_label.py first.")
    sys.exit(1)

# ── Load prompts ──────────────────────────────────────────────────────────────
records = []
with open(prompts_path) as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))
print(f"Loaded {len(records)} prompts from {prompts_path}")

# ── Resume ────────────────────────────────────────────────────────────────────
existing = {}
if output_path.exists() and not args.no_resume:
    with open(output_path) as f:
        existing = {int(k): v for k, v in json.load(f).items()}
    print(f"Resuming: {len(existing)} already labeled.")

to_do = [r for r in records if int(r["feature_id"]) not in existing]
print(f"To label: {len(to_do)}")

if args.dry_run:
    rec = records[0]
    print(f"\nDRY RUN — first record:")
    print(f"  feature_id: {rec['feature_id']}")
    print(f"  ms_score:   {rec['ms_score']:.3f}")
    print(f"  images:     {len(rec['images_b64'])}")
    print(f"  user_text:\n{rec['user_text'][:500]}")
    sys.exit(0)

if not to_do:
    print("Nothing to do.")
    sys.exit(0)

# ── Load model ────────────────────────────────────────────────────────────────
print(f"\nLoading {args.model} ...")
from transformers import MllamaForConditionalGeneration, AutoProcessor

model = MllamaForConditionalGeneration.from_pretrained(
    args.model,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    cache_dir=str(HF_CACHE),
)
processor = AutoProcessor.from_pretrained(
    args.model,
    cache_dir=str(HF_CACHE),
)
model.eval()
print("Model loaded.")

SYSTEM_PROMPT = (
    "You are a neural network interpretability assistant. "
    "You will be shown images and metadata from a Sparse Autoencoder (SAE) feature "
    "trained on a robot manipulation policy (PI0 on LIBERO tasks). "
    "Your job is to produce a short label for what this feature detects. "
    "Respond with valid JSON only, no other text: "
    '{\"label\": \"<5-10 word description>\", '
    '\"confidence\": \"high|medium|low\", '
    '\"type\": \"visual|motor|task|state|unclear\"}'
)


def b64_to_pil(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def label_feature(rec: dict) -> dict:
    images = [b64_to_pil(b) for b in rec["images_b64"]]

    # Llama 3.2 Vision uses <|image|> tokens in the text
    image_tags = "".join(["<|image|>\n"] * len(images))
    user_text = f"{image_tags}{rec['user_text']}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_text},
    ]

    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        images=images if images else None,
        text=input_text,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    # Strip input tokens
    n_input = inputs["input_ids"].shape[-1]
    raw = processor.decode(out_ids[0][n_input:], skip_special_tokens=True).strip()

    try:
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"label": raw[:200], "confidence": "low", "type": "unclear"}


# ── Main loop ─────────────────────────────────────────────────────────────────
labels = dict(existing)

def save():
    with open(output_path, "w") as f:
        json.dump({str(k): v for k, v in labels.items()}, f, indent=2)

try:
    for rec in tqdm(to_do, desc="Labeling"):
        feat_id = int(rec["feature_id"])
        result  = label_feature(rec)
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
except KeyboardInterrupt:
    print("\nInterrupted — saving...")

save()
print(f"\nDone. {len(labels)} labels saved to {output_path}")

# ── Summary ───────────────────────────────────────────────────────────────────
type_counts = {}
for v in labels.values():
    t = v.get("type", "?")
    type_counts[t] = type_counts.get(t, 0) + 1

print("\nFeature type breakdown:")
for t, n in sorted(type_counts.items(), key=lambda x: -x[1]):
    print(f"  {t:20s} {n}")

print("\nTop labels by MS score:")
top = sorted(labels.items(), key=lambda x: -x[1].get("ms_score", 0))[:10]
for fid, v in top:
    print(f"  F{fid:5d}  MS={v['ms_score']:.3f}  [{v['confidence']:6s}]  {v['label']}")
